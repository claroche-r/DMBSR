import numpy as np
import kornia
from skimage.io import imsave
from scipy import sparse as sps
import torch
from matplotlib import pyplot as plt


class TrajectoryLengthLoss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(TrajectoryLengthLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, positions):
        diff = positions[:,1:,]-positions[:,:-1,]
        distances = torch.norm(diff, p=2, dim=2)
        loss = torch.mean(distances)
        return loss

class CurvatureLoss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(CurvatureLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, positions):
        diff_vectors = positions[:,1:,]-positions[:,:-1,]
        diff_vector_norm = torch.norm(diff_vectors, p=2, dim=2)
        dot_products = torch.sum(diff_vectors[:,1:,:] * diff_vectors[:,:-1,:], dim=2)
        cos_angles = dot_products/(diff_vector_norm[:,1:] * diff_vector_norm[:,:-1])
        angles = torch.acos(cos_angles)
        curvatures = angles/(torch.norm(diff_vectors[:,1:]/2, p=2, dim=2) + torch.norm(diff_vectors[:,:-1]/2, p=2, dim=2))
        loss = torch.sum(torch.abs(curvatures))
        return loss

class SymmetricLoss(torch.nn.Module):
    def __init__(self, loss_type='L1'):
        super(SymmetricLoss,self).__init__()
        self.loss_type = 'L1'
        
    def forward(self,found_pos, gt_pos):
        N_pose = len(found_pos)
        middle_index = N_pose//2
        symmetric_loss = torch.Tensor([0.]).to(found_pos.device)
        for i in range(middle_index):
            avg_loss = (found_pos[i] + found_pos[N_pose-1 - i]) - (gt_pos[i] + gt_pos[N_pose-1 - i])
            diff_loss = (found_pos[i] - found_pos[N_pose-1 - i]) - (gt_pos[i] - gt_pos[N_pose-1 - i])
            if self.loss_type=='L1':
                symmetric_loss += torch.sum(torch.abs(avg_loss) + torch.abs(diff_loss) )
        symmetric_loss += torch.sum(torch.abs(found_pos[middle_index-1] - gt_pos[middle_index-1]))
        return symmetric_loss
                             
class Kernels2DLoss(torch.nn.Module):

    def __init__(self, loss_type='L2', padding=0):
        super(Kernels2DLoss, self).__init__()
        self.eps = 1e-9
        self.padding = padding
        self.loss_type = loss_type

    def forward(self, found_positions, gt_positions, img_shape, intrinsics):
        M, N, C = img_shape
        N_pose = len(found_positions)
        x = torch.arange(-1, 1, 2./N)
        y = torch.arange(-1, 1, 2./M)
        xx, yy = torch.meshgrid(x, y)
        loc = torch.vstack([(xx.T).flatten(), (yy.T).flatten(), torch.ones_like(xx).flatten()]).to(found_positions.device)

        kernel_loss = torch.Tensor([0.]).to(found_positions.device)
        for n in range(N_pose):
            found_homo = compute_homography_from_position(found_positions[n], intrinsics)
            gt_homo = compute_homography_from_position(gt_positions[n], intrinsics)

            found_homo_inv = torch.linalg.inv(found_homo)
            gt_homo_inv = torch.linalg.inv(gt_homo)

            proj_found = found_homo_inv @ loc  # 3x
            loc2D_found = proj_found[0,0,:2, :] / (proj_found[0,0, 2, :] + self.eps)  # 2x(MxN)

            proj_gt = gt_homo_inv @ loc  # 3x(MxN)
            loc2D_gt = proj_gt[0,0,:2, :] / (proj_gt[0,0, 2, :] + self.eps)  # 2x(MxN)

            if self.loss_type == 'L2':
                kernel_loss += torch.mean((loc2D_found - loc2D_gt)**2)
            elif self.loss_type == 'L1':
                kernel_loss += torch.sum(torch.abs(loc2D_found - loc2D_gt))
            elif self.loss_type == 'maxL2':
                kernel_loss_n = torch.sum((loc2D_found - loc2D_gt)**2, dim=0)
                kernel_loss = torch.max(kernel_loss, kernel_loss_n)
                kernel_loss = torch.mean(kernel_loss)

            elif self.loss_type == 'maxL1':
                kernel_loss_n = torch.sum(torch.abs(loc2D_found - loc2D_gt), dim=0)
                kernel_loss = torch.max(kernel_loss, kernel_loss_n)
                kernel_loss = torch.mean(kernel_loss)

        return kernel_loss



def reblur_homographies(sharp_image, camera_positions, intrinsics, forward = True):
    '''
    sharp_image: BxCxHxW
    camera_positions: BxPx3
    intrinsics: 3x3
    '''
    H = sharp_image.size(2)
    W = sharp_image.size(3)
    reblured_image = torch.zeros_like(sharp_image).to(sharp_image.device)
    n_positions = camera_positions.shape[1]
    warper = kornia.geometry.HomographyWarper(H, W, padding_mode='reflection')
    for n in range(n_positions):
        if forward:
            dst_homo_src_n = compute_homography_from_position(camera_positions[0, n, :], intrinsics)
        else:
            dst_homo_src_n = compute_homography_from_position(camera_positions[0, n, :], intrinsics, inverse=True)

        # dst_homo_src = torch.unsqueeze(intrinsics @ camera_positions, dim=0)
        # dst_homo_src  = camera_model()

        # print('dst_homo_src ', dst_homo_src)

        src_homo_dst_n: torch.Tensor = torch.inverse(dst_homo_src_n)


        # print('iter %d:' %i, 'dst_homo_src: ', dst_homo_src)
        # print('src_homo_dst', src_homo_dst)
        img_src_to_dst_n = warper(sharp_image, src_homo_dst_n)
        reblured_image += img_src_to_dst_n / n_positions

    return reblured_image

def rigid_transform(RX=0., RY=0., RZ=0., TX=0., TY=0., depth=1):
    '''
    Implementation of Rigid Transform with Rodrigues formula por rotation:
    https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Ver depth acá: https://eng.ucmerced.edu/people/zhu/CVPR14_deblurdepth.pdf
    '''

    R = np.sqrt(RX * RX + RY * RY + RZ * RZ)

    if (R > 0):
        RX = RX / R
        RY = RY / R
        RZ = RZ / R
        R_sin = np.sin(R)
        R_cos = np.cos(R)
        R_cosc = 1 - R_cos

        P = np.zeros((3, 3))
        P[0, 0] = R_cos + RX * RX * R_cosc
        P[1, 0] = RZ * R_sin + RY * RX * R_cosc
        P[2, 0] = -RY * R_sin + RZ * RX * R_cosc
        P[0, 1] = -RZ * R_sin + RX * RY * R_cosc
        P[1, 1] = R_cos + RY * RY * R_cosc
        P[2, 1] = RX * R_sin + RZ * RY * R_cosc
        P[0, 2] = RY * R_sin + RX * RZ * R_cosc
        P[1, 2] = -RX * R_sin + RY * RZ * R_cosc
        P[2, 2] = R_cos + RZ * RZ * R_cosc
        P = P + 1.0 / depth * np.array([[0, 0, TX], [0, 0, TY], [0, 0, 0]])

    else:
        P = np.array([[1, 0, TX / depth], [0, 1, TY / depth], [0, 0, 1]])

    return P

# def generarK(img_shape, pose, A=None, depth=1):
#     M, N, C = img_shape
#     N_pose = len(pose)
#     pose_weight = 1.0 / N_pose
#     K = sps.csr_matrix((M * N, M * N))
#     x = np.arange(0., N, 1)
#     y = np.arange(0., M, 1)
#     xx, yy = np.meshgrid(x, y)
#     loc = np.array([xx.flatten(order='F'), yy.flatten(order='F'), np.ones_like(xx).flatten(order='F')])
#
#     if A is None:
#         f = np.max(img_shape)
#         pi = M / 2
#         pj = N / 2
#         A = np.array([[f, 0, pj], [0, f, pi], [0, 0, 1]])
#
#     A_inv = np.linalg.inv(A)
#     # print('A_inv', A_inv)
#     homographies_list = []
#     for n in range(N_pose):
#         print('step %d/%d' % (n + 1, N_pose))
#         TX = pose[n, 0]
#         TY = pose[n, 1]
#         TZ = pose[n, 2]
#         RX = pose[n, 3]
#         RY = pose[n, 4]
#         RZ = pose[n, 5]
#         # print('pose: ', n, RX, RY, RZ, TX, TY, TZ)
#
#         Pi = rigid_transform(RX, RY, RZ, TX, TY, depth)
#         # print(n, ' P:', Pi)
#         Hi = A @ Pi @ A_inv
#         # print(n, ' Hi:', Hi)
#
#         homographies_list.append(Hi)
#
#         Hi_inv = np.linalg.inv(Hi)
#
#         # print('Hi', Hi)
#         # print('Hi_inv', Hi_inv)
#
#         proj_temp = Hi_inv @ loc  # 3x(MxN)
#         # print('proj_temp', proj_temp.shape)
#         location = proj_temp[:2, :] / (proj_temp[2, :] + 1e-10)  # 2x(MxN)
#         # print('location', location.shape)
#         loc_j = location[0]
#         loc_j_floor = np.floor(location[0])
#         # print('loc_x_floor', loc_x_floor.shape)
#         loc_i = location[1]
#         loc_i_floor = np.floor(location[1])
#         dif_i = loc_i - loc_i_floor
#         dif_j = loc_j - loc_j_floor
#         # print('loc_i', loc_i[0:20])
#         # print('loc_j', loc_j[0:20])
#         # print('dif_i', dif_i[0:20])
#         # print('dif_j', dif_j[0:20])
#         # print('dif_x', dif_x.shape)
#         # print('dif_x', dif_x.shape)
#         weight = np.concatenate(((1 - dif_i) * (1 - dif_j), dif_i * (1 - dif_j), (1 - dif_i) * dif_j, dif_i * dif_j))
#         # print('weight', weight.shape)
#         # print('weights_N', weight[0:20])
#         row_ind = np.hstack((loc_i_floor, loc_i_floor + 1, loc_i_floor, loc_i_floor + 1))
#         col_ind = np.hstack((loc_j_floor, loc_j_floor, loc_j_floor + 1, loc_j_floor + 1))
#         # print('row_ind', row_ind[0:20])
#         # print('col_ind', col_ind[0:20])
#         rows_cond = np.logical_and(row_ind >= 0, row_ind < M)
#         cols_cond = np.logical_and(col_ind >= 0, col_ind < N)
#         final_cond = np.logical_and(rows_cond, cols_cond)
#         # print('final_cond', final_cond.shape)
#         # print('final_cond', final_cond[0:20])
#
#         temp_loc = loc_j_floor * M + loc_i_floor
#         # print('temp_loc', temp_loc.shape)
#         temp_col_ind = np.concatenate((temp_loc, temp_loc + 1, temp_loc + M, temp_loc + M + 1))
#         # print('temp_col_ind', temp_col_ind.shape)
#         # print('temp_col_ind', temp_col_ind[0:20])
#
#         temp_row_ind = np.concatenate(
#             (np.arange(0, M * N), np.arange(0, M * N), np.arange(0, M * N), np.arange(0, M * N)))
#         # print('temp_row_ind', temp_row_ind.shape)
#         # print('temp_row_ind', temp_row_ind[0:20])
#         temp_value_ind = weight
#         # print('temp_value_ind', temp_value_ind.shape)
#
#         K_mat = sps.csr_matrix((temp_value_ind[final_cond], (temp_row_ind[final_cond], temp_col_ind[final_cond])),
#                                shape=(M * N, M * N))
#
#         # print(K.shape, weight.shape, K_mat.shape)
#         K = K + pose_weight * K_mat
#
#     return K, homographies_list



def generarK(img_shape, pose, padding=65, A=None, depth=1):
    M, N, C = img_shape
    N_pose = len(pose)
    pose_weight = 1.0 / N_pose
    K = sps.csr_matrix(((M + 2 * padding) * (N + 2 * padding), (M + 2 * padding) * (N + 2 * padding)))
    x = np.arange(0, N + 2 * padding, 1.)
    y = np.arange(0, M + 2 * padding, 1.)
    xx, yy = np.meshgrid(x, y)
    loc = np.array([xx.flatten(order='F'), yy.flatten(order='F'), np.ones_like(xx).flatten(order='F')])

    if A is None:
        f = np.max(img_shape)
        pi = (M + 2 * padding) / 2
        pj = (N + 2 * padding) / 2
        A = np.array([[f, 0, pj], [0, f, pi], [0, 0, 1]])

    A_inv = np.linalg.inv(A)
    # print('A_inv', A_inv)
    homographies_list = []
    for n in range(N_pose):
        print('step %d/%d' % (n + 1, N_pose))
        TX = pose[n, 0]
        TY = pose[n, 1]
        TZ = pose[n, 2]
        RX = pose[n, 3]
        RY = pose[n, 4]
        RZ = pose[n, 5]
        # print('pose: ', n, RX, RY, RZ, TX, TY, TZ)

        Pi = rigid_transform(RX, RY, RZ, TX, TY, depth)
        # print(n, ' P:', Pi)
        Hi = A @ Pi @ A_inv
        # print(n, ' Hi:', Hi)

        homographies_list.append(Hi)

        Hi_inv = np.linalg.inv(Hi)

        # print('Hi', Hi)
        # print('Hi_inv', Hi_inv)

        proj_temp = Hi_inv @ loc  # 3x(MxN)
        # print('proj_temp', proj_temp.shape)
        location = proj_temp[:2, :] / (proj_temp[2, :] + 1e-10)  # 2x(MxN)
        # print('location', location.shape)
        loc_j = location[0]
        loc_j_floor = np.floor(location[0])
        # print('loc_x_floor', loc_x_floor.shape)
        loc_i = location[1]
        loc_i_floor = np.floor(location[1])
        dif_i = loc_i - loc_i_floor
        dif_j = loc_j - loc_j_floor
        # print('loc_i', loc_i[0:20])
        # print('loc_j', loc_j[0:20])
        # print('dif_i', dif_i[0:20])
        # print('dif_j', dif_j[0:20])
        # print('dif_x', dif_x.shape)
        # print('dif_x', dif_x.shape)
        weight = np.concatenate(((1 - dif_i) * (1 - dif_j), dif_i * (1 - dif_j),
                                 (1 - dif_i) * dif_j, dif_i * dif_j))
        # print('weight', weight.shape)
        # print('weights_N', weight[0:20])
        row_ind = np.hstack((loc_i_floor, loc_i_floor + 1, loc_i_floor, loc_i_floor + 1))
        col_ind = np.hstack((loc_j_floor, loc_j_floor, loc_j_floor + 1, loc_j_floor + 1))
        # print('row_ind', row_ind[0:20])
        # print('col_ind', col_ind[0:20])
        rows_cond = np.logical_and(row_ind >= 0, row_ind < (M + 2 * padding))
        cols_cond = np.logical_and(col_ind >= 0, col_ind < (N + 2 * padding))
        final_cond = np.logical_and(rows_cond, cols_cond)
        # print('final_cond', final_cond.shape)
        # print('final_cond', final_cond[0:20])

        # Matlab indexing
        temp_loc = loc_j_floor * (M + 2 * padding) + loc_i_floor

        # print('temp_loc', temp_loc.shape)
        temp_col_ind = np.concatenate(
            (temp_loc, temp_loc + 1, temp_loc + (M + 2 * padding), temp_loc + (M + 2 * padding) + 1))
        # print('temp_col_ind', temp_col_ind.shape)
        # print('temp_col_ind', temp_col_ind[0:20])

        temp_row_ind = np.concatenate(
            (np.arange(0, (M + 2 * padding) * (N + 2 * padding)),
             np.arange(0, (M + 2 * padding) * (N + 2 * padding)),
             np.arange(0, (M + 2 * padding) * (N + 2 * padding)),
             np.arange(0, (M + 2 * padding) * (N + 2 * padding))))
        # print('temp_row_ind', temp_row_ind.shape)
        # print('temp_row_ind', temp_row_ind[0:20])
        temp_value_ind = weight
        # print('temp_value_ind', temp_value_ind.shape)

        K_mat = sps.csr_matrix((temp_value_ind[final_cond], (temp_row_ind[final_cond], temp_col_ind[final_cond])),
                               shape=((M + 2 * padding) * (N + 2 * padding), (M + 2 * padding) * (N + 2 * padding)))

        # print(K.shape, weight.shape, K_mat.shape)
        K = K + pose_weight * K_mat

    return K, homographies_list


def mostrar_kernels(K, img_shape, padding=65, window=65, output_name='kernels.png', depth=1):

    M, N, C = img_shape
    xs = np.arange(padding + window // 2 + 1, N + padding - window // 2, window)
    ys = np.arange(padding + window // 2 + 1, M + padding - window // 2, window)
    xx, yy = np.meshgrid(xs, ys)
    loc = np.array([xx.flatten(order='F'), yy.flatten(order='F'), np.ones_like(xx).flatten(order='F')])
    print(K.shape, img_shape, padding)
    output_img = np.zeros((M+2*padding, N+2*padding))
    for i in range(loc.shape[1]):
        cy = loc[1, i]
        cx = loc[0, i]
        img = np.zeros((M+2*padding, N+2*padding))
        img[cy, cx] = 1
        kernel_vector = K @ img.flatten(order='F')
        kernel_img = np.reshape(kernel_vector, (M+2*padding, N+2*padding), order='F')
        kernel_img = kernel_img[cy - window // 2:cy + window // 2, cx - window // 2:cx + window // 2].copy()

        #row_kernel_img = np.reshape(K[(M+2*padding) * cx + cy, :], ((M+2*padding), (N+2*padding)), order='F').todense()
        #row_kernel_img = row_kernel_img[cy - window // 2:cy + window // 2, cx - window // 2:cx + window // 2].copy()


        #col_kernel_img = np.reshape(K[:, (M+2*padding) * cx + cy], ((M+2*padding), (N+2*padding)), order='F').todense()
        #col_kernel_img = col_kernel_img[cy - window // 2:cy + window // 2, cx - window // 2:cx + window // 2].copy()


        mink = kernel_img.min()
        maxk = kernel_img.max()
        output_img[cy - window // 2:cy + window // 2, cx - window // 2:cx + window // 2] = (kernel_img - mink) / (
                    maxk - mink)
        output_img[cy - window // 2:cy + window // 2, cx - window // 2] = 1
        output_img[cy - window // 2:cy + window // 2, cx + window // 2] = 1
        output_img[cy - window // 2, cx - window // 2:cx + window // 2] = 1
        output_img[cy + window // 2, cx - window // 2:cx + window // 2] = 1

    if padding>0:
        output_img = output_img[padding:-padding, padding:-padding]
    imsave(output_name, output_img)


def compute_projection_matrix(camera_position):
    '''
    camera position: (3) array with rotation angles
    '''

    P = kornia.geometry.angle_axis_to_rotation_matrix(camera_position[None, :])

    return P

def compute_intrinsics(W,H):

    f = np.max([W, H])
    pi = H / 2
    pj = W / 2
    A = torch.Tensor([[f, 0, pj], [0, f, pi], [0, 0, 1]])
    return A


def compute_homography_from_position(camera_position, intrinsics, inverse=False, normalize=True):
    '''
    camera position: (3) array with rotation angles
    '''
    
    N = 2 * intrinsics[0, 2]
    M = 2 * intrinsics[1, 2]
    P = compute_projection_matrix(camera_position)

    dst_homo_src = intrinsics @ P @ torch.inverse(intrinsics)

    if inverse:
        dst_homo_src = torch.inverse(dst_homo_src)

    dst_homo_src = torch.unsqueeze(dst_homo_src, dim=0)

    if normalize:
        dst_homo_src = kornia.geometry.conversions.normalize_homography(dst_homo_src, (M, N), (M, N))

    return dst_homo_src

def draw3D(thetaX, thetaY, thetaZ):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(projection='3d')
    ax.plot(thetaX,thetaY,thetaZ,'*-')
    ax.plot([0],[0],[0],'*r')
    ax.set_xlabel('thetaX')
    ax.set_ylabel('thetaY')
    ax.set_zlabel('thetaZ')
    return fig

def show_positions(pos, gt_pos):
    fig = plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.plot(pos[:,0], pos[:,1], pos[:,2], '*-', label='positions found')
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], '*-', label='gt_positions')
    ax1.plot([0],[0],[0],'*r')
    ax1.plot(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], '*g', label='start point')
    ax1.plot(pos[0, 0], pos[0, 1], pos[0, 2], '*g', label='start point')
    ax1.plot(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], '*m', label='end point')
    ax1.plot(pos[-1, 0], pos[-1, 1], pos[-1, 2], '*m', label='end point')
    ax1.set_xlim(-0.01,0.01)
    ax1.set_ylim(-0.01,0.01)
    ax1.set_zlim(-0.01,0.01)
    ax1.set_xlabel('thetaX')
    ax1.set_ylabel('thetaY')
    ax1.set_zlabel('thetaZ')
    ax1.view_init(elev=10., azim=15)

    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.plot(pos[:,0], pos[:,1], pos[:,2], '*-', label='positions found')
    ax2.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], '*-', label='gt_positions')
    ax2.plot([0],[0],[0],'*r')
    ax2.plot(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], '*g', label='start point')
    ax2.plot(pos[0, 0], pos[0, 1], pos[0, 2], '*g', label='start point')
    ax2.plot(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], '*m', label='end point')
    ax2.plot(pos[-1, 0], pos[-1, 1], pos[-1, 2], '*m', label='end point')
    ax2.set_xlabel('thetaX')
    ax2.set_ylabel('thetaY')
    ax2.set_zlabel('thetaZ')
    ax2.view_init(elev=10., azim=15)
    plt.legend()
    #plt.savefig('3DPositions_comparison.png')
    return fig

