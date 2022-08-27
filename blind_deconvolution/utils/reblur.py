import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss
import time

def apply_saturation_function(img, max_value=0.5, get_derivative=False):
    '''
    Implements the saturated function proposed by Whyte
    https://www.di.ens.fr/willow/research/saturation/whyte11.pdf
    :param img: input image may have values above max_value
    :param max_value: maximum value
    :return:
    '''

    a=50
    img[img>max_value+0.5] = max_value+0.5  # to avoid overflow in exponential

    if get_derivative==False:
        saturated_image = img - 1.0/a*torch.log(1+torch.exp(a*(img - max_value)))
        output_image = F.relu(saturated_image + (1 - max_value)) - (1 - max_value)

        #del saturated_image
    else:
        output_image = 1.0 / ( 1 + torch.exp(a*(img - max_value)) )

    return output_image




def forward_reblur(sharp_estimated, kernels, masks, GPU=0, size='valid', padding_mode='reflect',
                   manage_saturated_pixels=True, max_value=0.5, stride=1):
    n_kernels = kernels.size(1)
    K = kernels.size(2)
    N = sharp_estimated.size(0)
    C = sharp_estimated.size(1)
    H = sharp_estimated.size(2)
    W = sharp_estimated.size(3)
    if size=='valid':
        H += -K + 1
        W += -K + 1
    else:
        padding = torch.nn.ReflectionPad2d(K // 2)
        sharp_estimated = padding(sharp_estimated)

    output_reblurred = torch.empty(N, n_kernels, C, H//stride, W//stride).cuda(GPU)

    for num in range(N):  # print('n = ',n)
        for c in range(C):
            # print('gt padded one channel shape: ', gt_n_padded_c.shape)

            conv_output = F.conv2d(sharp_estimated[num:num + 1, c:c + 1, :, :],
                                       kernels[num][:, np.newaxis, :, :], stride=stride)

            # print('conv output shape: ', conv_output.shape)
            output_reblurred[num:num + 1, :, c, :, :] = conv_output * masks[num:num + 1]
            del conv_output


    # print('reblur_image shape before sum:', reblurred_images.shape)
    output_reblurred = torch.sum(output_reblurred, (1))

    if manage_saturated_pixels:
        output_reblurred = apply_saturation_function(output_reblurred, max_value)

    return output_reblurred


def compute_Lp_norm(kernels_tensor, p):
    N, K, S, S = kernels_tensor.shape
    output = 0
    for n in range(N):
        for k in range(K):
            kernel = kernels_tensor[n, k, :, :]
            p_norm = torch.pow(torch.sum(torch.pow(torch.abs(kernel), p)), 1. / p)
            output = output + p_norm
    return output/(N*K)

def compute_total_variation_loss(img):
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).abs()).sum()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).abs()).sum()
    return  (tv_h + tv_w)/(img.shape[0]*img.shape[1])


def compute_masks_regularization_loss(masks, MASKS_REGULARIZATION_TYPE, MASKS_REGULARIZATION_LOSS_FACTOR):

    masks_regularization_loss = 0.
    if MASKS_REGULARIZATION_TYPE == 'L2':
        masks_regularization_loss = MASKS_REGULARIZATION_LOSS_FACTOR * torch.mean(masks * masks)
    elif MASKS_REGULARIZATION_TYPE == 'TV':
        masks_regularization_loss = compute_total_variation_loss(masks, MASKS_REGULARIZATION_LOSS_FACTOR)

    return masks_regularization_loss

def compute_kernels_regularization_loss(kernels, KERNELS_REGULARIZATION_TYPE, KERNELS_REGULARIZATION_LOSS_FACTOR):

    kernels_regularization_loss = 0.

    if KERNELS_REGULARIZATION_TYPE == 'L2':
        kernels_regularization_loss = KERNELS_REGULARIZATION_LOSS_FACTOR * torch.mean(kernels ** 2)
    if KERNELS_REGULARIZATION_TYPE == 'L1':
        kernels_regularization_loss = KERNELS_REGULARIZATION_LOSS_FACTOR * torch.mean(
            torch.abs(kernels))
    elif KERNELS_REGULARIZATION_TYPE == 'TV':
        kernels_regularization_loss = compute_total_variation_loss(kernels,
                                                                    KERNELS_REGULARIZATION_LOSS_FACTOR)
    elif KERNELS_REGULARIZATION_TYPE == 'Lp':
        kernels_regularization_loss = KERNELS_REGULARIZATION_LOSS_FACTOR * compute_Lp_norm(kernels,
                                                                                            p=0.5)

    return kernels_regularization_loss


def get_masks_weights(gt_masks):
    # store the contribution of each ground-truth mask,
    gt_masks_sums = gt_masks.sum(3).sum(2)

    # for each mask pixel is stored the contribution os its associated mask, size=(N, H, H)
    masks_areas = (gt_masks * gt_masks_sums[:, :, np.newaxis, np.newaxis]).sum(1)

    # when calculating the kernel differences for each pixel, the weight each pixel will have will be
    # inversely proportional to its associated mask area

    try:
        #print(gt_masks.shape)
        roi = (torch.max(gt_masks > 0.95, (1)).values).float() # only pixels with dominant kernel are used to compute loss
        #print(roi.shape, roi.dtype, masks_areas.shape)
        masks_weights = 1.0 / (masks_areas + 1) * roi
        #print(masks_weights.shape)
    except:
        print('Error computing mask weights')

    return masks_weights

def compute_reblurred_image_and_kernel_loss(sharp_image, kernels, masks, gt_kernels, gt_masks,
                        masks_weights, kernels_loss_type,  GPU=0, manage_saturated_pixels=True,
                        pairwise_matrix=None, stride=1):

    n_kernels = kernels.size(1)
    K = kernels.size(2)
    N = sharp_image.size(0)
    C = sharp_image.size(1)
    H = (sharp_image.size(2) - K + 1)//stride
    W = (sharp_image.size(3) - K + 1)//stride
    reblurred_images = torch.empty(N, n_kernels, C, H, W).cuda(GPU)

    Kgt = gt_kernels.shape[1]
    Wk = gt_kernels.shape[2]  # kernel side

    kernels_loss = torch.Tensor([0.]).cuda(GPU);
    mse_loss = torch.nn.MSELoss(reduction='none')
    l1_loss = torch.nn.L1Loss(reduction='none')
    huber_loss = torch.nn.SmoothL1Loss(reduction='none')
    kl_loss = torch.nn.KLDivLoss(reduction='none')
    for n in range(N):
        gt_masks_nn = gt_masks[n].view(Kgt, H * W)  # *(1/(masks_sums[n][nonzero]))
        gt_kernels_nn = gt_kernels[n].view(Kgt, Wk * Wk)
        gt_kernels_per_pixel = torch.mm(gt_kernels_nn.t(), gt_masks_nn)
        masks_weights_nn = masks_weights[n].view(H * W)

        predicted_kernels_per_pixel = torch.mm(kernels[n].contiguous().view(n_kernels, Wk * Wk).t(),
                                               masks[n].contiguous().view(n_kernels, H * W))

        if kernels_loss_type == 'L2':
            #per_pixel_kernel_diff = (predicted_kernels_per_pixel - gt_kernels_per_pixel)**2
            per_pixel_kernel_diff = mse_loss(predicted_kernels_per_pixel, gt_kernels_per_pixel)
            kernels_loss += (per_pixel_kernel_diff.sum(0) * masks_weights[n].view(H * W)).sum() / N
        elif kernels_loss_type == 'L1':
            #per_pixel_kernel_diff = (predicted_kernels_per_pixel - gt_kernels_per_pixel).abs()
            per_pixel_kernel_diff = l1_loss(predicted_kernels_per_pixel, gt_kernels_per_pixel)
            kernels_loss += (per_pixel_kernel_diff.sum(0) * masks_weights[n].view(H * W)).sum() / N
        elif kernels_loss_type == 'Huber':
            per_pixel_kernel_diff = huber_loss(predicted_kernels_per_pixel, gt_kernels_per_pixel)
            kernels_loss += (per_pixel_kernel_diff.sum(0) * masks_weights[n].view(H * W)).sum() / N
        elif kernels_loss_type == 'KL':
            per_pixel_kernel_diff = kl_loss(torch.log(predicted_kernels_per_pixel), gt_kernels_per_pixel)
            kernels_loss += (per_pixel_kernel_diff.sum(0) * masks_weights[n].view(H * W)).sum() / N
        elif kernels_loss_type == 'IND':
            # for i in range(H*W): # para cada pixel de la image
            #     a = gt_kernels_per_pixel[:,i:i+1]  #k_gt
            #     b = predicted_kernels_per_pixel[:,i:i+1]  #k_p
            #     kernels_loss += torch.squeeze(torch.mm(a.t(), torch.mm(pairwise_matrix, b)))/(H*W*N)
            start_IND = time.time()
            divs=8
            len = (H * W) // divs
            for i in range(divs):  # para cada fila de la imagen
                a = gt_kernels_per_pixel[:,i*len:len*(i+1)]  #k_gt
                b = predicted_kernels_per_pixel[:,i*len:len*(i+1)]  #k_p
                #diff=torch.abs(a-b)
                w = masks_weights_nn[i*len:len*(i+1)]
                distances = torch.diagonal(torch.mm(a.t(), torch.mm(pairwise_matrix, b)))
                #print(distances.shape, distances.min(), distances.max())
                kernels_loss += torch.mean(w*distances)/divs
            stop_IND = time.time()
            print('IND finished in %f seconds' % (stop_IND-start_IND))
            # a = gt_kernels_per_pixel  #k_gt
            # b = predicted_kernels_per_pixel  #k_p
            # kernels_loss += torch.mean(torch.mm(a.t(), torch.mm(pairwise_matrix, b)))


        for c in range(C):
            conv_output = F.conv2d(sharp_image[n:n + 1, c:c + 1, :, :], kernels[n][:, np.newaxis, :, :], stride=stride)
            reblurred_images[n:n + 1, :, c, :, :] = conv_output * masks[n:n + 1]

    reblurred_images = torch.sum(reblurred_images, (1))

    if manage_saturated_pixels:
        output_reblurred = apply_saturation_function(reblurred_images)
    else:
        output_reblurred = reblurred_images
    
    return output_reblurred, kernels_loss







