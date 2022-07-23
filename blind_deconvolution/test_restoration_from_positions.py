import torch
from skimage.io import imread
from skimage.transform import rescale
from utils.homographies import compute_intrinsics
import numpy as np
from utils.RL_restoration_from_positions import RL_restore_from_positions, combined_RL_restore_from_positions
from models.network_nimbusr_pmpb import NIMBUSR_PMPB as net
from utils.visualization import save_image, tensor2im
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--blurry_image', '-b', type=str, help='blurry image', default='/home/guillermo/github/camera_shake/data/COCO_homographies_small_gf1/blurry/000000000009_0.jpg')
parser.add_argument('--positions', '-p', type=str, help='positions', default='/home/guillermo/github/camera_shake/data/COCO_homographies_small_gf1/positions/000000000009_0.txt')
parser.add_argument('--rescale_factor','-rf', type=float, default=1)

args = parser.parse_args()

def load_nimbusr_net():
    
    opt_net = { "n_iter": 8
        , "h_nc": 64
        , "in_nc": 4
        , "out_nc": 3
        , "ksize": 25
        , "nc": [64, 128, 256, 512]
        , "nb": 2
        , "gc": 32
        , "ng": 2
        , "reduction" : 16
        , "act_mode": "R" 
        , "upsample_mode": "convtranspose" 
        , "downsample_mode": "strideconv"}

    opt_data = { "phase": "train"
          , "dataset_type": "usrnet_multiblur"
          , "dataroot_H": "datasets/COCO/val2014"
          , "dataroot_L": None
          , "H_size": 256
          , "use_flip": True
          , "use_rot": True
          , "scales": [2]
          , "sigma": [0, 2]
          , "sigma_test": 15
          , "n_channels": 3
          , "dataloader_shuffle": True
          , "dataloader_num_workers": 16
          , "dataloader_batch_size": 16
          , "motion_blur": True

          , "coco_annotation_path": "datasets/COCO/instances_val2014.json"}

    path_pretrained = r'../model_zoo/NIMBUSR.pth'

    netG = net(n_iter=opt_net['n_iter'],
                   h_nc=opt_net['h_nc'],
                   in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'],
                   downsample_mode=opt_net['downsample_mode'],
                   upsample_mode=opt_net['upsample_mode']
                   )

    netG.load_state_dict(torch.load(path_pretrained))
    netG = netG.to('cuda')

    return netG


blurry_image_filename = args.blurry_image #'/home/guillermo/github/camera_shake/data/COCO_homographies_small_gf1/blurry/000000000009_0.jpg' 
positions_filename = args.positions #'/home/guillermo/github/camera_shake/data/COCO_homographies_small_gf1/positions/000000000009_0.txt'
#sharp_image_filename = '/home/guillermo/github/camera_shake/data/COCO_homographies_small_gf1//sharp/000000000009_0.jpg'
n_iters = 20
GPU = 0
n_positions = 25
restoration_method='NIMBUSR'
output_folder='restoration_from_homographies'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

blurry_image = rescale(imread(blurry_image_filename)/255.0, (args.rescale_factor,args.rescale_factor,1),anti_aliasing=True)
#sharp_image = rescale(imread(sharp_image_filename)/255.0,(0.6,0.6,1),anti_aliasing=True)
blurry_tensor = torch.from_numpy(blurry_image).permute(2,0,1)[None].cuda(GPU).float()
#sharp_tensor = torch.from_numpy(sharp_image).permute(2,0,1)[None].cuda(GPU).float()
initial_tensor = blurry_tensor.clone()

camera_positions_np = np.loadtxt(positions_filename, delimiter=',')
camera_positions_np = camera_positions_np[:,3:]


camera_positions = torch.from_numpy(camera_positions_np).cuda(GPU)[None].float()
_, C,H,W = blurry_tensor.shape
print(C,H,W)
intrinsics = compute_intrinsics(W, H).cuda(GPU)[None]


if restoration_method=='RL':
    output = RL_restore_from_positions(blurry_tensor, initial_tensor, camera_positions, n_iters, GPU, isDebug=True, reg_factor=1e-3)
    #combined_RL_restore_from_positions(blurry_tensor, initial_tensor, camera_positions, n_iters, GPU, isDebug=True)
else: 
    netG = load_nimbusr_net()
    noise_level = 0.01
    noise_level = torch.FloatTensor([noise_level]).view(1,1,1).cuda(GPU)
    output = netG(blurry_tensor, camera_positions, intrinsics, 1, sigma=noise_level[None,:])
    

img_name, ext = blurry_image_filename.split('/')[-1].split('.')    
output_img = tensor2im(torch.clamp(output[0].detach(),0,1) - 0.5)
save_image(output_img, os.path.join(output_folder, img_name + '_' + restoration_method + '.png' ))
save_image((255*blurry_image).astype(np.uint8), os.path.join(output_folder, img_name + '.png' ))
print('Output saved in ', os.path.join(output_folder, img_name + '_' + restoration_method + '.png' ))