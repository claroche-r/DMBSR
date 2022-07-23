import numpy as np
import argparse
from models.TwoHeadsNetwork import TwoHeadsNetwork
from models.network_nimbusr import NIMBUSR as net

import torch

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import gray2rgb


from torchvision import transforms
import os
import json

from utils.visualization import save_image, tensor2im, save_kernels_grid
from utils.restoration import RL_restore, combined_RL_restore


def load_nimbusr_net():
<<<<<<< HEAD
    
=======

>>>>>>> 28c6287c4b852cc711a49e8f259d6a1d931b164c
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


parser = argparse.ArgumentParser()
parser.add_argument('--blurry_images', '-b', type=str, required=True, help='list with the original blurry images or path to a blurry image')
parser.add_argument('--reblur_model', '-m', type=str, required=True, help='two heads reblur model')
parser.add_argument('--n_iters', '-n', type=int, default=30)
parser.add_argument('--K', '-k', type=int, default=25, help='number of kernels in two heads model')
parser.add_argument('--blur_kernel_size', '-bks', type=int, default=33, help='blur_kernel_szie')
parser.add_argument('--gpu_id', '-g', type=int, default=0)
parser.add_argument('--output_folder','-o', type=str, help='output folder', default='testing_results')
parser.add_argument('--resize_factor','-rf', type=float, default=1)
parser.add_argument('--saturation_method', type=str, default='combined')
parser.add_argument('--regularization','-reg', type=str, help='regularization method')
parser.add_argument('--reg_factor', type=float, default=1e-3, help='regularization factor')
parser.add_argument('--sat_threshold','-sth', type=float, default=0.99)
parser.add_argument('--gamma_factor', type=float, default=2.2, help='gamma correction factor')
parser.add_argument('--optim_iters', action='store_true', default=True, help='stop iterating when reblur loss is 1e-6')
parser.add_argument('--smoothing', action='store_true', default=True, help='apply smoothing to the saturated region mask')
parser.add_argument('--erosion', action='store_true', default=True, help='apply erosion to the non-saturated region')
parser.add_argument('--dilation', action='store_true', default=False, help='apply dilation to the saturated region using the kernel as structural element')
parser.add_argument('--restoration_method','-rm', type=str, default='RL')



'''
-b /media/carbajal/OS/data/datasets/cvpr16_deblur_study_real_dataset/real_dataset/coke.jpg  -m /media/carbajal/OS/data/models/ade_dataset/NoFC/gamma_correction/L1/L2_epoch150_epoch150_L1_epoch900.pkl -n 20  --saturation_method 'combined'
'''

args = parser.parse_args()

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

with open(os.path.join(args.output_folder, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

def get_images_list(list_path):

    with open(list_path) as f:
        images_list = f.readlines()
        images_list = [l[:-1] for l in images_list]
    f.close()

    return images_list


if args.blurry_images.endswith('.txt'):
    blurry_images_list = get_images_list(args.blurry_images)
else:
    blurry_images_list = [args.blurry_images]


two_heads = TwoHeadsNetwork(args.K).cuda(args.gpu_id)
two_heads.load_state_dict(torch.load(args.reblur_model, map_location='cuda:%d' % args.gpu_id))
two_heads.eval()


for i,blurry_path in enumerate(blurry_images_list):

    img_name, ext = blurry_path.split('/')[-1].split('.')
    blurry_image =  imread(blurry_path)
    blurry_image = blurry_image[:,:,:3]


    M, N, C = blurry_image.shape
    if args.resize_factor != 1:
        if len(blurry_image.shape) == 2:
            blurry_image = gray2rgb(blurry_image)
        new_shape = (int(args.resize_factor*M), int(args.resize_factor*N), C )
        blurry_image = resize(blurry_image,new_shape).astype(np.float32)


    initial_image = blurry_image.copy()

    blurry_tensor = transforms.ToTensor()(blurry_image)
    blurry_tensor = blurry_tensor[None,:,:,:]
    blurry_tensor = blurry_tensor.cuda(args.gpu_id)

    initial_restoration_tensor = transforms.ToTensor()(initial_image)
    initial_restoration_tensor = initial_restoration_tensor[None, :, :, :]
    initial_restoration_tensor = initial_restoration_tensor.cuda(args.gpu_id)

    save_image(tensor2im(initial_restoration_tensor[0] - 0.5), os.path.join(args.output_folder,
                                                       img_name + '.png' ))

    with torch.no_grad():
        blurry_tensor_to_compute_kernels = blurry_tensor**args.gamma_factor - 0.5
        kernels, masks = two_heads(blurry_tensor_to_compute_kernels)
        save_kernels_grid(blurry_tensor[0],kernels[0], masks[0], os.path.join(args.output_folder, img_name + '_kernels'+'.png'))


    output = initial_restoration_tensor


    with torch.no_grad():

        if args.restoration_method == 'RL': 

            if args.saturation_method == 'combined':
                output = combined_RL_restore(blurry_tensor, output, kernels, masks, args.n_iters,
                                         args.gpu_id, SAVE_INTERMIDIATE=True, saturation_threshold=args.sat_threshold,
                                         reg_factor=args.reg_factor, optim_iters=args.optim_iters, gamma_correction_factor=args.gamma_factor,
                                         apply_dilation=args.dilation, apply_smoothing=args.smoothing, apply_erosion=args.erosion)
            else:
                output = RL_restore(blurry_tensor, output, kernels, masks, args.n_iters,
                                args.gpu_id, SAVE_INTERMIDIATE=True,
                                method=args.saturation_method,gamma_correction_factor=args.gamma_factor,
                                saturation_threshold=args.sat_threshold, reg_factor=args.reg_factor)

        elif args.restoration_method == 'NIMBUSR': 
                netG = load_nimbusr_net()
                noise_level = 0.01
                noise_level = torch.FloatTensor([noise_level]).view(1,1,1).cuda(args.gpu_id)
                kernels_flipped = torch.flip(kernels, dims=(2, 3))
                output = netG(blurry_tensor, masks, kernels_flipped, 1, sigma=noise_level[None,:])


    output_img = tensor2im(torch.clamp(output[0],0,1) - 0.5)
    save_image(output_img, os.path.join(args.output_folder, img_name + '_' + args.restoration_method + '.png' ))
    print('Output saved in ', os.path.join(args.output_folder, img_name + '_' + args.restoration_method + '.png' ))

