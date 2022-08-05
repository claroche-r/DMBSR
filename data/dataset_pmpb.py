import os
import random
import numpy as np

from scipy import ndimage
from scipy.io import loadmat

import torch
import torch.nn.functional as F
import torch.utils.data as data
from blind_deconvolution.utils.homographies import compute_intrinsics

import utils.utils_image as util
import utils.utils_sisr as sisr
from utils import utils_deblur


class Dataset(data.Dataset):
    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3


        self.scales = opt['scales'] if opt['scales'] is not None else [1,2,3,4]
        self.sigma = opt['sigma'] if opt['sigma'] else [0, 25]
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        
        self.dataroot = self.opt['dataroot']
        self.blurry_files = os.listdir(os.path.join(self.dataroot,'blurry'))
        self.sharp_files = os.listdir(os.path.join(self.dataroot,'sharp'))
        self.positions_files = os.listdir(os.path.join(self.dataroot,'positions'))
        self.blurry_files.sort()
        self.sharp_files.sort()
        self.positions_files.sort()
        #self.ids = []
        self.count = 0

    def compute_intrinsics(W,H):
            
        f = np.max([W, H])
        pi = H / 2
        pj = W / 2
        A = np.array([[f, 0, pj], [0, f, pi], [0, 0, 1]])
        return A
    
    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------


        # Open image
        H_path = os.path.join(self.dataroot, 'sharp', self.sharp_files[index])
        L_path = os.path.join(self.dataroot, 'blurry', self.blurry_files[index])
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        img_L = util.imread_uint(L_path, self.n_channels)
        img_L = util.uint2single(img_L)

        img_H = util.single2tensor3(img_H)
        img_L = util.single2tensor3(img_L)

        #img_H = util.imresize(img_H, 0.5)
        #img_L = util.imresize(img_L, 0.5)
        
        P_path = os.path.join(self.dataroot, 'positions', self.positions_files[index])
        camera_positions_np = np.loadtxt(P_path, delimiter=',')
        camera_positions_np = camera_positions_np[:,3:]
        positions = torch.from_numpy(camera_positions_np).float()
        # ---------------------------
        # 1) scale factor, ensure each batch only involves one scale factor
        # ---------------------------
        if self.count % self.opt['dataloader_batch_size'] == 0:
            self.sf = random.choice(self.scales)

        self.count += 1


        # ------------------------------------
        # 5) Downsampling
        # ------------------------------------
        C, H, W = img_H.shape
        intrinsics = compute_intrinsics(W,H)
        #print('image size = (%d, %d)' % (W,H))



        # --------------------------------
        # 7) add noise
        # --------------------------------
        if random.randint(0, 8) == 1:
                noise_level = 0 / 255.0
        else:
            noise_level = np.random.randint(0, self.sigma_max) / 255.0
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)

        noise_level = torch.FloatTensor([noise_level]).view(1,1,1)
        return {'L': img_L, 'H': img_H, 'positions': torch.FloatTensor(positions), 'intrinsics': torch.FloatTensor(intrinsics),
                'sigma': noise_level, 'sf': self.sf, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        if self.opt['phase'] == 'train':
            return len(self.blurry_files)
        
