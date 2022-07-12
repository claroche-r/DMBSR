import os
import random
import numpy as np

from scipy import ndimage
from scipy.io import loadmat

import torch
import torch.nn.functional as F
import torch.utils.data as data

from pycocotools.coco import COCO

import utils.utils_image as util
import utils.utils_sisr as sisr
from utils import utils_deblur


class Dataset(data.Dataset):
    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size']
        self.sigma = opt['sigma'] if opt['sigma'] else [0, 25]
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else 0
        self.scales = opt['scales'] if opt['scales'] is not None else [1,2,3,4]
        self.motion_ker = loadmat('kernels/custom_blur_centered.mat')['kernels'][0]

        self.ksize = 33 # kernel size
        self.pca_size = 15
        self.min_p_mask = 20 ** 2  #Minimum number of pixels per mask to blur
        self.dataroot_H = self.opt['dataroot_H']
        self.coco = COCO(self.opt['coco_annotation_path'])
        indexes = self.coco.getImgIds()

        self.ids = []

        for i in indexes:
            img = self.coco.loadImgs(i)[0]
            if min(img['height'], img['width']) > self.opt['H_size'] + 49:
                self.ids.append(i)

        self.count = 0

    def random_kernel(self):
        r_value = random.randint(0, 7)
        if r_value>3 :  # Motion blur
            index_blur = random.randint(0, len(self.motion_ker) - 1)
            k = self.extend_kernel_size(self.motion_ker[index_blur], (self.ksize, self.ksize))

        else:   # Gaussian blur
            sf_k = random.choice(self.scales)
            k = sisr.gen_kernel(scale_factor=np.array([sf_k, sf_k]))
            mode_k = random.randint(0, 7)
            k = util.augment_img(k, mode=mode_k)

        return k
    
    def extend_kernel_size(self, kernel, size):
        h, w = kernel.shape
        assert h <= size[0] and w <= size[1]

        kernel_full = np.zeros(size)
        sh, sw = (size[0] - h) // 2, (size[1] - w) // 2
        eh, ew = size[0] - sh, size[1] - sw

        kernel_full[sh:eh, sw:ew] = kernel
        return kernel_full

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        # Get coco informations and segmentation map
        id_coco = self.ids[index]
        dico_coco = self.coco.loadImgs(id_coco)[0]
        annIds = self.coco.getAnnIds(imgIds=dico_coco['id'])
        anns = self.coco.loadAnns(annIds)

        # Open image
        H_path = os.path.join(self.dataroot_H, dico_coco['file_name'])
        L_path = H_path
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ---------------------------
        # 1) scale factor, ensure each batch only involves one scale factor
        # ---------------------------
        if self.count % self.opt['dataloader_batch_size'] == 0:
            self.sf = random.choice(self.scales)

        self.count += 1

        # --------------------------------
        # 2) randomly crop H patch
        # --------------------------------
        H, W, C = img_H.shape
        if self.sf == 3:
            H_patch = (self.patch_size - (self.patch_size % self.sf)) + 48  # We add 20 pixels on each side of the image to avoid side effects for the blurring
        else:
            H_patch = self.patch_size + 48 

        rnd_h = random.randint(0, max(0, H - H_patch - 1))
        rnd_w = random.randint(0, max(0, W - H_patch - 1))
        img_H = img_H[rnd_h:rnd_h + H_patch, rnd_w:rnd_w + H_patch, :]

        # ------------------------------------
        # 3) Blurring
        # ------------------------------------
        n, p, _ = img_H.shape
        blurred = np.zeros_like(img_H)
        mask_glob = np.zeros((n, p))
        kernel_map = torch.zeros((self.pca_size, H_patch, H_patch))
        basis = torch.FloatTensor(np.array([self.extend_kernel_size(self.random_kernel(), (self.ksize, self.ksize)) for _ in range(self.pca_size)]))
        basis = basis.view(self.pca_size, self.ksize, self.ksize)

        # Blur each mask
        mask_number = 0
        for ann in anns:
            mask = self.coco.annToMask(ann)
            mask = mask[rnd_h:rnd_h + H_patch, rnd_w:rnd_w + H_patch]

            if mask.sum() > self.min_p_mask and mask_number < self.pca_size - 1:
                mask_glob += mask.astype(int)

                # Get random blur kernel
                kernel = basis[mask_number]

                # Blur mask for smooth transition
                mask = ndimage.filters.convolve(mask.astype(float), kernel, mode='wrap')
                kernel_map[mask_number] = torch.FloatTensor(mask)

                # Blur image on the given mask 
                blurred += ndimage.filters.convolve(img_H, np.expand_dims(kernel, axis=2), mode='wrap') * mask[:,:,np.newaxis].repeat(3, 2)
                mask_number += 1

        # Blur background
        mask_background = (1 - (mask_glob > 0).astype(float))

        # Get random blur kernel
        kernel = basis[mask_number]

        # Blur mask
        mask_background = ndimage.filters.convolve(mask_background, kernel, mode='wrap')
        kernel_map[mask_number] = torch.FloatTensor(mask_background)


        # Blur image on the given mask 
        blurred += ndimage.filters.convolve(img_H, np.expand_dims(kernel, axis=2), mode='wrap') * mask_background[:,:,np.newaxis].repeat(3, 2)

        # ------------------------------------
        # 5) Downsampling
        # ------------------------------------
        H, W, _ = img_H.shape

        # ------------------------------------
        # 6) Remove margins
        # ------------------------------------
        blurred = blurred[24:-24, 24:-24, :]
        kernel_map = kernel_map[:, 24:-24, 24:-24]
        img_H = img_H[24:-24, 24:-24, :]

        img_H, blurred = util.single2tensor3(img_H), util.single2tensor3(blurred)

        blurred /= kernel_map.sum(axis=0).view(1, H_patch - 48, H_patch - 48).repeat(3,1,1)
        kernel_map /= kernel_map.sum(axis=0).view(1, H_patch - 48, H_patch - 48).repeat(self.pca_size,1,1)

        img_L = blurred[:,::self.sf,::self.sf]

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
        return {'L': img_L, 'H': img_H, 'kmap': kernel_map, 'basis': torch.FloatTensor(basis), 'sigma': noise_level, 'sf': self.sf, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        if self.opt['phase'] == 'train':
            return len(self.ids)
        else:
            return min(len(self.ids), 200)
