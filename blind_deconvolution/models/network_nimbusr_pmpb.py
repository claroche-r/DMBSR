import torch
import torch.nn as nn
import models.basicblock as B
import numpy as np
from utils import utils_image as util
from math import sqrt
import os
import subprocess
from utils.homographies import reblur_homographies

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""
def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


def filter_tensor(x, sf=3):
    z = torch.zeros(x.shape)
    z[..., ::sf, ::sf].copy_(x[..., ::sf, ::sf])
    return z


def hadamard(x, kmap):
    # Compute hadamard product (pixel-wise)
    # x: input of shape (C,H,W)
    # kmap: input of shape (H,W)

    C,H,W = x.shape
    kmap = kmap.view(1, H, W)
    kmap = kmap.repeat(C, 1, 1)
    return (x * kmap)


def convolve_tensor(x, k):
    # Compute product convolution
    # x: input of shape (C,H,W)
    # k: input of shape (H_k,W_k)

    H_k, W_k = k.shape
    C, H, W = x.shape
    k = torch.flip(k, dims =(0,1))
    k = k.view(1, 1, H_k, W_k).repeat(C, 1, 1, 1)
    x = x[None]
    x = torch.nn.functional.pad(x, (W_k//2,W_k//2,H_k//2,H_k//2), mode='circular')
    o = torch.nn.functional.conv2d(x, k, groups=C, padding=0, stride=1)
    return o[0]


def cross_correlate_tensor(x, k):
    # x: input of shape (C,H,W)
    # k: input of shape (H_k,W_k)
    
    C, H, W = x.shape
    H_k, W_k = k.shape
    k = k.view(1, 1, H_k, W_k).repeat(C, 1, 1, 1)
    x = x[None]
    x = torch.nn.functional.pad(x, (W_k//2,W_k//2,H_k//2,H_k//2), mode='circular')
    o = torch.nn.functional.conv2d(x, k, groups=C, padding=0, stride=1)
    return o[0]



def pmpb(x, positions, intrinsics):
    # Apply PMPB model blurry = sum(K_i P_i K^{-1} x)
    # x: input of shape (C,H,W)
    # positions: input of shape (P,H,W)
    # intrinsics: input of shape (P,3,3)
    
    y = reblur_homographies(x,positions,intrinsics[0],forward=True)
    return y[0]



def pmpb_batch(x, positions, intrinsics):
    # Apply PMPB model blurry = sum(K_i P_i K^{-1} x)
    # x: input of shape (B,C,H,W)
    # positions: input of shape (B,P,3)
    # intrinsics: input of shape (B,3,3)

    assert len(x) == len(positions) and len(positions) == len(intrinsics), print("Batch size must be the same for all inputs")
    
    return torch.cat([pmpb(x[i:i+1], positions[i:i+1], intrinsics[i:i+1])[None] for i in range(len(x))])


def transpose_pmpb(x, positions, intrinsics):
    # Apply the transpose of PMPB model blurry = sum(K_i P_i K^{-1} x)
    # x: input of shape (C,H,W)
    # positions: input of shape (P,H,W)
    # intrinsics: input of shape (P,H_k,W_k)
    
    assert len(positions) == len(intrinsics), str(len(positions)) + ',' +  str(len(intrinsics))
    y = reblur_homographies(x,positions,intrinsics[0],forward=False)
    
    return y[0]


def transpose_pmpb_batch(x, positions, intrinsics):
    # Apply the transpose of PMPB model model blurry = sum(H_i^T U_i x)
    # x: input of shape (B,C,H,W)
    # posiitons: input of shape (B,P,H,W)
    # intrinsics: input of shape (B,P,H_k,W_k)

    assert len(x) == len(positions) and len(positions) == len(intrinsics), print("Batch size must be the same for all inputs")
    
    return torch.cat([transpose_pmpb(x[i:i+1], positions[i:i+1], intrinsics[i:i+1])[None] for i in range(len(x))])

"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""


class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# --------------------------------------------
"""


class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()
        
    def forward_pos(self, x, STy, alpha, sf):
        I = torch.ones_like(STy) * alpha
        I[...,::sf,::sf] += 1
        return (STy + alpha * x) / I
        
    def forward_zer(self, x, STy, sf):
        res = x
        res[...,::sf,::sf] = STy[...,::sf,::sf]
        return res

    def forward(self, x, STy, alpha, sf, sigma):
        index_zer = (sigma.view(-1) == 0)
        index_pos = (sigma.view(-1) > 0)
        
        res = torch.zeros_like(x)
        
        res[index_zer,...] = self.forward_zer(x[index_zer,...], STy[index_zer,...], sf)
        res[index_pos,...] = self.forward_pos(x[index_pos,...], STy[index_pos,...], alpha[index_pos,...], sf)
        
        return res

"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""

class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x

"""
# --------------------------------------------
#   Main
# --------------------------------------------
"""


class NIMBUSR_PMPB(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(NIMBUSR_PMPB, self).__init__()

        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=(n_iter+1)*3, channel=h_nc)
        self.n = n_iter

    def forward(self, y, positions, intrinsics, sf, sigma):
        '''
        y: tensor, NxCxHxW
        positions: tensor, NxPx3
        intrinsics: tensor, Nx3x3
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''
        
        # Initialization
        STy = upsample(y, sf)
        x_0 = nn.functional.interpolate(y, scale_factor=sf, mode='nearest')
        z_0 = x_0
        h_0 = pmpb_batch(x_0, positions, intrinsics)
        u_0 = torch.zeros_like(z_0)
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))

        for i in range(self.n):
            # Hyper-params
            alpha = ab[:, i:i+1, ...]
            beta = ab[:, i+(self.n+1):i+(self.n+1)+1, ...]
            gamma = ab[:, i+2*(self.n+1):i+2*(self.n+1)+1, ...]

            # ADMM steps
            i_0 = x_0 - beta * transpose_pmpb_batch(h_0 - z_0 + u_0, positions, intrinsics)
            x_0 = self.p(torch.cat((i_0, gamma.repeat(1, 1, i_0.size(2), i_0.size(3))), dim=1))
            h_0 = pmpb_batch(x_0, positions, intrinsics)
            z_0 = self.d(h_0 + u_0, STy, alpha, sf, sigma)
            u_0 = u_0 + h_0 - z_0

        # Hyper-params
        beta = ab[:, 2*self.n+1:2*(self.n+1), ...]
        gamma = ab[:, 3*self.n+2:, ...]

        i_0 = x_0 - beta * transpose_pmpb_batch(h_0 - z_0 + u_0, positions, intrinsics)
        x_0 = self.p(torch.cat((i_0, gamma.repeat(1, 1, i_0.size(2), i_0.size(3))), dim=1))

        return x_0
