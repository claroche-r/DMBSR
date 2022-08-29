from collections import OrderedDict
from models.model_plain import ModelPlain
import numpy as np
from models.TwoHeadsNetwork import TwoHeadsNetwork as two_heads
from torch.optim import Adam
import torch
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils.reblur import compute_reblurred_image_and_kernel_loss, get_masks_weights
import utils.utils_image as util

class ModelBlindNIMBUSR(ModelPlain):
    """Train with inputs (L, positions, intrinsics, sf, sigma) and with pixel loss for USRNet"""
    
    def __init__(self, opt):
        super(ModelBlindNIMBUSR, self).__init__(opt)
        self.kernels_network = two_heads(self.opt_train['B'])
        self.kernels_network = self.model_to_device(self.kernels_network)
        self.mixed_precision = self.opt_train['mixed_precision']
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.kernels_network.train()  
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.reblur_loss_weight = self.opt_train['reblur_loss_weight']
        self.kernels2D_loss_weight = self.opt_train['kernels2D_loss_weight']
        self.grad_accum_iters = self.opt_train['grad_accum_iters'] if self.opt_train['grad_accum_iters'] is not None else 1
        self.log_dict = OrderedDict()         # log
    
    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)  # low-quality image
        self.gt_kernels = data['basis'].to(self.device)  # blur kernel
        self.gt_masks = data['kmap'].to(self.device)
        self.sf = np.int(data['sf'][0,...].squeeze().cpu().numpy()) # scale factor
        self.sigma = data['sigma'].to(self.device)  # noise level
        if need_H:
            self.H = data['H'].to(self.device)  # H

    # ----------------------------------------
    # feed (L, C) to netG and get E
    # ----------------------------------------
    def netG_forward(self):

        #oversampled_image = util.imresize(self.L[0].detach().cpu(), 2)[None]    
        self.kernels, self.masks = self.kernels_network(self.L-0.5)  #(oversampled_image.to(self.L.device))
        
        if self.opt_train['G_lossfn_weight']>0:
            self.E = self.netG(self.L, self.masks, self.kernels.flip(dims=(2,3)), self.sf, self.sigma)
        else:
            with torch.no_grad():
                self.E = self.netG(self.L, self.masks, self.kernels.flip(dims=(2,3)),  self.sf, self.sigma)

    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG.module.p, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_K = self.opt['path']['pretrained_netK']
        if load_path_K is not None:
            print('Loading model for K [{:s}] ...'.format(load_path_K))
            self.load_network(load_path_K, self.kernels_network, strict=self.opt_train['K_param_strict'], param_key='params')
        
        #for k,v in self.netG.module.p.named_parameters():
            #v.requires_grad = False

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        K_optim_params = []
        for k, v in self.kernels_network.named_parameters():
            if v.requires_grad:
                K_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
               
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)
        self.K_optimizer = Adam(K_optim_params, lr=self.opt_train['K_optimizer_lr'], weight_decay=0)
        
    

    
    def optimize_parameters_no_mixed_precision(self, current_step):
        
        self.netG_forward()
        #reblur_loss = torch.Tensor([0]).to(self.E.device); 
        #kernels2D_loss=torch.Tensor([0]).to(self.E.device); 
        if self.reblur_loss_weight>0 and  self.kernels2D_loss_weight>0: 
            masks_weights = get_masks_weights(self.gt_masks)
            reblured_image, kernels2D_loss = compute_reblurred_image_and_kernel_loss(self.H, self.kernels, self.masks, self.gt_kernels,
                                                                           self.gt_masks, masks_weights, 'L2')   
            reblur_loss = torch.nn.functional.mse_loss(reblured_image, self.L) 
                   
        B,C,K,K = self.kernels.shape
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H[:,:,K//2:-(K//2),K//2:-(K//2)])
        
        K_loss = reblur_loss + kernels2D_loss  
        
        T_loss = G_loss + K_loss
        
        T_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)


        if current_step % self.grad_accum_iters ==0:
            self.G_optimizer.step()
            self.K_optimizer.step()
            
            self.G_optimizer.zero_grad()
            self.K_optimizer.zero_grad()
            #print('weights updated at iter %d' %current_step)


        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['K_loss'] = K_loss.item() 
        self.log_dict['T_loss'] = T_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])
    
    
    
    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        if self.mixed_precision:
            self.optimize_parameters_mixed_precision(current_step)
        else:
            self.optimize_parameters_no_mixed_precision(current_step)
            
            
    def save(self, iter_label):
        super().save(iter_label)
        self.save_network(self.save_dir, self.kernels_network, 'kernels_network', iter_label)
        
    
    def kernels_grid(self, blurry_image, kernels, masks):          
        '''
        Draw and save CONVOLUTION kernels in the blurry image.
        Notice that computed kernels are CORRELATION kernels, therefore are flipped.
        :param blurry_image: Tensor (channels,M,N)
        :param kernels: Tensor (K,kernel_size,kernel_size)
        :param masks: Tensor (K,M,N)
        :return:
        '''
        
        K = masks.shape[0]
        M = masks.shape[1]
        N = masks.shape[2]
        kernel_size = kernels.shape[1]
        
        blurry_image = blurry_image.cpu().permute((1,2,0)).numpy()
        kernels = kernels.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
    
        grid_to_draw = blurry_image.copy()
        for i in range(kernel_size, M - kernel_size // 2, 2 * kernel_size):
            for j in range(kernel_size, N - kernel_size // 2, 2 * kernel_size):
                kernel_ij = np.zeros((kernel_size, kernel_size))
                for k in range(K):
                    kernel_ij += masks[k, i, j] * kernels[k]
                kernel_ij_norm = (kernel_ij - kernel_ij.min()) / (kernel_ij.max() - kernel_ij.min())
                grid_to_draw[i - kernel_size // 2:i + kernel_size // 2 + 1,
                j - kernel_size // 2:j + kernel_size // 2 + 1, :] = kernel_ij_norm[::-1, ::-1, None]
        
        return 255*grid_to_draw    

    def current_visuals(self):
        out_dict = super().current_visuals()   
        B,C,K,K = self.kernels.shape
        out_dict['H']=out_dict['H'][:,K//2:-(K//2),K//2:-(K//2)] 
        out_dict['gt_kernels_grid'] = self.kernels_grid(out_dict['L'], self.gt_kernels[0], self.gt_masks[0])
        out_dict['kernels_grid'] = self.kernels_grid(out_dict['L'], self.kernels[0], self.masks[0])
        return out_dict
    
    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        self.kernels_network.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()
        self.kernels_network.train()

