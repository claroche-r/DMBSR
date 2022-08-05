from collections import OrderedDict
from models.model_plain import ModelPlain
import numpy as np
from models.CameraShakeModelTwoBranches import CameraShakeModelTwoBranches as CameraShakeModel
from torch.optim import Adam
import torch
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils.homographies import masked_reblur_homographies

class ModelBlindPMPB(ModelPlain):
    """Train with inputs (L, positions, intrinsics, sf, sigma) and with pixel loss for USRNet"""
    
    def __init__(self, opt):
        super(ModelBlindPMPB, self).__init__(opt)
        self.pos_network = CameraShakeModel()
        self.pos_network = self.model_to_device(self.pos_network)
        

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.pos_network.train()  
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.reblur_loss_weight = self.opt_train['reblur_loss_weight']
        self.positions_loss_weight = self.opt_train['positions_loss_weight']
        self.kernels2D_loss_weight = self.opt_train['kernels2D_loss_weight']
        self.log_dict = OrderedDict()         # log
    
    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)  # low-quality image
        self.positions = data['positions'].to(self.device)  # blur kernel
        self.intrinsics = data['intrinsics'].to(self.device)
        self.sf = np.int(data['sf'][0,...].squeeze().cpu().numpy()) # scale factor
        self.sigma = data['sigma'].to(self.device)  # noise level
        self.name_img = data['H_path']
        if need_H:
            self.H = data['H'].to(self.device)  # H

    # ----------------------------------------
    # feed (L, C) to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        self.estimated_positions = self.pos_network(self.L)
        self.E = self.netG(self.L, self.estimated_positions, self.intrinsics, self.sf, self.sigma)

    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG.module.p, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_K = self.opt['path']['pretrained_netK']
        if load_path_K is not None:
            print('Loading model for K [{:s}] ...'.format(load_path_K))
            self.load_network(load_path_K, self.pos_network, strict=self.opt_train['K_param_strict'], param_key='params')
        
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
        for k, v in self.pos_network.named_parameters():
            if v.requires_grad:
                K_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
               
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)
        self.K_optimizer = Adam(K_optim_params, lr=self.opt_train['K_optimizer_lr'], weight_decay=0)
        
    
        # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.K_optimizer.zero_grad()
        
        self.netG_forward()
        
        reblured_image, mask = masked_reblur_homographies(self.E, self.estimated_positions, self.intrinsics)
        reblur_loss = torch.nn.functional.mse_loss(reblured_image, self.L, reduction='none') 
        reblur_loss = self.reblur_loss_weight * reblur_loss.masked_select((mask > 0.9)).mean()       
        
        positions_loss = self.positions_loss_weight * torch.min(torch.nn.functional.mse_loss( self.positions, self.estimated_positions ),
                                                 torch.nn.functional.mse_loss( self.positions, torch.flip(self.estimated_positions,dims=[1] )) )
        
        _, C, H, W = self.H.shape
        kernels2D_loss = self.kernels2D_loss_weight * torch.min(kernels2D_loss(self.estimated_positions[0], self.positions[0], (H, W, C), self.intrinsics),
                                            kernels2D_loss(torch.flip(self.estimated_positions[0],dims=[0]), self.positions[0], (H, W, C), self.intrinsics))
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        
        K_loss = reblur_loss + positions_loss + kernels2D_loss
        
        T_loss = G_loss + K_loss
        
        T_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()
        self.K_optimizer.step()

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