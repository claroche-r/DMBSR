from collections import OrderedDict
from models.model_plain import ModelPlain
import numpy as np
from models.CameraShakeModelTwoBranches import CameraShakeModelTwoBranches as CameraShakeModel
from torch.optim import Adam
import torch
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils.homographies import masked_reblur_homographies, Kernels2DLoss, show_positions, CurvatureLoss2, \
                               generarK, mostrar_kernels, RotationsLoss
import utils.utils_image as util

class ModelBlindPMPB(ModelPlain):
    """Train with inputs (L, positions, intrinsics, sf, sigma) and with pixel loss for USRNet"""
    
    def __init__(self, opt):
        super(ModelBlindPMPB, self).__init__(opt)
        dataset_opt = opt['datasets']['train']
        self.n_positions = dataset_opt['n_positions'] if dataset_opt['n_positions'] is not None else 25
        self.pos_network = CameraShakeModel(self.n_positions)
        self.pos_network = self.model_to_device(self.pos_network)
        self.mixed_precision = self.opt_train['mixed_precision']
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        

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
        self.kernels2DLoss = Kernels2DLoss()
        self.curvatureLoss = CurvatureLoss2()
        
        self.reblur_loss_weight = self.opt_train['reblur_loss_weight']
        self.positions_loss_weight = self.opt_train['positions_loss_weight']
        self.kernels2D_loss_weight = self.opt_train['kernels2D_loss_weight']
        self.curvature_loss_weight = self.opt_train['curvature_loss_weight']
        self.rotations_loss_weight = self.opt_train['rotations_loss_weight']
        self.rotationsLoss = RotationsLoss()
        self.grad_accum_iters = self.opt_train['grad_accum_iters'] if self.opt_train['grad_accum_iters'] is not None else 1
        self.iter = 0
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

        if self.mixed_precision:
            #with torch.cuda.amp.autocast():
                #self.estimated_positions = self.pos_network(self.L-0.5)
                
                print('It is not possible to use mixed precision due to inability to compute inverse operation')
                #if self.opt_train['G_lossfn_weight']>0:
                #    self.E = self.netG(self.L, self.estimated_positions, self.intrinsics, self.sf, self.sigma)
                #else:
                #    with torch.no_grad():
                #        self.E = self.netG(self.L, self.estimated_positions, self.intrinsics, self.sf, self.sigma)
        else:
            self.estimated_positions = self.pos_network(self.L-0.5)        
        
        if self.opt_train['G_lossfn_weight']>0:
            self.E = self.netG(self.L, self.estimated_positions, self.intrinsics, self.sf, self.sigma)
        else:
            with torch.no_grad():
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
        
    
    

    def optimize_parameters_mixed_precision(self, current_step):
        
        self.G_optimizer.zero_grad()
        self.K_optimizer.zero_grad()
                     
        self.netG_forward()
            
        reblured_image, mask = masked_reblur_homographies(self.E, self.estimated_positions, self.intrinsics[0])
        
        _, C, H, W = self.H.shape
        
        kernels2D_loss = self.kernels2D_loss_weight * torch.min(self.kernels2DLoss(self.estimated_positions[0], self.positions[0], (H, W, C),  self.intrinsics[0]),
                                            self.kernels2DLoss(torch.flip(self.estimated_positions[0],dims=[0]), self.positions[0], (H, W, C),  self.intrinsics[0]))
                    
       
        reblur_loss = torch.nn.functional.mse_loss(reblured_image, self.L, reduction='none') 
        reblur_loss = self.reblur_loss_weight * reblur_loss.masked_select((mask > 0.9)).mean()       
        
        positions_loss = self.positions_loss_weight * torch.min(torch.nn.functional.mse_loss( self.positions, self.estimated_positions ),
                                                torch.nn.functional.mse_loss( self.positions, torch.flip(self.estimated_positions,dims=[1] )) )
        
        

        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        
        K_loss = reblur_loss + positions_loss + kernels2D_loss
        
        T_loss = G_loss + K_loss
                
        self.scaler.scale(T_loss).backward()
        self.scaler.unscale_(self.G_optimizer)
        self.scaler.unscale_(self.K_optimizer)

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.scaler.step(self.G_optimizer)
        self.scaler.step(self.K_optimizer)

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
        
        self.scaler.update()

    
    def optimize_parameters_no_mixed_precision(self, current_step):
        
        self.netG_forward()
        reblur_loss = torch.Tensor([0]).to(self.E.device); positions_loss=torch.Tensor([0]).to(self.E.device); 
        kernels2D_loss=torch.Tensor([0]).to(self.E.device); curvature_loss=torch.Tensor([0]).to(self.E.device)
        rotations_loss = torch.Tensor([0]).to(self.E.device)
        if self.reblur_loss_weight>0: 
            reblured_image, mask = masked_reblur_homographies(self.H, self.estimated_positions, self.intrinsics[0])
            reblur_loss = torch.nn.functional.mse_loss(reblured_image, self.L, reduction='none') 
            reblur_loss = self.reblur_loss_weight * reblur_loss.masked_select((mask > 0.9)).mean()       
        
        del reblured_image, mask 
        
        if self.positions_loss_weight:
            positions_loss = self.positions_loss_weight * torch.min(torch.nn.functional.mse_loss( self.positions, self.estimated_positions ), 
                        torch.nn.functional.mse_loss( self.positions, torch.flip(self.estimated_positions,dims=[1] )) )
        
        if self.kernels2D_loss_weight:
            _, C, H, W = self.H.shape
            kernels2D_loss = self.kernels2D_loss_weight * torch.min(self.kernels2DLoss(self.estimated_positions[0], self.positions[0], (H, W, C),  self.intrinsics[0]),
                                           self.kernels2DLoss(torch.flip(self.estimated_positions[0],dims=[0]), self.positions[0], (H, W, C),  self.intrinsics[0]))
        
        if self.curvature_loss_weight:
            curvature_loss = self.curvature_loss_weight * self.curvatureLoss(self.estimated_positions)
            print('curvature loss: ',  curvature_loss)
        
        if self.rotations_loss_weight:
            rotations_loss = self.rotations_loss_weight * torch.min(self.rotationsLoss(self.estimated_positions, self.positions),
                                                    self.rotationsLoss(torch.flip(self.estimated_positions, dims=[1]), self.positions))
            print('rotations loss: ',  rotations_loss)
        
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        
        K_loss = reblur_loss + positions_loss + kernels2D_loss  + curvature_loss + rotations_loss
        
        T_loss = (G_loss + K_loss)/self.grad_accum_iters
        
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
            print('weights updated at iter %d' %current_step)
            torch.cuda.empty_cache()


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
        self.save_network(self.save_dir, self.pos_network, 'pos_network', iter_label)

    def current_visuals(self):
        out_dict = super().current_visuals()    

        found_positions_np = self.estimated_positions[0].detach().cpu().numpy() 
        gt_positions_np = self.positions[0].detach().cpu().numpy()                   
        fig = show_positions(found_positions_np, gt_positions_np)
        out_dict['fig']=fig
        
        pose = np.zeros((found_positions_np.shape[0], 6))
        pose[:, 3:] = gt_positions_np
        C,M,N = self.L.shape[1:]
        K, _ = generarK( (M,N,C) , pose)
        kernels_gt =mostrar_kernels(K, (M,N,C), output_name = "kernels_gt.png")
        out_dict['kernels_gt']=kernels_gt*255
        pose[:, 3:] = found_positions_np
        K, _ = generarK((M,N,C), pose)
        kernels_estimated = mostrar_kernels(K, (M,N,C), output_name = "kernels_estimated.png")
        out_dict['kernels_estimated']=kernels_estimated*255
        #fig.savefig(os.path.join(OUTPUT_DIR, 'iter_%d_positions_found.png' % n_update))


        return out_dict
    
    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        self.pos_network.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()
        self.pos_network.train()

