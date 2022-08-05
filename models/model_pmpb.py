from models.model_plain import ModelPlain
import numpy as np


class ModelPMPB(ModelPlain):
    """Train with four inputs (L, k, sf, sigma) and with pixel loss for USRNet"""

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
        self.E = self.netG(self.L, self.positions, self.intrinsics, self.sf, self.sigma)

    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG.module.p, strict=self.opt_train['G_param_strict'], param_key='params')
        
        #for k,v in self.netG.module.p.named_parameters():
            #v.requires_grad = False

