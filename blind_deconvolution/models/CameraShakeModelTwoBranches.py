""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F



class CameraShakeModelTwoBranches(nn.Module):
    def __init__(self, n_positions=25, center=False):
        super(CameraShakeModelTwoBranches, self).__init__()

        self.inc_rgb = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.inc_grid_rgb = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.inc_gray = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.n_positions= n_positions
        self.center = center

        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.feat =   nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.feat6_gap = nn.AvgPool2d(8)  #8

        self.feat5_gap = PooledSkip( 1)  # 1024
        self.feat4_gap = PooledSkip( 2)   # 512
        self.feat3_gap = PooledSkip( 4)   # 256
        self.feat2_gap = PooledSkip( 8)   # 256

        self.roll_up1 = Up(1024,1024, 512)
        self.roll_up2 = Up(512,512, 256)
        self.roll_up3 = Up(256,256, 256)
        self.roll_up4 = Up(256,128, 128)

        self.roll_end = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, n_positions*1, kernel_size=8),
        )

        self.pitch_yaw_up1 = Up(1024,1024, 512)
        self.pitch_yaw_up2 = Up(512,512, 256)
        self.pitch_yaw_up3 = Up(256,256, 256)
        self.pitch_yaw_up4 = Up(256,128, 128)

        self.pitch_yaw_end = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, n_positions*2, kernel_size=8),
        )


    def forward(self, x):
        #Encoder
        if x.shape[1]==3:
            x1 = self.inc_rgb(x)
        elif x.shape[1]==5:
            x1 = self.inc_grid_rgb(x)
        else:
            x1 = self.inc_gray(x)
        x1_feat, x2 = self.down1(x1)
        x2_feat, x3 = self.down2(x2)
        x3_feat, x4 = self.down3(x3)
        x4_feat, x5 = self.down4(x4)
        x5_feat, x6 = self.down5(x5)
        x6_feat = self.feat(x6)

        feat6_gap = x6_feat.mean((2,3), keepdim=True) #self.feat6_gap(x6_feat)
        #print('x6_feat: ', x6_feat.shape,'feat6_gap: ' , feat6_gap.shape)
        feat5_gap = self.feat5_gap(x5_feat)
        #print('x5_feat: ', x5_feat.shape,'feat5_gap: ' , feat5_gap.shape)
        feat4_gap = self.feat4_gap(x4_feat)
        #print('x4_feat: ', x4_feat.shape,'feat4_gap: ' , feat4_gap.shape)
        feat3_gap = self.feat3_gap(x3_feat)
        #print('x3_feat: ', x3_feat.shape,'feat3_gap: ' , feat3_gap.shape)
        feat2_gap = self.feat2_gap(x2_feat)

        #print('x2_feat: ', x2_feat.shape,'feat2_gap: ' , feat2_gap.shape)
        #print(feat5_gap.shape, feat4_gap.shape)
        r1 = self.roll_up1(feat6_gap, feat5_gap)
        #print('k1 shape', k1.shape)
        r2 = self.roll_up2(r1, feat4_gap)
        #print('k2 shape', k2.shape)
        r3 = self.roll_up3(r2, feat3_gap)
        #print('k3 shape', k3.shape)
        r4 = self.roll_up4(r3, feat2_gap)
        #print('k4 shape', k4.shape)
        r5 = self.roll_end(r4)
        N, _, H, W = r5.shape  # H and W should be one
        r = r5.view(N, self.n_positions, 1)

        py1 = self.pitch_yaw_up1(feat6_gap, feat5_gap)
        # print('k1 shape', k1.shape)
        py2 = self.pitch_yaw_up2(py1, feat4_gap)
        # print('k2 shape', k2.shape)
        py3 = self.pitch_yaw_up3(py2, feat3_gap)
        # print('k3 shape', k3.shape)
        py4 = self.pitch_yaw_up4(py3, feat2_gap)
        # print('k4 shape', k4.shape)
        py5 = self.pitch_yaw_end(py4)
        N, _, H, W = py5.shape  # H and W should be one
        py = py5.view(N, self.n_positions, 2)

        k= torch.cat([py, r], dim=2)

        if self.center:
            mean_position = torch.mean(k, axis=2)
            k = k -mean_position

        return  k
""" Parts of the U-Net model """


class Down(nn.Module):
    """double conv and then downscaling with maxpool"""

    def __init__(self, in_channels, out_channels, antialiased=False, antialiased_kernel_size=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,  kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
           # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            #nn.BatchNorm2d(out_channels),
        )

        if antialiased:
            self.down_sampling = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=1),
                Downsample(channels=out_channels, filt_size=antialiased_kernel_size, stride=2)
        )
        else:
            self.down_sampling = nn.MaxPool2d(2)


    def forward(self, x):
        feat = self.double_conv(x)
        down_sampled = self.down_sampling(feat)
        return feat, down_sampled


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, feat_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            #print('init bilinear')
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #print('finish bilinear')
        else:
            #print('init conv transpose')
            self.up = nn.ConvTranspose2d(in_channels,  in_channels, kernel_size=2, stride=2)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.feat = nn.Sequential(
            nn.Conv2d(feat_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.double_conv(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        #print('input feat forward', x.shape)
        feat = self.feat(x)
        return feat

class PooledSkip(nn.Module):
    def __init__(self,output_spatial_size):
        super().__init__()

        self.output_spatial_size = output_spatial_size

    def forward(self, x):
        global_avg_pooling = x.mean((2,3), keepdim=True) #self.gap(x)
        #print('gap shape:' , global_avg_pooling.shape)
        return global_avg_pooling.repeat(1,1,self.output_spatial_size,self.output_spatial_size)


class SpatialPooledSkip(nn.Module):
    def __init__(self, input_channels, input_spatial_size, output_spatial_size, antialiased=False,
                 antialiased_kernel_size=3):
        super().__init__()

        if antialiased:
            # self.gap = Downsample(channels=input_channels, filt_size=antialiased_kernel_size, stride=input_spatial_size)
            self.gap = nn.AvgPool2d(input_spatial_size)  # I don know how to change this layer, it is shift equivariant
        else:
            # self.gap = nn.AvgPool2d(input_spatial_size)
            self.gap = nn.AvgPool2d(16, stride=16)

        self.output_spatial_size = output_spatial_size

    def forward(self, x):
        global_avg_pooling = self.gap(x)
        # print('gap shape:' , global_avg_pooling.shape)
        return global_avg_pooling  # global_avg_pooling.repeat(1,1,self.output_spatial_size,self.output_spatial_size)
