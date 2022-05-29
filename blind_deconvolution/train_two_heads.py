import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
from models.TwoHeadsNetwork import TwoHeadsNetwork 

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from datasets.COCONonUniformBlurDataset import COCONonUniformBlurDataset

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from utils.visualization import Tensorboard_Visualizer

import torch.multiprocessing as mp

import traceback

from utils.reblur import compute_reblurred_image_and_kernel_loss, \
                         compute_kernels_regularization_loss, \
                         compute_masks_regularization_loss, forward_reblur

import time

parser = argparse.ArgumentParser(description="Kernel estimator")
parser.add_argument("-e", "--epochs", type=int, default=1000)
parser.add_argument("-se", "--start_epoch", type=int, default=0)  # not used so far
parser.add_argument("-b", "--batchsize", type=int, default=4)
parser.add_argument("-s", "--imagesize", type=int, default=256)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-d", "--dataset", help='dataset used: [ADE20K, ADE20K_Street, BSD, ADE20K_Lights]', type=str, default='ADE20K')
parser.add_argument('-mob' ,'--max_objects_to_blur', type=int, help='maximum number of objects to blur when using ADE', default=2)
parser.add_argument("-dr", "--dataset_reblur", help='dataset used ony to deblur: [GoPro]', type=str, default='GoPro')
parser.add_argument('-o' ,'--output_dir', help='directory to output the results', default='./checkpoints/two_heads')
parser.add_argument('-m' ,'--model_name', help='model name', default='non_uniform_kernel.pkl')
parser.add_argument('-k' ,'--n_kernels', type=int, help='number of kernels to estimate', default=25)
parser.add_argument('-ks', '--kernel_size', type=int, help='blur kernel size', default=33)
parser.add_argument('-ket', '--kernel_exposure_time', type=str, help='blur kernel exposure time', default='1')
parser.add_argument('-l', '--loss', help='optimization loss, [mse, kld, kernel, kernel+mse, kernel+px_space, kernel+px_space, kernel+mse+grad, kernel+grad, kernel+mse+reblur, kernel+grad+reblur, kernel+mse+grad+reblur]', default='kernel+mse')
parser.add_argument('-klf', '--kernel_loss_factor', type=float, help='kernel loss factor', default=1.0)
parser.add_argument('-klt', '--kernel_loss_type', type=str, help='kernel loss type', default='L2')
parser.add_argument('-rlf', '--reblur_loss_factor', type=float, help='reblur loss factor', default=1.0)
parser.add_argument('-a','--architecture', type=str, help='architecture to use, [base_model, Xia, simplified, TwoSteps]', default='Xia')
parser.add_argument('-krlf', '--kernels_regularization_loss_factor', type=float, help='kernel regularization loss factor', default=0.0)
parser.add_argument('-krt', '--kernels_regularization_type', type=str, help='kernel regularization type [none, L1, L2, TV, Lp]', default='none')
parser.add_argument('-mrlf', '--masks_regularization_loss_factor', type=float, help='masks regularization loss factor', default=0.0)
parser.add_argument('-mrt', '--masks_regularization_type', type=str, help='masks regularization type [none, L2, TV]', default='none')
parser.add_argument('-sf', '--subsampling_factor', type=int, help='subsampling factor', default=1)
parser.add_argument('--augment_illumination', default=False, action='store_true', help='whether to augment illumination')
parser.add_argument('--gamma_correction', default=False, action='store_true', help='whether to perform gamma_correction')
parser.add_argument('--gamma_factor', type=float, default=2.2, help='gamma correction factor')
parser.add_argument('--superres_factor', type=int, default=1, help='super-resolution factor')
args = parser.parse_args()

# Hyper Parameters
#METHOD = "DEFAULT"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
START_EPOCH = args.start_epoch
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize
DATASET = args.dataset
DATASET_REBLUR = args.dataset_reblur
OUTPUT_DIR = args.output_dir
K = args.n_kernels  # number of filters
KERNEL_SIZE = args.kernel_size
EXPOSURE_TIME = args.kernel_exposure_time
MODEL_NAME = args.model_name
MODEL_PREFIX = MODEL_NAME.split('/')[-1]
MODEL_PREFIX = MODEL_PREFIX[:-4]
LOAD_MODEL = True
EPOCHS_TO_SAVE = [1,2,5,10,25,50,100, 150,200,300,400,500,750,1000,2000,3000,4000,5000]
DISPLAY_PROGRESS_EVERY_N_UPDATES = 100
LOSS = args.loss
KERNEL_LOSS_TYPE = args.kernel_loss_type
KERNEL_LOSS_FACTOR = args.kernel_loss_factor
REBLUR_LOSS_FACTOR = args.reblur_loss_factor
ARCHITECTURE = args.architecture
KERNELS_REGULARIZATION_TYPE = args.kernels_regularization_type
KERNELS_REGULARIZATION_LOSS_FACTOR = args.kernels_regularization_loss_factor
MASKS_REGULARIZATION_TYPE = args.masks_regularization_type
MASKS_REGULARIZATION_LOSS_FACTOR = args.masks_regularization_loss_factor
VALIDATE = True
METHOD = ARCHITECTURE + '_K' + str(K) + '_krt_' + KERNELS_REGULARIZATION_TYPE + '_' + str(KERNELS_REGULARIZATION_LOSS_FACTOR)   + \
         '_mrt_' + MASKS_REGULARIZATION_TYPE + '_' + str(MASKS_REGULARIZATION_LOSS_FACTOR) + '_bks_' + str(KERNEL_SIZE) + '_ket_' + str(EXPOSURE_TIME)
AUGMENT_ILLUMINATION = args.augment_illumination
GAMMA_CORRECTION = args.gamma_correction
GAMMA_FACTOR = args.gamma_factor
MAX_OBJECTS_TO_BLUR = args.max_objects_to_blur
SUPERRES_FACTOR = args.superres_factor



def save_deblur_images(images, iteration, epoch):
    filename = './checkpoints/' + METHOD + "/epoch" + str(epoch) + "/" + "Iter_" + str(iteration) + "_deblur.png"
    if not os.path.exists('./checkpoints/%s' % METHOD ):
        os.makedirs('./checkpoints/%s' % METHOD)
    torchvision.utils.save_image(images, filename)


def weight_init(m, verbose=True):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if verbose:
            print('initializing ' + classname)
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0.0, 0.5 * math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        if verbose:
            print('initializing ' + classname)
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        if verbose:
            print('initializing ' + classname)
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def compute_kernels_from_base(base_kernels, masks):
    output_kernel = torch.empty((masks.shape[-2],masks.shape[-1],base_kernels.shape[-2]*base_kernels.shape[-1])).cuda(GPU)
    for k in range(base_kernels.shape[0]):
        kernel_k = base_kernels[ k, :, :].view(-1)
        masks_k = masks[k, :, :]
        aux = masks_k[:, :, None] * kernel_k[None, None, :]
        output_kernel += aux
        del aux
        torch.cuda.empty_cache()
    return output_kernel


def main():

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    try:
        print("cuda is available: ", torch.cuda.is_available())
        torch.cuda.set_device(GPU)
        
        losses = ['mse', 'kernel+mse']
        if not LOSS in losses:
            print('loss must be one of these: ', losses)
            return

      
        two_heads = TwoHeadsNetwork(K)
        two_heads.apply(weight_init).cuda(GPU)
        summary(two_heads,(3,256,256))
        #print(two_heads.parameters())

        if os.path.exists(OUTPUT_DIR) == False:
            os.system('mkdir -p ' + OUTPUT_DIR )

        if os.path.exists(OUTPUT_DIR + METHOD) == False:
            os.system('mkdir ' + os.path.join(OUTPUT_DIR, METHOD))

        if os.path.exists(MODEL_NAME) and LOAD_MODEL:
            two_heads.load_state_dict(torch.load(MODEL_NAME,  map_location='cuda:%d' % GPU), strict=False)
            print("load two_heads success")

        two_heads_optim = torch.optim.Adam(two_heads.parameters(), lr=LEARNING_RATE)

        train_dataset = COCONonUniformBlurDataset(
            sharp_image_files='./data/COCO/annotations/instances_train2017.json',
            kernel_image_files='./data/kernel_dataset/blur_kernels_train.txt',
            sharp_root_dir='./data/COCO/images/train2017/',
            kernel_root_dir='./data/kernel_dataset/' + 'size_' + str(KERNEL_SIZE) + '_exp_' + str(EXPOSURE_TIME),
            crop_size=IMAGE_SIZE,
            kernel_size=KERNEL_SIZE,
            max_objects_to_blur=MAX_OBJECTS_TO_BLUR,
            augment_illumination=AUGMENT_ILLUMINATION,
            gamma_correction=GAMMA_CORRECTION, gamma_factor=2.2,
            # indices=[1263, 50, 700, 1000],
            # seed=33,
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))


        test_dataset = COCONonUniformBlurDataset(
            sharp_image_files='./data/COCO/annotations/instances_val2017.json',
            kernel_image_files='./data/kernel_dataset/blur_kernels_test.txt',
            sharp_root_dir='./data/COCO/images/val2017/',
            kernel_root_dir='./data/kernel_dataset/' + 'size_' + str(KERNEL_SIZE) + '_exp_' + str(EXPOSURE_TIME),
            crop_size=IMAGE_SIZE,
            kernel_size=KERNEL_SIZE,
            max_objects_to_blur=MAX_OBJECTS_TO_BLUR,
            augment_illumination=AUGMENT_ILLUMINATION,
            # indices=[4],
            transform=transforms.Compose([
                transforms.ToTensor()
            ]))

        
        coco_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=6000)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=BATCH_SIZE//4, sampler=coco_sampler)       
        print('data loader length = %d' % len(train_dataloader))

        number_of_steps_per_epoch = len(train_dataloader)
        two_heads_network_scheduler = StepLR(two_heads_optim, step_size=number_of_steps_per_epoch*EPOCHS//2, gamma=0.5)

        if VALIDATE == True:
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        writer = Tensorboard_Visualizer(OUTPUT_DIR)
        n_update=0  # number of updates

        mse = nn.MSELoss().cuda(GPU)

        print("Training...")
        for epoch in range(START_EPOCH, EPOCHS):
            start_time = time.time() # it is used to show time elapsed between 50 updates

            for iteration, images in enumerate(train_dataloader):
                gt_sharp = (images['sharp_image'] - 0.5).cuda(GPU)
                gt_kernels = images['kernels'].cuda(GPU)
                gt_masks = images['masks'].cuda(GPU)

                N = gt_sharp.size(0)
                C = gt_sharp.size(1)
                H = gt_sharp.size(2) - (gt_kernels.size(2) - 1)
                W = gt_sharp.size(3) - (gt_kernels.size(3) - 1)

                blur_image = (images['blurry_image'] - 0.5).cuda(GPU)
                kernels, masks = two_heads(blur_image)

                # store the contribution of each ground-truth mask,
                gt_masks_sums = gt_masks.sum(3).sum(2)

                # for each mask pixel is stored the contribution os its associated mask, size=(N, H, H)
                masks_areas = (gt_masks * gt_masks_sums[:, :, np.newaxis, np.newaxis]).sum(1)

                # when calculating the kernel differences for each pixel, the weight each pixel will have will be
                # inversely proportional to its associated mask area

                try:
                    #print(gt_masks.shape)
                    roi = (torch.max(gt_masks > 0.95, (1)).values).float() # only pixels with dominant kernel are used to compute loss
                    #print(roi.shape, roi.dtype, masks_areas.shape)
                    masks_weights = 1.0 / (masks_areas + 1) * roi
                    #print(masks_weights.shape)
                except:
                    print('Error computing mask weights')


                mse_loss = torch.Tensor([0.]).cuda(GPU)

                if 'kernel+mse' in LOSS:
                    reblurred_images, kernels_loss = compute_reblurred_image_and_kernel_loss(
                                gt_sharp, kernels, masks, gt_kernels, gt_masks, masks_weights,
                                KERNEL_LOSS_TYPE, GPU, stride=SUPERRES_FACTOR)
                    reblur_diff = (reblurred_images - blur_image)**2 * masks_weights[:,np.newaxis,:,:]
                    mse_loss = reblur_diff.sum()/N

                elif 'mse' in LOSS:
                    reblurred_images = forward_reblur(gt_sharp, kernels, masks, GPU)
                    reblur_diff = (reblurred_images - blur_image)**2 * masks_weights[:,np.newaxis,:,:]
                    mse_loss = reblur_diff.sum()/N
                    kernels_loss = torch.Tensor([0.]).cuda(GPU)


                kernels_regularization_loss = compute_kernels_regularization_loss(kernels, KERNELS_REGULARIZATION_TYPE,
                                                                                  KERNELS_REGULARIZATION_LOSS_FACTOR)

                masks_regularization_loss = compute_masks_regularization_loss(masks, MASKS_REGULARIZATION_TYPE, MASKS_REGULARIZATION_LOSS_FACTOR)

                loss = mse_loss*REBLUR_LOSS_FACTOR  + KERNEL_LOSS_FACTOR * kernels_loss + kernels_regularization_loss + masks_regularization_loss      # 2.8GB GPU usage
                if n_update % DISPLAY_PROGRESS_EVERY_N_UPDATES == 0:
                    print('mse_loss', mse_loss.item(), 'kernel_loss', kernels_loss.item(), 'KERNEL FACTOR', KERNEL_LOSS_FACTOR, 'N', N)
                    writer.add_scalar('learning rate', get_lr(two_heads_optim), n_update)
                    writer.add_scalar('Loss/train/con_mascaras\(kernels\)', kernels_loss, n_update)
                    writer.add_scalar('Loss/train/con_mascaras (reblur)', mse_loss, n_update)
                    writer.add_scalar('Loss/train/con_mascaras (kernels + reblur)', loss, n_update)
                    torch.cuda.empty_cache()   # 4GB


                two_heads.zero_grad()
                loss.backward()
                two_heads_optim.step()
                two_heads_network_scheduler.step()


                if iteration == 0 and (epoch % 10 ==0 ): # first image of the epoch

                    ############################################################################
                    ################       START  IMAGES LOG             #######################
                    ############################################################################
                    # create grid of images (assume images between 0 and 1)
                    blur_img_grid = torchvision.utils.make_grid(images['blurry_image'])
                    writer.add_image('train/imgs_input', blur_img_grid, n_update)

                    if 'mse' in LOSS or 'px_space' in LOSS:
                        reblur_img_grid = torchvision.utils.make_grid(reblurred_images + 0.5)
                        writer.add_image('train/imgs_reblur', reblur_img_grid, n_update)

                        diff_img_grid = torchvision.utils.make_grid(torch.abs(reblurred_images - blur_image),
                                                                    normalize=True)
                        writer.add_image('train/imgs_diff_abs_img', diff_img_grid, n_update)


                    gt_img_grid = torchvision.utils.make_grid(gt_sharp[:,:,
                                                              gt_kernels.size(2) // 2:-(gt_kernels.size(2) // 2),
                                                              gt_kernels.size(3) // 2:-(gt_kernels.size(3) // 2)] + 0.5)
                    writer.add_image('train/imgs_sharp', torch.clamp(gt_img_grid,0.0,1.0), n_update)

                    for n in range(gt_sharp.shape[0]):

                        gt_kernels_n = gt_kernels[n]
                        writer.show_kernels(gt_kernels_n, 'train/gt_input_kernels/image%d' % n, n_update)

                        gt_masks_n = gt_masks[n]
                        writer.show_masks(gt_masks_n, 'train/gt_input_masks/image%d'  % n, n_update)

                        kernels_n = kernels[n, :, :, :]
                        writer.show_kernels(kernels_n, 'train/blur_kernels/image%d' % n, n_update)

                        mask_n = masks[n, :, :, :]
                        writer.show_masks(mask_n, 'train/blur_masks/image%d' % n, n_update)


                        writer.show_kernels_grid(reblurred_images[n].detach()+0.5,kernels_n, mask_n, 'train/kernels_grid/image%d' % n, n_update)
                        writer.show_kernels_grid(torch.clamp(reblurred_images[n].detach() + 0.5, 0, 1), kernels_n, mask_n,
                                                 'train/kernels_grid/image%d' % n, n_update)
                        writer.show_kernels_grid(images['blurry_image'][n], gt_kernels_n, gt_masks_n,'train/kernels_grid/image%d_gt' % n, n_update)

                    print('Training images saved')
                    ############################################################################
                    ################       END  IMAGES LOG             #########################
                    ############################################################################



                if iteration % DISPLAY_PROGRESS_EVERY_N_UPDATES == 0:
                    stop_time = time.time()
                    print('update %d in epoch %d  iteration %d, time elapsed = %.02f seconds' % (
                    n_update, epoch, iteration, stop_time - start_time))
                    start_time = time.time()

                ########################################################################################################
                ##############################  VALIDATION  ############################################################
                ########################################################################################################

                if iteration == (len(train_dataloader) - 1) and VALIDATE is True:  # last iter of the epoch

                    print("Validating with masks...")
                    start = time.time()
                    accumulated_kernels_loss_val, accumulated_reblur_loss_val = 0, 0
                    for iter_val, images_val in enumerate(test_dataloader):
                        with torch.no_grad():
                            blur_image_val = Variable(images_val['blurry_image'] - 0.5).cuda(GPU)
                            gt_val = Variable(images_val['sharp_image'] - 0.5).cuda(GPU)
                            gt_kernels_val = images_val['kernels'].cuda(GPU)
                            gt_masks_val = images_val['masks'].cuda(GPU)

                            kernels_val, masks_val = two_heads(blur_image_val)
                            # store the contribution of each ground-truth mask,
                            gt_masks_val_sums = gt_masks_val.sum(3).sum(2)
                            # for each mask pixel is stored the contribution os its associated mask, size=(N, H, H)
                            masks_areas_val = (gt_masks_val * gt_masks_val_sums[:, :, np.newaxis, np.newaxis]).sum(1)
                            # when calculating the kernel differences for each pixel, the weight each pixel will have will be
                            # inversely proportional to its associated mask area
                            try:
                                # print(gt_masks.shape)
                                roi = (torch.max(gt_masks_val > 0.95, ( 1)).values).float()  # only pixels with dominant kernel are used to compute loss
                                # print(roi.shape, roi.dtype, masks_areas.shape)
                                masks_weights_val = 1.0 / (masks_areas_val + 1) * roi
                                # print(masks_weights.shape)
                            except:
                                print('Error computing mask weights')

                            reblurred_images_val, kernels_loss_val = compute_reblurred_image_and_kernel_loss(
                                gt_val, kernels_val, masks_val, gt_kernels_val, gt_masks_val, masks_weights_val,
                                KERNEL_LOSS_TYPE, GPU, stride=SUPERRES_FACTOR)

                            loss_val = mse(blur_image_val, reblurred_images_val)
                            accumulated_reblur_loss_val += loss_val.item()
                            accumulated_kernels_loss_val += kernels_loss_val.item()

                            if (epoch % 2==0) and iter_val == (len(test_dataloader) - 1):

                                # create grid of images (assume images between 0 and 1)
                                blur_img_grid_val = torchvision.utils.make_grid(images_val['blurry_image'])
                                reblurred_images_val = torch.clamp((reblurred_images_val + 0.5), 0, 1)

                                reblurred_img_grid_val = torchvision.utils.make_grid(reblurred_images_val)
                                sharp_img_grid_val = torchvision.utils.make_grid(images_val[
                                                                                 'sharp_image'])  # print(images['blur_image'].min(),images['blur_image'].max())

                                # write to tensorboard
                                writer.add_image('val/imgs_input_val', blur_img_grid_val, n_update)
                                writer.add_image('val/imgs_reblurred_val', reblurred_img_grid_val, n_update)
                                writer.add_image('val/imgs_sharp_val', torch.clamp(sharp_img_grid_val,0.0,1.0) , n_update)

                                # writer.add_graph(two_heads, images['blur_image'].cuda(GPU))

                                for n in range(gt_val.shape[0]):
                                    gt_kernels_val_n = gt_kernels_val[n]
                                    writer.show_kernels(gt_kernels_val_n, 'val/gt_input_kernels/image%d' % n, n_update, scale_each=True)

                                    gt_masks_val_n = gt_masks_val[n]
                                    writer.show_masks(gt_masks_val_n, 'val/gt_input_masks/image%d' % n, n_update)

                                    kernels_val_n = kernels_val[n, :, :, :]
                                    writer.show_kernels(kernels_val_n, 'val/blur_kernels/image%d' % n, n_update, scale_each=True)

                                    mask_val_n = masks_val[n, :, :, :]
                                    writer.show_masks(mask_val_n, 'val/blur_masks/image%d' % n, n_update)

                                    writer.show_kernels_grid(reblurred_images_val[n].detach(),kernels_val_n, mask_val_n, 'val/kernels_grid/image%d' % n, n_update)
                                    writer.show_kernels_grid(images_val['blurry_image'][n], gt_kernels_val_n, gt_masks_val_n,'val/kernels_grid/image%d_gt' % n, n_update)
                                print('Validation images saved')

                    stop = time.time()
                    accumulated_reblur_loss_val = accumulated_reblur_loss_val / len(test_dataloader)
                    accumulated_kernels_loss_val = accumulated_kernels_loss_val / len(test_dataloader)
                    writer.add_scalar('Loss/val', accumulated_reblur_loss_val, n_update)
                    writer.add_scalar('Loss/val_kernels', accumulated_kernels_loss_val, n_update)
                    print('Validation with masks finished, time elapsed = %.02f, kernels loss = %f, reblur loss = %f ' %( stop - start, accumulated_kernels_loss_val, accumulated_reblur_loss_val))


                n_update += 1


            if epoch in EPOCHS_TO_SAVE:
                current_epoch_model = os.path.join(OUTPUT_DIR, METHOD, MODEL_PREFIX + '_epoch%d.pkl' % epoch)
                torch.save(two_heads.state_dict(), current_epoch_model)



            torch.save(two_heads.state_dict(), os.path.join(OUTPUT_DIR, METHOD, MODEL_PREFIX))

        #
        #     #break
        #
        #
        # writer.close()
        return

    except Exception as e:
        torch.cuda.empty_cache()
        print(e)
        traceback.print_exc()
        return

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
