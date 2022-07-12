import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image, ImageDraw
from skimage.io import imsave
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
import os
import numpy as np
import random
from scipy import io
from scipy.ndimage import convolve
#from matplotlib import pyplot as plt

from pycocotools.coco import COCO
#from pycocotools import mask as maskUtils


class COCONonUniformBlurDataset(Dataset):
    def __init__( self, sharp_image_files,  kernel_image_files, sharp_root_dir, kernel_root_dir, crop_size=256,
                 kernel_size=33, max_objects_to_blur=2,
                  transform=None, indices=None, seed=None, gray_scale=False,
                  percentage_of_deltas=0, augment_illumination=True, jitter_illumination=1,
                  gamma_correction=False, gamma_factor=2.2):
        """
        Args:
             sharp_image_files: Path to sharp images list file
             kernel_image_files:  Path to kernels images list file
             sharp_root_dir: sharp images names are relative to this directory
             kernel_root_dir: kernel images names are relative to this directory
             crop_size: blurred images are cropped to this size
             transform: Optional transform to be appeared on a sample
        """
        self.supercategories=['person', 'vehicle','animal']  #
        self.coco = COCO(sharp_image_files)
        self.labels_to_blur = self.coco.getCatIds(supNms=self.supercategories)

        self.crop_size = crop_size
        self.min_objetc_area = 0.05 * self.crop_size ** 2
        self.images_of_interest = []
        keys = self.coco.anns.keys()
        for key in keys:
            min_area_cond = self.coco.anns[key]['area'] > self.min_objetc_area
            is_crowd = self.coco.anns[key]['iscrowd']
            cat_cond = self.coco.anns[key]['category_id'] in self.labels_to_blur
            if cat_cond and min_area_cond and not is_crowd:
                self.images_of_interest.append(self.coco.anns[key]['image_id'])

        self.images_of_interest = np.unique(np.array(self.images_of_interest))
        if indices is not None:
            self.images_of_interest = [self.images_of_interest[i] for i in indices]
        self.sharp_root_dir = sharp_root_dir

        # only the labels of this list are blurred independently of the background
        # full list of ADE classes: https://github.com/CSAILVision/sceneparsing/tree/master/visualizationCode/color150
        kernel_file = open(kernel_image_files, 'r')
        #self.n_kernels = 4  # number of kernels used to generate the data
        #self.kernel_image_files = kernel_file.readlines()[0:self.n_kernels] # se usan los primeros N kernels, está duro
        self.kernel_image_files = kernel_file.readlines() # se usan todos los kernels
        self.kernel_root_dir = kernel_root_dir
        self.kernel_size = kernel_size
        self.percentage_of_deltas = percentage_of_deltas
        self.gray_scale = gray_scale

        self.transform = transform
        self.max_objects_to_blur = max_objects_to_blur
        self.augment_illumination = augment_illumination
        self.jitter_illumination = jitter_illumination
        self.gamma_correction = gamma_correction
        self.gamma_factor = gamma_factor

        self.seed = seed

    def __len__(self):
        return len(self.images_of_interest)


    def get_interesing_crop(self, sharp_image, img_id):
        '''
        :param sharp_image:
        :param img_id:
        :return:
            sharp_image_crop
            obj_masks
        '''


        W = sharp_image.shape[1]
        H = sharp_image.shape[0]
        K = self.kernel_size

        only_background = True
        num_attempt = 0
        max_attempts = 100 if self.max_objects_to_blur > 0  else 1

        annIds = self.coco.getAnnIds(imgIds=img_id, areaRng=[self.min_objetc_area, np.inf], iscrowd=False)
        anns = self.coco.loadAnns(annIds)
        n_objects = len(anns)
        #print('Number of objects in image: ', len(anns))
        while(only_background and num_attempt < max_attempts ):
            try:
                Ws = np.random.randint(0, W - self.crop_size -K + 1)
                Hs = np.random.randint(0, H - self.crop_size -K + 1)
                sharp_image_crop = sharp_image[Hs:Hs + self.crop_size + K - 1, Ws:Ws + self.crop_size + K  - 1, :]
            except:
                print('Error while cropping: W=%d, H=%d, crop_size=%d' % (W, H, self.crop_size))

            ious = []
            selected_indices = []
            for n in range(n_objects):
                bbox = anns[n]['bbox']
                p0_x = bbox[0]
                p0_y = bbox[1]
                p1_x = p0_x + bbox[2]
                p1_y = p0_y + bbox[3]
                i0_x = max(p0_x, Ws + K//2)
                i0_y = max(p0_y, Hs + K//2)
                i1_x = min(p1_x, Ws + K//2  + self.crop_size)
                i1_y = min(p1_y, Hs + K//2 +  self.crop_size)
                if (i0_x < i1_x) and (i0_y < i1_y): # when intersections exists
                    u0_x = min(p0_x, Ws)
                    u0_y = min(p0_y, Hs)
                    u1_x = max(p1_x, Ws + K//2 + self.crop_size)
                    u1_y = max(p1_y, Hs + K//2+ self.crop_size)
                    iou = (i1_x - i0_x)*(i1_y - i0_y)/((u1_x - u0_x)*(u1_y - u0_y))
                    if iou > 0.1:
                        only_background = False
                        selected_indices.append(n)
                        ious.append(iou)
                        #print('Object %d added' % n)
            num_attempt = num_attempt + 1

        orden = np.argsort(np.asarray(ious))
        num_objects = np.min([len(orden), self.max_objects_to_blur])
        #print('%d objects added in image %d' % (num_objects, img_id))
        obj_masks = []
        selected_indices = np.asarray(selected_indices)[orden[:num_objects]]
        if len(selected_indices) > 2:
            print('Ojo indices')
        for ind in selected_indices:
            if type(anns[ind]['segmentation']) == list:
                # polygon
                seg = anns[ind]['segmentation'][0]
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                poly[:,0] -= Ws + K//2
                poly[:,1] -= Hs + K // 2
                mask_to_draw = Image.new('L', (self.crop_size, self.crop_size), 0)
                # print(poly.shape)
                poly_to_draw = np.round(poly.flatten()).astype(np.int)
                # print(poly_to_draw, poly_to_draw.dtype,poly_to_draw.shape)
                ImageDraw.Draw(mask_to_draw).polygon(list(poly_to_draw), outline=1, fill=1)
                mask_k = np.array(mask_to_draw)
                # print(mask.shape, mask.dtype, np.unique(mask), np.sum(mask)/(mask.shape[0] * mask.shape[1]))
                mask_k_area = mask_k.sum()
                if mask_k_area > self.min_objetc_area:
                    obj_masks.append(mask_k)
                #plt.figure()
                #plt.imshow(128 * mask_k, cmap='gray')
                #plt.show()
        # else:
            #     # mask
            #     # t = img[ann['image_id']]
            #     print(anns[ind]['segmentation'])
            #     if type(anns[ind]['segmentation']['counts']) == list:
            #         rle = maskUtils.frPyObjects([anns[ind]['segmentation']], H, W)
            #     else:
            #         rle = [anns[ind]['segmentation']]
            #     m = maskUtils.decode(rle)
            #     # m_img = np.ones( (m.shape[0], m.shape[1], 3) )
            #     # if ann['iscrowd'] == 1:
            #     #    color_mask = np.array([2.0,166.0,101.0])/255
            #     # if ann['iscrowd'] == 0:
            #     #    color_mask = np.random.random((1, 3)).tolist()[0]
            #     # for i in range(3):
            #     #    m_img[:,:,i] = color_mask[i]
            #
            #     if anns[ind]['iscrowd'] == 1:
            #         print('is crowd')
            #     plt.figure()
            #     # plt.imshow(np.dstack( (m_img, m*0.5)), cmap='gray')
            #     plt.imshow(128 * m, cmap='gray')
            #     plt.show()

        if len(obj_masks) > 2:
            print('Ojo')

        return sharp_image_crop, obj_masks

    def get_random_kernel(self):
        random_number = np.random.rand()
        if random_number < self.percentage_of_deltas:
            kernel = np.zeros((self.kernel_size,self.kernel_size))
            kernel[self.kernel_size//2, self.kernel_size//2] = 1
        else:
            # this is the blurry background
            idx_kernel = np.random.randint(len(self.kernel_image_files))
            kernel_name = self.kernel_image_files[idx_kernel][0:-1].split('/')
            kernel = io.loadmat(os.path.join(self.kernel_root_dir, kernel_name[0], kernel_name[1]))['K']
        return kernel


    def generate_blurry_sharp_pair(self, sharp_image_crop, kernels, masks_to_send):
        K = self.kernel_size

        if self.gray_scale:
            sharp_image_crop = (255*rgb2gray(sharp_image_crop)).astype(np.uint8)
            sharp_image_crop = sharp_image_crop[:,:,None]
            blurry_image = np.zeros((sharp_image_crop.shape[0] - K + 1, sharp_image_crop.shape[1] - K + 1, 1),
                                    dtype=np.float32)
        else:
            blurry_image = np.zeros((sharp_image_crop.shape[0]-K+1,sharp_image_crop.shape[1]-K+1,3),  dtype=np.float32)

        if self.gamma_correction:
            sharp_image_crop = 255.0*( (sharp_image_crop/255.0)** self.gamma_factor)

        if self.augment_illumination:
            if sharp_image_crop.dtype == np.uint8:
                sharp_image_crop = rgb2hsv(sharp_image_crop)
                sharp_image_crop[:,:,2] *= (1+self.jitter_illumination*(np.random.rand()-0.5))
                sharp_image_crop = 255*hsv2rgb(sharp_image_crop)
            else:
                sharp_image_crop = rgb2hsv(sharp_image_crop/255)
                sharp_image_crop[:,:,2] *= (1+self.jitter_illumination*(np.random.rand()-0.5))
                sharp_image_crop = 255*hsv2rgb(sharp_image_crop)


        # blurry image is generated
        for i in range(len(kernels)):
            kernel = kernels[i]
            blurry_k = convolve(sharp_image_crop, kernel[::-1, ::-1, None])
            blurry_k = blurry_k[K // 2:-(K // 2), K // 2:-(K // 2)]
            mask = masks_to_send[:,:,i]
            #plt.imshow(mask)
            #plt.show()
            blurry_image += mask[:, :, None] * blurry_k

        return blurry_image, sharp_image_crop

    def __getitem__(self, idx):

        # An image is randomly chosen
        min_size = 0  # image must have more than min_size pixels width/height
        num_channels = 0  # only images with three channels are valid images
        while (min_size < self.crop_size + self.kernel_size) or (num_channels < 3):
            image_id = self.images_of_interest[idx]
            image_data = self.coco.loadImgs([image_id])[0]
            sharp_image = np.array(Image.open(os.path.join(self.sharp_root_dir, image_data['file_name'])))
            min_size = np.min([sharp_image.shape[0],sharp_image.shape[1]])
            num_channels = len(sharp_image.shape)
            idx = np.random.randint(len(self.images_of_interest))

        #kernels.append(kernel)  # kernel associated with the background

        # seg_filename = image_fullname.replace('.jpg', '_seg.png')
        # seg_img = np.array(Image.open(seg_filename))
        #
        # B = np.array(seg_img)[:, :, 2]
        # values_of_segmented_instances = np.unique(B)
        #
        # #print('Number of segmented instance in the image: ', len(values_of_segmented_instances))
        #
        # # Read attributes
        # # format: '%03d; %s;  %d; %s; "%s"\n', instance, name, whole(j)==0, crop, atr
        # # format: Instance, part level (0 for objects), crop, class name, corrected_raw_name, list-of-attributes
        # with open(image_fullname.replace('.jpg', '_atr.txt'), 'r') as f:
        #     text = f.readlines()
        # corrected_raw_names, part_level = [], []
        # for line in text:
        #     data = line.split(' # ')
        #     is_part = int(data[1]) > 0
        #     if not is_part:
        #         corrected_raw_names.append(data[4])
        #
        # #print('Number of labels for instances in the image: ', len(corrected_raw_names))
        #
        # W = sharp_image.shape[1]
        # H = sharp_image.shape[0]
        #
        # if self.seed is not None:
        #     np.random.seed(self.seed)
        # #print(image_name, W, H)
        #
        # only_background = True
        # num_attempt = 0
        # max_attempts = 100 if self.max_objects_to_blur > 0  else 1
        #
        # while(only_background and num_attempt < max_attempts ):
        #     try:
        #         Ws = np.random.randint(0, W - self.crop_size -K + 1)
        #         Hs = np.random.randint(0, H - self.crop_size -K + 1)
        #         sharp_image_crop = sharp_image[Hs:Hs + self.crop_size + K - 1, Ws:Ws + self.crop_size + K  - 1, :]
        #         B_crop = B[ Hs + K//2:Hs + K//2 + self.crop_size, Ws + K//2:Ws + K//2 + self.crop_size]
        #     except:
        #         print('Error while cropping: W=%d, H=%d, crop_size=%d' % (W, H, self.crop_size))
        #     values_of_segmented_instances_in_crop = np.unique(B_crop)
        #     # check that at least one of the objetcs is an object of interest
        #     for val in values_of_segmented_instances_in_crop:
        #         # the name of the label is found
        #         index = np.argwhere(values_of_segmented_instances==val)[0][0]
        #         instance_label = corrected_raw_names[index - 1]
        #         if instance_label in self.labels_to_saturate or instance_label in self.labels_to_blur and index > 0:  # primer elemento no tiene etiqueta
        #             only_background = False
        #     num_attempt +=1
        #
        # #if num_attempt==max_attempts and self.max_objects_to_blur>0 :
        # #    print('Maximum number of attempts reached, no good image found :-(')
        #

        # An interesting crop is chosen
        K = self.kernel_size

        # structures to save mask and kernels.
        masks = []
        kernels = []

        kernel = self.get_random_kernel()
        kernels.append(kernel)
        mask = np.ones((self.crop_size, self.crop_size), dtype=np.float32 )
        masks.append(mask)

        sharp_image_crop, objects_masks = self.get_interesing_crop(sharp_image, image_id)
        for mask in objects_masks:
            kernel = self.get_random_kernel()
            kernels.append(kernel)
            masks.append(mask.astype(np.float32))
            masks[0][mask == 1] = 0  # se ponen a cero las posiciones del background que fueron sustituidas

        #masks[0]=convolve(masks[0], kernels[0][::-1])
        masks_to_send = np.zeros((sharp_image_crop.shape[0]-K+1,sharp_image_crop.shape[1]-K+1,self.max_objects_to_blur+1), dtype=np.float32)
        kernels_to_send = np.zeros((kernel.shape[0],kernel.shape[1], self.max_objects_to_blur+1), dtype=np.float32)

        #print('%d masks to send ' % (len(masks)))
        # masks are convolved with kernels to smooth them and avoid discontinuities
        for i, mask in enumerate(masks):
                kernel = kernels[i]
                mask = convolve(mask, kernel[::-1, ::-1])
                masks_to_send[:,:,i] = mask
                kernels_to_send[:,:,i] = kernel
                
                if len(masks)>2:
                     imsave(f'kernel_{idx}_{i}.png', (kernel-kernel.min())/(kernel.max()-kernel.min()))
                #plt.imshow(kernel)
                #plt.show()
                #plt.imshow(mask)
                #plt.show()


        # masks must be normalized because after filtering the sum is not one any more
        try:
            masks_sum = np.sum(masks_to_send, axis=2)
            masks_to_send = masks_to_send/ ( masks_sum[:,:,None] + 1e-6 )
        except:
            print('%d values in the masks summatory are zero' % np.sum(masks_sum==0))


        blurry_image, sharp_image_crop = self.generate_blurry_sharp_pair(sharp_image_crop, kernels, masks_to_send)

        if len(masks) > 2:
            imsave(f'masks_{idx}.png', masks_to_send)
            imsave(f'blurry_{idx}.png', blurry_image)
            imsave(f'sharp_{idx}.png', sharp_image_crop[K // 2:-(K // 2), K // 2:-(K // 2)])
        #plt.imshow(blurry_image/255)
        #plt.show()
        masks_to_send = self.transform(masks_to_send)
        kernels_to_send = self.transform(kernels_to_send)

        try :
            blurry_image = self.transform(blurry_image/255.)
            blurry_image = torch.clamp(blurry_image, 0.0, 1.0)
            if sharp_image_crop.dtype == np.uint8:
                sharp_image_crop = self.transform(sharp_image_crop)
            else:
                sharp_image_crop = self.transform(sharp_image_crop.astype(np.float32)/255)
        except :
            print("Oops!  Error with image ...")

        return {'sharp_image': sharp_image_crop, 'blurry_image': blurry_image, 'kernels': kernels_to_send, 'masks': masks_to_send}
