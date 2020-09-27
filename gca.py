import os
import cv2
import toml
import numpy as np

import torch
from torch.nn import functional as F

import utils
from   utils import CONFIG
import networks

class Gca:
    def __init__(self, model_dir, model_config):
        print("Loading GCA-Matting...")
        with open(model_config) as f:
            utils.load_config(toml.load(f))
        
        self.net = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder)
        
        if torch.cuda.is_available():
            checkpoint = torch.load(model_dir)
            self.net.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
            self.net.cuda()
        else:
            checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
            self.net.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
        self.net.eval()

    def single_inference(self, image_dict, return_offset=False):
        with torch.no_grad():
            image, trimap = image_dict['image'], image_dict['trimap']
            alpha_shape = image_dict['alpha_shape']
            
            if torch.cuda.is_available():
                image = image.cuda()
                trimap = trimap.cuda()
            alpha_pred, info_dict = self.net(image, trimap)

            if CONFIG.model.trimap_channel == 3:
                trimap_argmax = trimap.argmax(dim=1, keepdim=True)

            alpha_pred[trimap_argmax == 2] = 1
            alpha_pred[trimap_argmax == 0] = 0

            h, w = alpha_shape
            test_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
            test_pred = test_pred.astype(np.uint8)
            test_pred = test_pred[32:h+32, 32:w+32]

            if return_offset:
                short_side = h if h < w else w
                ratio = 512 / short_side
                offset_1 = utils.flow_to_image(info_dict['offset_1'][0][0,...].data.cpu().numpy()).astype(np.uint8)
                # write softmax_scale to offset image
                scale = info_dict['offset_1'][1].cpu()
                offset_1 = cv2.resize(offset_1, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
                text = 'unknown: {:.2f}, known: {:.2f}'.format(scale[-1,0].item(), scale[-1,1].item())
                offset_1 = cv2.putText(offset_1, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, thickness=2)

                offset_2 = utils.flow_to_image(info_dict['offset_2'][0][0,...].data.cpu().numpy()).astype(np.uint8)
                # write softmax_scale to offset image
                scale = info_dict['offset_2'][1].cpu()
                offset_2 = cv2.resize(offset_2, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
                text = 'unknown: {:.2f}, known: {:.2f}'.format(scale[-1,0].item(), scale[-1,1].item())
                offset_2 = cv2.putText(offset_2, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, thickness=2)

                return test_pred, (offset_1, offset_2)
            else:
                return test_pred, None


    def generator_tensor_dict(self, image, trimap):
        # read images
        sample = {'image': image, 'trimap': trimap, 'alpha_shape': trimap.shape}

        # reshape
        h, w = sample["alpha_shape"]
        
        if h % 32 == 0 and w % 32 == 0:
            padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
            padded_trimap = np.pad(sample['trimap'], ((32,32), (32, 32)), mode="reflect")
            sample['image'] = padded_image
            sample['trimap'] = padded_trimap
        else:
            target_h = 32 * ((h - 1) // 32 + 1)
            target_w = 32 * ((w - 1) // 32 + 1)
            pad_h = target_h - h
            pad_w = target_w - w
            padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
            padded_trimap = np.pad(sample['trimap'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
            sample['image'] = padded_image
            sample['trimap'] = padded_trimap

        # ImageNet mean & std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        # convert GBR images to RGB
        image, trimap = sample['image'][:,:,::-1], sample['trimap']
        # swap color axis
        image = image.transpose((2, 0, 1)).astype(np.float32)

        # kernel_size_row = 3
        # kernel_size_col = 3
        # kernel = np.ones((3, 3), np.uint8)

        # trimap = cv2.erode(trimap, kernel, iterations=3)
        # trimap = cv2.GaussianBlur(trimap,(15,15),cv2.BORDER_DEFAULT)

        trimap[trimap < 40] = 0
        trimap[trimap >= 220] = 2
        trimap[trimap >= 40] = 1
        # normalize image
        image /= 255.

        # to tensor
        sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
        sample['image'] = sample['image'].sub_(mean).div_(std)

        if CONFIG.model.trimap_channel == 3:
            sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()
        elif CONFIG.model.trimap_channel == 1:
            sample['trimap'] = sample['trimap'][None, ...].float()
        else:
            raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

        # add first channel
        sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

        return sample

    def run(self, image, trimap):
        image_dict = self.generator_tensor_dict(image, trimap)
        pred, offset = self.single_inference(image_dict)

        return pred