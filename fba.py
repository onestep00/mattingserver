# Our libs
from networks.transforms import trimap_transform, groupnorm_normalise_image
from networks.models import build_model
from dataloader import PredDataset

# System libs
import os
import argparse

# External libs
import cv2
import numpy as np
import torch

class Fba:
  def np_to_torch(self, x):
      return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()


  def scale_input(self, x: np.ndarray, scale: float, scale_type) -> np.ndarray:
      ''' Scales inputs to multiple of 8. '''
      h, w = x.shape[:2]
      h1 = int(np.ceil(scale * h / 8) * 8)
      w1 = int(np.ceil(scale * w / 8) * 8)
      x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
      return x_scale


  def predict_fba_folder(self, model, args):
      save_dir = args.output_dir

      dataset_test = PredDataset(args.image_dir, args.trimap_dir)

      gen = iter(dataset_test)
      for item_dict in gen:
          image_np = item_dict['image']
          trimap_np = item_dict['trimap']

          fg, bg, alpha = pred(image_np, trimap_np, model)

          cv2.imwrite(os.path.join(save_dir, item_dict['name'][:-4] + '_fg.png'), fg[:, :, ::-1] * 255)
          cv2.imwrite(os.path.join(save_dir, item_dict['name'][:-4] + '_bg.png'), bg[:, :, ::-1] * 255)
          cv2.imwrite(os.path.join(save_dir, item_dict['name'][:-4] + '_alpha.png'), alpha * 255)


  def pred(self, image_np: np.ndarray, trimap_np: np.ndarray, model) -> np.ndarray:
      ''' Predict alpha, foreground and background.
          Parameters:
          image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
          trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
          Returns:
          fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
          bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
          alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
      '''
      h, w = trimap_np.shape[:2]

      image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
      trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

      with torch.no_grad():

          image_torch = np_to_torch(image_scale_np)
          trimap_torch = np_to_torch(trimap_scale_np)

          trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np))
          image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')

          output = model(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)

          output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
      alpha = output[:, :, 0]
      fg = output[:, :, 1:4]
      bg = output[:, :, 4:7]

      alpha[trimap_np[:, :, 0] == 1] = 0
      alpha[trimap_np[:, :, 1] == 1] = 1
      fg[alpha == 1] = image_np[alpha == 1]
      bg[alpha == 0] = image_np[alpha == 0]
      return fg, bg, alpha


  def __init__(self, model_dir, model_config):

      parser = argparse.ArgumentParser()
      # Model related arguments
      class Args:
        encoder = 'resnet50_GN_WS'
        decoder = 'fba_decoder'
        weights = 'FBA.pth'
      args=Args()
      model = build_model(args)
      model.eval()
      fg, bg, alpha = pred(image_np, trimap_np, model)
