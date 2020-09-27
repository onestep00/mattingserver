import sys
sys.path.insert(0, 'U-2-Net')

from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image

# from data_loader import RescaleT
# from data_loader import ToTensorLab

from model import U2NET  # full size version 173.6 MB
# from model import U2NETP # small version u2net 4.7 MB

# model_dir = './saved_models/u2net/u2net.pth'

class U2net:
    def __init__(self, model_dir, image_size):
        print("Loading U-2-Net...")
        self.image_size = int(image_size)
        self.net = U2NET(3, 1)
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_dir))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
        self.net.eval()


    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn


    def preprocess(self, image):
        # image = Image.fromarray(image, 'RGB')

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        sample = transform(image)

        return sample

    # def preprocess2(self, image):
    #     label_3 = np.zeros(image.shape)
    #     label = np.zeros(label_3.shape[0:2])

    #     if (3 == len(label_3.shape)):
    #         label = label_3[:, :, 0]
    #     elif (2 == len(label_3.shape)):
    #         label = label_3

    #     if (3 == len(image.shape) and 2 == len(label.shape)):
    #         label = label[:, :, np.newaxis]
    #     elif (2 == len(image.shape) and 2 == len(label.shape)):
    #         image = image[:, :, np.newaxis]
    #         label = label[:, :, np.newaxis]

    #     transform = transforms.Compose([RescaleT(self.image_size), ToTensorLab(flag=0)])
    #     sample = transform({
    #         'imidx': np.array([0]),
    #         'image': image,
    #         'label': label
    #     })

    #     return sample


    def run(self, img):
        torch.cuda.empty_cache()

        sample = self.preprocess(img)
        inputs_test = sample.unsqueeze(0)
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = self.net(inputs_test)

        # Normalization.
        pred = d1[:, 0, :, :]
        predict = self.normPRED(pred)

        # Convert to PIL Image
        # COnvert to np
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        # im = Image.fromarray(predict_np * 255).convert('RGB')
        im = predict_np * 255

        # Cleanup.
        del d1, d2, d3, d4, d5, d6, d7

        return im