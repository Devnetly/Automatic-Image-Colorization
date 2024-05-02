import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

class ColorizationDataset(Dataset):
    def __init__(self, data_dir, input_size):
        self.data_dir = data_dir
        self.input_size = input_size
        self.image_paths = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_paths[idx])
        bgr_image = cv2.imread(img_path)
        bgr_image = cv2.resize(bgr_image, (256, 256))
        bgr_image = (bgr_image / 255.0).astype(np.float32)

        # convert to rgb
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # get l,a,b channels
        orig_l = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)[:, :, :1] 
        orig_a = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)[:, :, 1] 
        orig_b = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)[:, :, 2] 
        orig_ab = np.stack((orig_a, orig_b), axis=-1)


        img_l = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
        img_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))

        

        return rgb_image, orig_l, orig_ab, img_gray_rgb
        


    

