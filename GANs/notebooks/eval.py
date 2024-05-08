import numpy as np
import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
import pandas as pd
import sys
sys.path.append('../src')
from model import MainModel
from PIL import Image
import os
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet


def build_res_unet(n_input=1, n_output=2, size=256):
    body = create_body(resnet18(), pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(DEVICE)
    return net_G

hist_path = '../histories/Normal GAN'
splits = ['Train', 'Val']

train_df = pd.read_csv(f"{hist_path}/Train/normal_gan.csv")
val_df = pd.read_csv(f"{hist_path}/Val/normal_gan.csv")

train_df = train_df.groupby('epoch').mean()
val_df = val_df.groupby('epoch').mean()

#plot train vs val loss of generator and discriminator side by side
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(train_df.index, train_df['loss_G'], label='Generator Train Loss')
plt.plot(val_df.index, val_df['loss_G'], label='Generator Val Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_df.index, train_df['loss_D'], label='Discriminator Train Loss')
plt.plot(val_df.index, val_df['loss_D'], label='Discriminator Val Loss')
plt.legend()
plt.show()


#plot the generator loss vs descriminator loss for each split side by side
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
plt.plot(val_df.index, val_df['loss_G'], label='Generator Val Loss')
plt.plot(val_df.index, val_df['loss_D']*10, label='Discriminator Val Loss')
plt.title('Generator vs Discriminator Loss On Validation Set')
plt.legend()
plt.show()

def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

folder = "D:\\AIDS\\S2\\Project\\coco-2017-dataset\\coco2017\\test2017"
gs_images = []
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder, filename)
        if is_grey_scale(image_path):
            gs_images.append(filename)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#net_G = build_res_unet(n_input=1, n_output=2, size=256)
#net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
model = MainModel()
model.load_state_dict(
    torch.load(
        "normal_gan.pt",
        map_location=device
    )
)

random_images = random.sample(gs_images, 9)
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

images = []

for i, image_path in enumerate(random_images):
    img = Image.open(folder+'\\'+image_path)
    img = img.resize((256, 256))
    images.append(img)
    axs[i // 3, i % 3].imshow(img)
    axs[i // 3, i % 3].axis('off')

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 3, figsize=(10, 10))

for img in images:
    # to make it between -1 and 1
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(DEVICE))
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    axs[i // 3, i % 3].imshow(colorized)
    axs[i // 3, i % 3].axis('off')
    
plt.tight_layout()
plt.show()




