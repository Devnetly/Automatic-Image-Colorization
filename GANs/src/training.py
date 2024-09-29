import os
import glob
import torch
from discriminator import PatchDiscriminator
from unet import Unet
from utils import init_model
from torch import nn, optim
from utils import make_dataloaders
from unet import Unet
from dotenv import load_dotenv
load_dotenv()

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 400000
SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(os.getenv('DATASET_PATH'), "coco-2017-dataset")

paths = glob.glob(DATA_DIR+"/coco2017/train2017/*.jpg")
train_paths = paths[:16000] 
val_paths = paths[16000:]

train_dl = make_dataloaders(paths=train_paths, split='train')
val_dl = make_dataloaders(paths=val_paths, split='val')

disriminator = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), DEVICE)
generator = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), DEVICE)

def train(disriminator, generator, loss, optimizer_g, optimizer_d, train_dl, val_dl, epochs, device):
    
    for epoch in range(epochs):
        for data in train_dl:
            
            L = data['L'].to(device)
            ab = data['ab'].to(device)
            images = torch.cat([L, ab], dim=1)

            disriminator.zero_grad()

            output = disriminator(images)
            label = torch.ones_like(output, device=device)

            real_loss = loss(output, label)

            real_loss.backward()

            fake_color = generator(L)
            fake_image = torch.cat([L, fake_color], dim=1)

            output = disriminator(fake_image)
            label = torch.zeros_like(output, device=device)

            fake_loss = loss(output, label)

            fake_loss.backward()

