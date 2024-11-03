import os
import glob
import torch
import pandas as pd
from discriminator import PatchDiscriminator
from unet import Unet
from utils import init_model, create_loss_meters, make_dataloaders, build_res_unet
from torch import nn, optim
from unet import Unet
from argparse import ArgumentParser
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv()

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 2
DISC_START_EPOCH = 1
SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(os.getenv('DATASET_PATH'), "coco-2017-dataset")
HISTORIES_DIR = os.getenv('HISTORIES_PATH')
MODELS_DIR = os.getenv('MODELS_PATH')
DISPLAY_EVERY = 100

paths = glob.glob(DATA_DIR+"/coco2017/train2017/*.jpg")

def train(
    disriminator, 
    generator, 
    rec_loss, 
    adv_loss,
    lambda_L1,
    optimizer_g, 
    optimizer_d, 
    train_dl, 
    val_dl, 
    epochs, 
    device
):

    disriminator.to(device)
    generator.to(device)

    loss_meters = create_loss_meters()

    train_df = {
        'epoch': [],
        **{k: [] for k in loss_meters.keys()}
    }

    val_df = {
        'epoch': [],
        **{k: [] for k in loss_meters.keys()}
    }
    
    for epoch in range(epochs):
            
        for phase in ['train', 'val']:

            loader = train_dl if phase == 'train' else val_dl
            is_training = phase == 'train'
            df = train_df if phase == 'train' else val_df

            for data in tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                
                L = data['L'].to(device)
                ab = data['ab'].to(device)
                images = torch.cat([L, ab], dim=1)

                fake_loss = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
                real_loss = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
                advertarial_loss = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)

                if epoch >= DISC_START_EPOCH:

                    if is_training:
                        disriminator.zero_grad()

                    with torch.set_grad_enabled(is_training):
                        output = disriminator(images)

                    label = torch.ones_like(output, device=device)

                    real_loss = adv_loss(output, label)

                    if is_training:
                        real_loss.backward()

                    with torch.set_grad_enabled(is_training):
                        fake_color = generator(L)

                    fake_image = torch.cat([L, fake_color], dim=1)

                    with torch.set_grad_enabled(is_training):
                        output = disriminator(fake_image.detach())

                    label = torch.zeros_like(output, device=device)

                    fake_loss = adv_loss(output, label)

                    if is_training:
                        fake_loss.backward()
                        optimizer_d.step()
                
                if is_training:
                    generator.zero_grad()

                if epoch >= DISC_START_EPOCH:
                    output = disriminator(fake_image)
                    label = torch.ones_like(output, device=device)
                    advertarial_loss = adv_loss(output, label)

                l1_loss = rec_loss(fake_color, ab)
                loss =  advertarial_loss + l1_loss * lambda_L1

                loss_meters['loss_D_fake'].update(fake_loss.item())
                loss_meters['loss_D_real'].update(real_loss.item())
                loss_meters['loss_D'].update(0.5 * fake_loss.item() + 0.5 * real_loss.item())
                loss_meters['loss_G_GAN'].update(advertarial_loss.item())
                loss_meters['loss_G_L1'].update(l1_loss.item())
                loss_meters['loss_G'].update(loss.item())

                if is_training:
                    loss.backward()
                    optimizer_g.step()

            msg = f'{phase} epoch: {epoch + 1}/{epochs} | '

            for k, v in loss_meters.items():
                msg += f'{k}: {v.avg} | '
                df[k].append(v.avg)
                v.reset()

            df['epoch'].append(epoch)

            print(msg)

    train_df = pd.DataFrame(train_df)
    val_df = pd.DataFrame(val_df)
    
    train_df.to_csv(os.path.join(HISTORIES_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(HISTORIES_DIR, 'val.csv'), index=False)

    checkponint = {
        'discriminator': disriminator.state_dict(),
        'generator': generator.state_dict(),
    }

    checkpoint_path = os.path.join(MODELS_DIR, 'checkpoint.pth')
    torch.save(checkponint, checkpoint_path)

def main(args):
    
    train_paths = paths[:16000] 
    val_paths = paths[16000:]

    train_dl = make_dataloaders(batch_size=args.batch_size,paths=train_paths, split='train')
    val_dl = make_dataloaders(batch_size=args.batch_size,paths=val_paths, split='val')

    if args.resnet == "true":
        generator = init_model(build_res_unet(n_input=1, n_output=2, size=args.size), DEVICE)
    else:
        generator = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), DEVICE)

    disriminator = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), DEVICE)

    rec_loss = nn.L1Loss()
    adv_loss = nn.BCEWithLogitsLoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_d = optim.Adam(disriminator.parameters(), lr=args.lr)

    train(
        disriminator, 
        generator, 
        rec_loss, 
        adv_loss,
        args.lambd,
        optimizer_g, 
        optimizer_d, 
        train_dl, 
        val_dl, 
        args.epochs, 
        DEVICE
    )

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--lambd', type=float, default=100.)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--resnet', type=str, default="false", choices=["true", "false"])
    parser.add_argument('--size', type=int, default=SIZE)
    parser.add_argument('--pretrain-generator', type=str, choices=["true", "false"], default="false")

    args = parser.parse_args()

    main(args)