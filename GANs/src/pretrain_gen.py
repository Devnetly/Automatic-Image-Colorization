import os
import glob
import torch
import pandas as pd
from dataclasses import dataclass
from utils import make_dataloaders, build_res_unet
from loss import AverageMeter
from torch import nn, optim
from argparse import ArgumentParser
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv()


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 2
    size: int = 256
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir: str = os.path.join(os.getenv('DATASET_PATH'), "coco-2017-dataset")
    histories_dir: str = os.getenv('HISTORIES_PATH')
    models_dir: str = os.getenv('MODELS_PATH')


def train(net_G, loss_fn, optimizer, train_dl, epochs, device):
    net_G.to(device)
    loss_fn.to(device)
    loss_meter = AverageMeter()

    train_df = {
        'epoch': [],
        'loss': []
    }
    for epoch in range(epochs):
            loader = train_dl 
            df = train_df 

            for data in tqdm(loader, desc=f'epoch {epoch}'):
                L, ab = data['L'].to(device), data['ab'].to(device)
                loss = torch.tensor(0.0, dtype=torch.float32, device=device)

                net_G.zero_grad()
                output = net_G(L)
                loss = loss_fn(output, ab)
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item(), L.size(0))

            df['epoch'].append(epoch)
            df['loss'].append(loss_meter.avg)

            print(f'epoch: {epoch} loss: {loss_meter.avg}')
        
    train_df = pd.DataFrame(train_df)
    
    folder = os.path.join(TrainingConfig.histories_dir, 'Res18-Unet')
    train_df.to_csv(os.path.join(folder, 'train.csv'), index=False)
    checkpoint_path = os.path.join(TrainingConfig.models_dir, "res18-unet.pth")
    torch.save(net_G.state_dict(), checkpoint_path)

def main(args : TrainingConfig):
    paths = glob.glob(os.path.join(args.data_dir, "coco2017/train2017/*.jpg"))

    train_dl = make_dataloaders(batch_size=args.batch_size, paths=paths, split='train')

    net_G = build_res_unet(n_input=1, n_output=2, size=args.size)
    optimizer = optim.Adam(net_G.parameters(), lr=args.learning_rate)
    loss_fn = nn.L1Loss()

    train(net_G, loss_fn, optimizer, train_dl, args.epochs, TrainingConfig.device)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=TrainingConfig.batch_size)
    parser.add_argument('--learning_rate', type=float, default=TrainingConfig.learning_rate)
    parser.add_argument('--epochs', type=int, default=TrainingConfig.epochs)
    parser.add_argument('--size', type=int, default=TrainingConfig.size)

    args = parser.parse_args()
    args = TrainingConfig(**vars(args))

    main(args)




