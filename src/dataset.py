import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train', size=256):
        self.split = split
        self.size = size
        self.paths = paths

        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((size, size), Image.BICUBIC),
                transforms.RandomHorizontalFlip()  # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((size, size), Image.BICUBIC)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)

        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)