from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description

    def __init__(self, data, mode):
        self.data = data

        steps = [
            tv.transforms.ToPILImage(),
            tv.transforms.Resize(300),
            tv.transforms.CenterCrop(300),
        ]
        if mode == 'train':
            steps += [tv.transforms.RandomHorizontalFlip(), tv.transforms.RandomVerticalFlip()]
        steps += [tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)]

        self.transform = tv.transforms.Compose(steps)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image = imread(row['filename'])
        image = gray2rgb(image)
        image = self.transform(image)
        label = torch.tensor([row['crack'], row['inactive']], dtype=torch.float32)
        return image, label
