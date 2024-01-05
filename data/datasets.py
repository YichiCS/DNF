import os

import torch
import utils

from utils import io
from torch.utils.data import Dataset

class real_fake(Dataset):
    def __init__(self, root):
        imgs = []
        for path in os.listdir(root):
            
            path_prefix_4 = path[:4]

            if path_prefix_4 == "real":
                label = 0
            elif path_prefix_4 == "fake":
                label = 1
            else:
                label = 1
            
            childpath = os.path.join(root, path)
            for img_path in os.listdir(childpath):
                imgs.append((os.path.join(childpath, img_path), label))

        self.imgs = imgs
            
    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        # print(img_path)
        data = io.data_read(img_path)

        return data, label
    
    def __len__(self):
        return len(self.imgs)
