
import os
import yaml
import argparse

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from random import choice

from PIL import Image
from torch.utils.data import Dataset

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default="config.yaml", 
        help="Name of the config, under ./dnf/config"
    )
    parser.add_argument(
        "--dataset", type=str, default='./dataset', 
        help='The path to dataset')


    args = parser.parse_args()

    with open(os.path.join("./dnf/configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[Device]: {device}")
    args.device = device

    return args, config

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}

def custom_resize(img, opt):
    # print(opt.rz_interp)
    # exit()
    interp = opt.rz_interp
    # interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])


class ImageDataset(Dataset):
    def __init__(self, root, opt):

        self.root = root
        self.save_root = root + '_dnf'
        os.makedirs(self.save_root, exist_ok=True)
        print(f"[DNF Dataset]: {self.save_root}")
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: custom_resize(img, opt)),
            transforms.ToTensor(),
        ])

        self.paths = []  
        self.save_paths = []
        for foldername, subfolders, filenames in os.walk(self.root): 
            if not os.path.exists(foldername.replace(root, self.save_root)):
                os.mkdir(foldername.replace(root, self.save_root))
                # print(foldername.replace(root, self.save_root))
            for filename in filenames: 
                path = os.path.join(foldername, filename)
                save_path = path.replace(root, self.save_root) 
                self.paths.append(path) 
                self.save_paths.append(save_path)  
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        path = self.paths[index]
        save_path = self.save_paths[index]

        x = Image.open(path)
        x = self.transform(x)

        return x, path, save_path
    
def inversion_first(x, seq, model):

    with torch.no_grad():
        n = x.size(0)
        t = (torch.ones(n) * seq[0]).to(x.device)
        et = model(x, t)

    return et

def norm(x):


    return (x - x.min()) / (x.max() - x.min())