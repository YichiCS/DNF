import os
import csv
import time
import shutil

import torch
import torch.nn as nn
import torchvision.utils as tvu
import torchmetrics
from torch.utils.data import DataLoader

import numpy as np
import utils

from tqdm import tqdm

from data import dataset_prepare
from models.resnet import resnet50
from models.xception import xception

from utils import inversion
from utils import io
from models.diffusion import Model as Model_diffusion



class akl(object):
    def __init__(self, config, device=None):
        self.config = config
        if device is None:
            device = (
                torch.device("cuda") 
                if torch.cuda.is_available() 
                else torch.device("cpu")
                )
        self.device = device

        self.model_var_type = config.model.var_type

        self.seq = list(map(int, np.linspace(
            0, 
            config.diffusion.num_diffusion_timesteps, 
            config.diffusion.steps + 1
        )))

        beta = np.linspace(
            config.diffusion.beta_start, 
            config.diffusion.beta_end, 
            config.diffusion.num_diffusion_timesteps + 1, 
            dtype=np.float64
        )

        alpha = (1-beta)
        alpha_cumprod = np.cumprod(alpha, axis=0)
        alpha_next_cumprod = np.append(alpha_cumprod[1:], 0.0)
        alpha_prev_cumprod = np.append(1.0, alpha_cumprod[:-1])

        self.parameters = {
            "beta": torch.from_numpy(beta).float().to(self.device),
            "alpha_cumprod": torch.from_numpy(alpha_cumprod).float().to(self.device), 
            "alpha_next_cumprod": torch.from_numpy(alpha_next_cumprod).float().to(self.device), 
            "alpha_prev_cumprod": torch.from_numpy(alpha_prev_cumprod).float().to(self.device), 
        }
        self.model_dict = {
            'Resnet50':resnet50,
            'Xception':xception,
        }

    def train(self):

        config = self.config
        device = self.device
        model_dict = self.model_dict

        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        ckpt_dir = os.path.join(config.train.ckpt_dir, config.train.model_type, config.exp.name, time_str)
        os.makedirs(ckpt_dir)

        csv_file = os.path.join(ckpt_dir, 'record.csv')
        f = open(csv_file, 'w')
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Acc', 'R_Acc', 'F_Acc', 'mAP'])
        
        print("Model perparing...")
        model = model_dict[config.train.model_type]().to(device)

        lr = config.train.lr
        epoch = config.train.epoch
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()

        print("DataSet preparing...")
        train_dataset = dataset_prepare(config.train.dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)

        min_acc = 0


        for epoch in range(0, epoch):
            print(f"Epoch: {epoch + 1}")
            
            map_metric = torchmetrics.AveragePrecision()
            acc_metric = torchmetrics.Accuracy(average='micro', num_classes=2)
            label_acc_metric = torchmetrics.Accuracy(average=None, num_classes=2)

            sum_loss = 0.0

            for batch_idx, (imgs, labels) in enumerate(tqdm(train_dataloader)):
                imgs, labels = imgs.to(device), labels.to(device)
                # print(labels)
                inputs = imgs
                outputs = model(inputs)

                softmax = nn.Softmax(dim=1)
                results = softmax(outputs)

                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                sum_loss = sum_loss + loss.item()
                _, preds = torch.max(results.data, dim=1)
                loss = sum_loss / (batch_idx + 1)

                map_metric.update(preds.to('cpu'), labels.to('cpu'))
                acc_metric.update(preds.to('cpu'), labels.to('cpu'))
                label_acc_metric.update(preds.to('cpu'), labels.to('cpu'))

            map_result = map_metric.compute()
            acc_result = acc_metric.compute()
            label_acc_result = label_acc_metric.compute()

            utils.result(map_result, acc_result, label_acc_result,  writer, epoch)
            
            model_path = os.path.join(ckpt_dir, f"{config.exp.name}_epoch{epoch + 1}.pt")
            torch.save(model.state_dict(), model_path)

            acc_test = self.val(pretrained=model_path, writer=writer)
            if acc_test > min_acc :
                min_acc = acc_test
                model_path = os.path.join(ckpt_dir, f"best_epoch{epoch}.pt")
                torch.save(model.state_dict(), model_path)


        f.close()
        print("The training is finished")

    def val(self, pretrained, writer=None):
        config = self.config
        device = self.device
        model_dict = self.model_dict

        print("DataSet preparing...")
        val_dataset = dataset_prepare(config.train.val)
        val_dataloader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=True)

        
        print("Models preparing...")
        model = model_dict[config.train.model_type]().to(device)
        model.load_state_dict(torch.load(pretrained))

        model.eval()

        map_metric = torchmetrics.AveragePrecision()
        acc_metric = torchmetrics.Accuracy(average='micro', num_classes=2)
        label_acc_metric = torchmetrics.Accuracy(average=None, num_classes=2)
        
        for _, (imgs, labels) in enumerate(tqdm(val_dataloader)):
            imgs, labels = imgs.to(device), labels.to(device)
            inputs = imgs
            outputs = model(inputs)

            softmax = nn.Softmax(dim=1)
            results = softmax(outputs)
            _, preds = torch.max(results.data, dim=1)

            map_metric.update(preds.to('cpu'), labels.to('cpu'))
            acc_metric.update(preds.to('cpu'), labels.to('cpu'))
            label_acc_metric.update(preds.to('cpu'), labels.to('cpu'))
            
        map_result = map_metric.compute()
        acc_result = acc_metric.compute()
        label_acc_result = label_acc_metric.compute()
        
        utils.result(map_result, acc_result, label_acc_result, writer)

        return acc_result.item()

    def test(self):
        print("This is test function.")
        config = self.config
        device = self.device
        model_dict = self.model_dict
        
        print("Models preparing...")
        model_pretrained = config.qt.model
        model = model_dict[config.qt.model_type]().to(device)
        model.load_state_dict(torch.load(model_pretrained))
        model.eval()
        
        root = config.qt.root
        temp = config.qt.temp
        sub_dirs = os.listdir(root)
        sub_dirs.sort()
        all_map_metric = torchmetrics.AveragePrecision()  
        all_acc_metric = torchmetrics.Accuracy(average='micro', num_classes=2)  
        all_label_acc_metric = torchmetrics.Accuracy(average=None, num_classes=2)  
        
        for sub_dir in sub_dirs:  
            shutil.rmtree(temp)  
            os.mkdir(temp)  
            shutil.copytree(os.path.join(root, sub_dir), os.path.join(temp, sub_dir))  
            val_dataset = dataset_prepare(temp)  
            val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)  
        
            map_metric = torchmetrics.AveragePrecision()  
            acc_metric = torchmetrics.Accuracy(average='micro', num_classes=2)  
            label_acc_metric = torchmetrics.Accuracy(average=None, num_classes=2)  
        
            for batch_idx, (imgs, labels) in enumerate(val_dataloader):  
                imgs, labels = imgs.to(device), labels.to(device)  
                inputs = imgs  
                outputs = model(inputs)  
        
                softmax = nn.Softmax(dim=1)  
                results = softmax(outputs)  
                _, preds = torch.max(results.data, dim=1)  
        
                map_metric.update(preds.to('cpu'), labels.to('cpu'))  
                acc_metric.update(preds.to('cpu'), labels.to('cpu'))  
                label_acc_metric.update(preds.to('cpu'), labels.to('cpu'))  
        
                all_map_metric.update(preds.to('cpu'), labels.to('cpu'))  
                all_acc_metric.update(preds.to('cpu'), labels.to('cpu'))  
                all_label_acc_metric.update(preds.to('cpu'), labels.to('cpu'))  
        
            map_result = map_metric.compute()  
            acc_result = acc_metric.compute()  
            label_acc_result = label_acc_metric.compute()  
        
            utils.result(map_result, acc_result, label_acc_result, epoch=sub_dir)  
        
        all_map_result = all_map_metric.compute()  
        all_acc_result = all_acc_metric.compute()  
        all_label_acc_result = all_label_acc_metric.compute()  
        
        utils.result(all_map_result, all_acc_result, all_label_acc_result)  


    def img2noise(self):
        print("This is i2n function.")
        config = self.config
        device = self.device
        diffusion = Model_diffusion(config)
        diffusion.load_state_dict(torch.load(config.i2n.diffusion, map_location=self.device))
        diffusion = diffusion.to(device)
        diffusion = torch.nn.DataParallel(diffusion)
        diffusion.eval()
        root = config.i2n.root
        target = config.i2n.target
        os.makedirs(target, exist_ok=True)
        sub_dirs = os.listdir(root)
        sub_dirs.sort()
        from tqdm import tqdm

        class ImageDataset(torch.utils.data.Dataset):
            def __init__(self, imgs, root, sub_dir):
                self.imgs = imgs
                self.root = root
                self.sub_dir = sub_dir
                self.dir = dir

            def __len__(self):
                return len(self.imgs)

            def __getitem__(self, index):
                img = self.imgs[index]
                x = io.image_read(os.path.join(self.root, self.sub_dir, img))
                return x, img
            
        for sub_dir in tqdm(sub_dirs):
            print(sub_dir)
            imgs = os.listdir(os.path.join(root, sub_dir))
            os.mkdir(os.path.join(target, sub_dir))
            dataset = ImageDataset(imgs, root, sub_dir)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
            for batch in tqdm(dataloader):
                x, img_names = batch
                noise = inversion.one_inversion_steps(x, self.seq, diffusion, self.parameters)
                for i in range(len(noise)):
                    tvu.save_image(noise[i], os.path.join(target, sub_dir, img_names[i]))