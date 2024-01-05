import torch
import torchvision
import torchvision.utils as tvu
import torchvision.transforms as transforms  
from torchvision.transforms import InterpolationMode
  
# Used for resnet training
def data_read(img_path):  
    
    img = torchvision.io.read_image(img_path).float()  

    if img.shape[0] == 1:  
        img = torch.cat([img] * 3, dim=0)
    elif img.shape[0] == 4:  
        img = img[:3, :, :]

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
 
    img = transform(img)  
    img = img / 255.0  
    # print(img)
    # import sys
    # sys.exit()
    
    
    return img  

# Used for diffusion model convert image into noise
def image_read(img_path):  
    
    img = torchvision.io.read_image(img_path).float()  

    if img.shape[0] == 1:  
        img = torch.cat([img] * 3, dim=0)
    elif img.shape[0] == 4:  
        img = img[:3, :, :]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
    ])
 
    img = transform(img)  
    img = img / 255.0  
    img = img * 2.0 - 1.0
    
    return img  

def tvu_save(img, img_path):

    tvu.save_image(img, img_path)

    return 0