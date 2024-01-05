import torch 

from data.datasets import real_fake


def dataset_prepare(root):

    dataset = real_fake(root)
    
    return dataset