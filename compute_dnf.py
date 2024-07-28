"""
obtain the DNF copy of all images under PATH
"""
import os
import numpy as np

import torch

from tqdm import tqdm


from torchvision.utils import save_image

from dnf.diffusion import Model
from dnf.utils import _DNFDataset
from dnf.utils import parse_args_and_config, inversion_first, norm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':

    args, config = parse_args_and_config()

    seq = list(map(int, np.linspace(
            0, 
            config.diffusion.num_diffusion_timesteps, 
            config.diffusion.steps + 1
        )))

    diffusion = Model(config)
    diffusion.load_state_dict(torch.load(args.diffusion_ckpt))
    diffusion = diffusion.to(args.device)
    diffusion.eval()

    dataset = _DNFDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.num_threads))

    for x, save_path in tqdm(dataloader):
        x = x.to(args.device)
        dnf = inversion_first(x, seq, diffusion)
        for idx, item in enumerate(dnf):
            save_image(item, save_path[idx])
