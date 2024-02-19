"""
obtain the DNF copy of all images under PATH
"""

import numpy as np

import torch

from tqdm import tqdm


from torchvision.utils import save_image

from dnf.diffusion import Model
from dnf.utils import ImageDataset
from dnf.utils import parse_args_and_config, inversion_first, norm


if __name__ == '__main__':

    args, config = parse_args_and_config()

    print("Converting the Image to DNF")
    print(f'[Dataset]: {args.dataset}')

    seq = list(map(int, np.linspace(
            0, 
            config.diffusion.num_diffusion_timesteps, 
            config.diffusion.steps + 1
        )))

    diffusion = Model(config)
    diffusion.load_state_dict(torch.load(config.ckpt))
    diffusion = diffusion.to(args.device)
    diffusion.eval()

    batch_size = 2

    dataset = ImageDataset(args.dataset, config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    for x, path, save_path in tqdm(dataloader):

        dnf = inversion_first(x, seq, diffusion)
        # dnf_gray = torch.sum(dnf, dim=1)
        # print(dnf_gray.shape)
        for i in range(batch_size):
            # save_image(norm(dnf_gray[i]), save_path[i])
            # save_image(norm(dnf[i]), save_path[i])
            save_image(dnf[i], save_path[i])
        #     print(save_path[i])
        # exit()
