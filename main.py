import argparse
import yaml
import sys
import os
import torch
import numpy as np
import warnings

from runner.run import akl

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default="demo.yml", help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--train", action="store_true", help="train function")
    parser.add_argument("--val", action="store_true", help="val the model on dataset")
    parser.add_argument("--test", action="store_true", help="test function")
    parser.add_argument("--i2n", action="store_true", help="transform the image to DNF")
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    warnings.filterwarnings("ignore", message=".*")

    runner = akl(config)
    if args.train:
        runner.train()
    elif args.val:
        runner.val()
    elif args.test:
        runner.test()
    elif args.i2n:
        runner.img2noise()

    return 0


if __name__ == "__main__":
    sys.exit(main())
