# Diffusion-Noise-Feature-Accurate-and-Fast-Generated-Image-Detection

[Yichi Zhang](https://yichics.github.io/) and [Xiaogang Xu](https://xiaogang00.github.io/)

Code repository for the paper: [Diffusion Noise Feature: Accurate and Fast Generated Image Detection](https://arxiv.org/abs/2312.02625v2). 

![fig](fig/fig1.png)

### Baseline

The code is based on [CNNDetction](https://github.com/PeterWang512/CNNDetection)


**Model Preparation**

The checkpoints can be download from [Huggingface](https://huggingface.co/toru0035/DNF).

Download the LSUN Bedroom pretrained DDIM from [here](https://heibox.uni-heidelberg.de/f/f179d4f21ebc4d43bbfe/?dl=1). Place it in `./weights/diffusion/` .

**Dataset Preparation**

The testsets can be download from [Huggingface](https://huggingface.co/datasets/toru0035/DNFTestSet).
Download the **DiffusionForensics** from  [DIRE](https://github.com/ZhendongWang6/DIRE). 

Please refer to [CNNDetction](https://github.com/PeterWang512/CNNDetection) for the storage path of the dataset.

**Transform Image to DNF**

```
python compute_dnf.py
```

**Training**

```
python train.py 
```
**Evaluation**

```
python eval.py 
```


Please refer to `./options` for variables that determine the program’s execution.


### Citation 

```
@article{zhang2023diffusion,
  title={Diffusion noise feature: Accurate and fast generated image detection},
  author={Zhang, Yichi and Xu, Xiaogang},
  journal={arXiv preprint arXiv:2312.02625},
  year={2023}
}
```
