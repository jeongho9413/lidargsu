# LidarGSU

* This is the official implementation of our paper, [**Gait Sequence Upsampling using Diffusion Models for Single LiDAR Sensors**](https://arxiv.org/abs/2410.08680).
* LidarGSU is a sparse-to-dense upsampling model for 3D LiDAR pedestrian point clouds, designed to improve the generalization capability of existing person identification models.

## Overview

<p align="center">
  <img src="assets/lidargsu_network.png" width="900"/></br>
  <span align="center">Overview of the 2VGait, which learns two viewpoint-invariant gait shapes in varying point cloud densities using an attention-based approach.</span> 
</p>


## Results

<p align="center">
  <img src="assets/lidargsu_result_gif1.gif" width="900"/></br>
  <span align="center">Overview of the 2VGait, which learns two viewpoint-invariant gait shapes in varying point cloud densities using an attention-based approach.</span> 
</p>


<p align="center">
  <img src="assets/lidargsu_result_gif3.gif" width="600"/></br>
  <span align="center">Overview of the 2VGait, which learns two viewpoint-invariant gait shapes in varying point cloud densities using an attention-based approach.</span> 
</p>


## Setup

### Dataset (Optional)

For training and evaluation, download a [**SUSTeck1K**](https://lidargait.github.io/) and put it into `datasets/susteck1k` dir.
**KUGait30**, which was used in our work for real-world senarios, is not publicly available at the moment.
However, you can refer to our implementation and adapt the training and evaluation process for your own data using this codebase.


### Dependencies

Clone this repository:

```bash
git clone https://github.com/jeongho9413/lidargsu.git
cd lidagsu
```



