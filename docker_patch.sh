#!/bin/bash

pip uninstall camtools -y
pip install git+https://gitee.com/yxlao/camtools.git -U

pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard loguru plyfile
pip install setuptools==59.5.0 imageio==2.11.1 yapf==0.30.0
