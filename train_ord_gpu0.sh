#!/bin/bash

# antman
# apple
# chest
# gamepad
# ping_pong_racket
# porcelain_mug
# tpiece
# wood_bowl

CUDA_VISIBLE_DEVICES=0 python train_ord.py --config ./configs/single_light/ord.txt --datadir ./data/ord/antman/test --expname ord_antman
CUDA_VISIBLE_DEVICES=0 python train_ord.py --config ./configs/single_light/ord.txt --datadir ./data/ord/apple/test --expname ord_apple
CUDA_VISIBLE_DEVICES=0 python train_ord.py --config ./configs/single_light/ord.txt --datadir ./data/ord/chest/test --expname ord_chest
CUDA_VISIBLE_DEVICES=0 python train_ord.py --config ./configs/single_light/ord.txt --datadir ./data/ord/gamepad/test --expname ord_gamepad
