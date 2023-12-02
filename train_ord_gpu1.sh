#!/bin/bash

# antman
# apple
# chest
# gamepad
# ping_pong_racket
# porcelain_mug
# tpiece
# wood_bowl

CUDA_VISIBLE_DEVICES=1 python train_ord.py --config ./configs/single_light/ord.txt --datadir ./data/ord/ping_pong_racket/test --expname ord_ping_pong_racket
CUDA_VISIBLE_DEVICES=1 python train_ord.py --config ./configs/single_light/ord.txt --datadir ./data/ord/porcelain_mug/test --expname ord_porcelain_mug
CUDA_VISIBLE_DEVICES=1 python train_ord.py --config ./configs/single_light/ord.txt --datadir ./data/ord/tpiece/test --expname ord_tpiece
CUDA_VISIBLE_DEVICES=1 python train_ord.py --config ./configs/single_light/ord.txt --datadir ./data/ord/wood_bowl/test --expname ord_wood_bowl
