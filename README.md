# TensoIR Experiments for "ERROR: Evaluation of Reconstruction and Rendering for Object Relighting"

## Dataset

Download and extract the dataset as follows:

```bash
# Folder structure
# data/dataset
# ├── bmvs
# │   ├── bear
# │   ├── clock
# │   ├── dog
# │   ├── durian
# │   ├── jade
# │   ├── man
# │   ├── sculpture
# │   └── stone
# ├── dtu
# │   ├── scan37
# │   ├── scan40
# │   ├── scan55
# │   ├── scan63
# │   ├── scan65
# │   ├── scan69
# │   ├── scan83
# │   └── scan97
# ├── ord
# │   ├── antman
# │   ├── apple
# │   ├── chest
# │   ├── gamepad
# │   ├── ping_pong_racket
# │   ├── porcelain_mug
# │   ├── tpiece
# │   └── wood_bowl
# ├── synth4relight
# │   ├── air_baloons
# │   ├── chair
# │   ├── hotdog
# │   └── jugs
# └── synth4relight_subsampled
#     ├── air_baloons
#     ├── chair
#     ├── hotdog
#     └── jugs
```

## Dependencies

In addition to the author's original dependencies, install the following 
dependencies as well.

```bash
pip install setuptools==59.5.0 imageio==2.11.1 yapf==0.30.0 ipdb matplotlib
```

## Train

### object-relighting-dataset (ord)

```bash
export PYTHONPATH=.

# Train
python train_ord.py \
  --config ./configs/single_light/ord.txt \
  --datadir ./data/dataset/ord/antman/test \
  --expname ord_antman

# Novel view synthesis
# Note: change the checkpoint path accordingly.
python train_ord.py \
   --config ./configs/single_light/ord.txt \
   --datadir ./data/dataset/ord/antman/test \
   --expname ord_antman \
   --render_only 1 \
   --render_test 1 \
   --ckpt log/ord_antman-xxx-xxx/checkpoints/ord_antman_xxx.th

# Relighting
# Note: change the checkpoint path accordingly.
python scripts/relight_ord.py \
  --config configs/relighting_test/ord_relight.txt \
  --batch_size 800 \
  --datadir ./data/dataset/ord/antman/test \
  --hdrdir ./data/dataset/ord/antman/test \
  --geo_buffer_path ./relighting/ord_antman \
  --ckpt log/ord_antman-xxx-xxx/checkpoints/ord_antman_xxx.th
```

### ORD (synth4relight_subsampled)

```bash

# Train
python train_ord.py \
  --config ./configs/single_light/ord.txt \
  --datadir ./data/dataset/synth4relight_subsampled/air_baloons \
  --expname synth4relight_subsampled_air_baloons

# Novel view synthesis

```

### ORD (dtu)

```bash
python train_ord.py \
  --config ./configs/single_light/ord.txt \
  --datadir ./data/dataset/dtu/scan37 \
  --expname dtu_scan37
python train_ord.py \
   --config ./configs/single_light/ord.txt \
   --datadir ./data/dataset/dtu/scan37 \
   --expname dtu_scan37 \
   --render_only 1 \
   --render_test 1 \
   --ckpt log/dtu_scan37-20230602-144207/checkpoints/dtu_scan37_10000.th
```

### ORD (bmvs)

```bash
python train_ord.py \
  --config ./configs/single_light/ord.txt \ 
  --datadir ./data/dataset/bmvs/man \
  --expname bmvs_man
python train_ord.py \
   --config ./configs/single_light/ord.txt \
   --datadir ./data/dataset/bmvs/man \
   --expname bmvs_man \
   --render_only 1 \
   --render_test 1 \
   --ckpt log/bmvs_man-20230604-002748/checkpoints/bmvs_man_70000.th
```
