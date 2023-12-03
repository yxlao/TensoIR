# TensoIR Experiments for "ERROR: Evaluation of Reconstruction and Rendering for Object Relighting"

## Dataset

Download and extract the dataset as follows:

```bash
# Folder structure
# data
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
dependencies. See `README_old.md` for the author's original dependencies.

```bash
conda create -n tensoir python=3.8
conda activate tensoir

# Official deps (install specific version of torch base on your system).
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard loguru plyfile

# Additional deps.
pip install setuptools imageio yapf ipdb matplotlib camtools==0.1.4
```

## Train, NVS, and Relighting

We provide example commands for running training, novel view synthesis (NVS)
and relighting.

We provide `gen_commands.py` to automatically generate commands to run novel
view synthesis and relighting. This script will locate the latest checkpoint
automatically.

```bash
# This is important!
export PYTHONPATH=.

# Train
python train_ord.py \
  --config ./configs/single_light/ord.txt \
  --datadir ./data/ord/antman/test \
  --expname ord_antman

# Gen commands to automatically detect the latest checkpoint.
# Commands will be saved to commands.txt
python gen_commands.py

# Novel view synthesis
# Note: change the checkpoint path accordingly.
python train_ord.py \
   --config ./configs/single_light/ord.txt \
   --datadir ./data/ord/antman/test \
   --expname ord_antman \
   --render_only 1 \
   --render_test 1 \
   --ckpt log/ord_antman-xxx-xxx/checkpoints/ord_antman_xxx.th

# Relighting
# Note: change the checkpoint path accordingly.
python scripts/relight_ord.py \
  --config configs/relighting_test/ord_relight.txt \
  --batch_size 800 \
  --datadir ./data/ord/antman/test \
  --hdrdir ./data/ord/antman/test \
  --geo_buffer_path ./relighting/ord_antman \
  --ckpt log/ord_antman-xxx-xxx/checkpoints/ord_antman_xxx.th
```

## Evaluation

We provide lists of files for preparing evaluation.

```bash                  
eval
├── bmvs_nvs.json              # List of files for NVS on BMVS
├── dtu_nvs.json               # List of files for NVS on DTU
├── ord_nvs.json               # List of files for NVS on object-relighting-dataset
├── ord_relight.json           # List of files for relighting on object-relighting-dataset
├── prepare_eval.py            # Copy the files to the evaluation folder
├── synth4relight_nvs.json     # List of files for NVS on synth4relight
└── synth4relight_relight.json # List of files for relighting on synth4relight
``` 

In the file list json files:

- `gt_path`": Path to the ground-truth file
- `pd_dst_path`": Path to the location where the prediction file will be copied
- `pd_src_path`": Path to the prediction file

Run `prepare_eval.py` to copy all `pd_src_path` to `pd_dst_path`.
