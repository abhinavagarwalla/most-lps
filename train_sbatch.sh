#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --job-name=NAME
#SBATCH --time=7-12:00:00
#SBATCH --signal=TERM@120
#SBATCH --mail-user=EMAIL
#SBATCH --mail-type=FAIL
#SBATCH --output=output.txt
#SBATCH -w node
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env

PYTHONPATH="${PYTHONPATH}:." python -m torch.distributed.launch --nproc_per_node=8 --master_port 98992 ./tools/train.py configs/nusc/encoder_decoder/nusc_centerpoint_encoderdecoder_0075voxel_fix_bn_z_scale_fade_instance_max.py --work_dir ./WORK_DIR