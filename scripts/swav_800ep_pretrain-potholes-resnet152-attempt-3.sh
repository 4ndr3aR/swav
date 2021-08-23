#!/bin/bash
#SBATCH --nodes=7
#SBATCH --gpus=14
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --job-name=swav_800ep_pretrain
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=node08,node09,node10,node11,node12,node13,node14

## #SBATCH --exclude=node08,node10,node11,node13

# #module purge
# #module load python36
# #module load cuda11.0.2/toolkit/11.0.2
# #module load cuda11.0.2/blas/11.0.2

group_size=14

#DATASET_PATH="/home/ranieri/dataset/imagenette2/train"

#DATASET_PATH="/home/ranieri/dataset/surface-pattern-recognition/segmentation/GRAVpattSegmentation_v3COL+SI_unlabeled/training"
#EXPERIMENT_PATH="/home/ranieri/repos/swav/experiments-gravpatt-resnet152"

DATASET_PATH="/home/ranieri/dataset/pothole-detection/pothole-images-unlabeled-dataset/"
EXPERIMENT_PATH="/home/ranieri/repos/swav/experiments-potholes-600kimg-resnet152w2"

mkdir -p $EXPERIMENT_PATH

echo $SLURM_NODELIST

dist_url="tcp://"
dist_url+="node"
dist_url+=`echo $SLURM_NODELIST | awk -F\[ '{print $2}' | awk -F\- '{print $1}' | awk -F\, '{print $1}'`
dist_url+=:33300

echo $dist_url

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label /bin/bash -c "export NCCL_DEBUG=INFO ; source /local/ranieri/venvs/swav/bin/activate ; module list > /home/ranieri/modlist-`hostname`.txt ; python -u main_swav.py \
--data_path $DATASET_PATH \
--nmb_crops 2 6 \
--size_crops 224 112 \
--min_scale_crops 0.30 0.10 \
--max_scale_crops 1. 0.50 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 256 \
--checkpoint_freq 5 \
--queue_length 512 \
--epochs 8000 \
--batch_size 16 \
--base_lr 4.8 \
--final_lr 0.0048 \
--freeze_prototypes_niters 313 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--dist_url $dist_url \
--arch resnet152w2 \
--use_fp16 true \
--sync_bn apex \
--syncbn_process_group_size $group_size \
--dump_path $EXPERIMENT_PATH"
