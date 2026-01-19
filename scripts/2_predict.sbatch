#!/bin/bash
#SBATCH -J predict_video3_full
#SBATCH -p scavenger
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -o /scratch/%u/video3/logs/predict_video3_full.%j.out
#SBATCH -e /scratch/%u/video3/logs/predict_video3_full.%j.err

set -euo pipefail
cd /scratch/$USER/video3
source /home/eleuteri/miniconda/etc/profile.d/conda.sh
conda activate yolopose

MODEL=/scratch/$USER/video3/runs/pose_8020_plus90x30/weights/best.pt
VIDEO=/scratch/$USER/video3/raw/20250811_110514_rotCW.mp4
OUT_PROJECT=/scratch/$USER/video3/runs_predict
NAME=video3_full_bestpt_long

echo "Predicting on VIDEO: $VIDEO with model $MODEL"
yolo pose predict model="$MODEL" source="$VIDEO" save=True save_txt=True project="$OUT_PROJECT" name="$NAME" device=0

# copy results to home for download
mkdir -p /home/$USER/video3_predict_video3_full
if [ -d "$OUT_PROJECT/$NAME" ]; then
  cp -r "$OUT_PROJECT/$NAME" /home/$USER/video3_predict_video3_full/
elif [ -d "$OUT_PROJECT/$NAME/predict" ]; then
  cp -r "$OUT_PROJECT/$NAME/predict" /home/$USER/video3_predict_video3_full/
else
  echo "Warning: expected output folder not found: $OUT_PROJECT/$NAME" >&2
fi

# create tar in home
tar -C /home/$USER -czf /home/$USER/video3_predict_video3_full.tgz video3_predict_video3_full
echo "Tar created: /home/$USER/video3_predict_video3_full.tgz"
