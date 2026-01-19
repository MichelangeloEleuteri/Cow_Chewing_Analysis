#!/bin/bash -l
#SBATCH -J pose_8020_plus90x30
#SBATCH -p scavenger
#SBATCH --qos=standard
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=g011
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -o /scratch/%u/video3/logs/train_8020.%j.out
#SBATCH -e /scratch/%u/video3/logs/train_8020.%j.err

set -xeo pipefail
ulimit -n 4096

cd /scratch/$USER/video3

# use conda base present on login
source /home/eleuteri/miniconda/etc/profile.d/conda.sh || true
eval "$(conda shell.bash hook)" || true
conda activate yolopose

# diagnostics
echo "USER: $USER"
echo "which conda -> $(which conda)"
conda info --base || true
which yolo || true
yolo -v || true
python - <<'PY'
import sys
try:
    import ultralytics, torch
    print("ultralytics", ultralytics.__version__)
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("diag error:", e, file=sys.stderr)
PY
nvidia-smi || true

# prepare folders
RUNS=/scratch/$USER/video3/runs
mkdir -p "$RUNS"
mkdir -p /scratch/$USER/video3/logs

# sanity checks: yaml & model
DATA_YAML=/scratch/$USER/video3/yolo_from_txt/data/data_upright_plus90x30px_8020.yaml
echo "Checking YAML: $DATA_YAML"
[ -f "$DATA_YAML" ] || { echo "ERROR: missing DATA yaml: $DATA_YAML"; exit 2; }

# check model candidate(s)
echo "Available yolo model files in cwd:"
ls -1 /scratch/$USER/video3/*.pt /scratch/$USER/video3/yolo_from_txt/*.pt /home/$USER/*.pt 2>/dev/null || true
echo "If yolo11s-pose.pt missing, change model parameter in this script."

# print small dataset stats
BASE=/scratch/$USER/video3/yolo_from_txt/data
echo "[train images]" $(ls -1 "$BASE/images/train" | egrep -i '\.(jpg|jpeg|png)$' | wc -l)
echo "[val images]"   $(ls -1 "$BASE/images/val"   | egrep -i '\.(jpg|jpeg|png)$' | wc -l)
echo "[train labels]" $(ls -1 "$BASE/labels/train" | wc -l)
echo "[val labels]"   $(ls -1 "$BASE/labels/val"   | wc -l)

# training command (one line)
yolo pose train model=yolo11s-pose.pt data="$DATA_YAML" imgsz=640 batch=-1 epochs=100 device=0 workers=4 project="$RUNS" name=pose_8020_plus90x30 save_period=5
