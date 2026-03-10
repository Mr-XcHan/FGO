#!/bin/bash
#SBATCH --job-name=Qwen_CEWE
#SBATCH --partition=a100
#SBATCH --nodes=1                       
#SBATCH --gres=gpu:2                   
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=FGO/logs/Qwen_CEWE_%j_%t.out

# ✅ 1. 激活你的 Conda 环境
source ~/.bashrc
conda activate FGO

# ✅ 2. 分布式训练相关环境变量（自动设置）
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))
export RANK=$SLURM_NODEID
export NUM_GPUS_PER_NODE=2
export WORLD_SIZE=$SLURM_NNODES
export GPUS=$((WORLD_SIZE * NUM_GPUS_PER_NODE))

echo "RANK: $SLURM_NODEID"
echo "GPUS = $GPUS"
echo "MASTER_ADDR=$MASTER_ADDR"

nvidia-smi


# ✅ 3. 设置 python 包路径（如有必要）
export PYTHONPATH=FGO/src:$PYTHONPATH

# ✅ 4. 启动训练
accelerate launch \
  --config_file FGO/recipes/accelerate_configs/zero2.yaml \
  --num_machines $WORLD_SIZE \
  --machine_rank $RANK \
  --num_processes $GPUS \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  FGO/src/open_r1/grpo.py \
  --config FGO/recipes/Qwen2.5-Math-1.5B/grpo/config_demo.yaml \
  --output_dir FGO/output_logs/CEWE/GRPO/Qwen2.5-Math-1.5B \

