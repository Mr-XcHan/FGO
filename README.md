# FGO - Long Chain-of-Thought Compression via Fine-Grained Group Policy Optimization

This repository is an implementation of paper: `Long Chain-of-Thought Compression via Fine-Grained Group Policy Optimization`.

The project builds on Hugging Face Transformers / TRL and focuses on GRPO-style post-training with composable reward functions for math reasoning and structured output formats.

**Highlights**
- GRPO training entry points and variants (e.g., `grpo.py`, `grpo_pure.py`, `grpo_wo_entropy.py`, `grpo_wo_length.py`).
- Composable reward functions (accuracy, format, tag count, length/repetition penalties, code rewards) in `src/open_r1/rewards*.py`.
- Dataset loading and chat-template formatting in `src/open_r1/grpo*.py` and `src/open_r1/utils/`.
- Math answer normalization and verification (`src/open_r1/math_normalize.py`, via `math-verify`).
- SLURM scripts and helpers for training, evaluation, and generation.

**Repository Layout**
- `src/open_r1/`: core training and evaluation code.
- `recipes/`: model and training configs (including `accelerate` configs).
- `train_grpo/` `train_fgo/` `train_wo_entropy/` `train_wo_length/`: SLURM launch scripts.
- `slurm/`: training/serving/generation SLURM templates and notes.
- `scripts/`: data generation, evaluation, upload helpers.
- `tests/`: tests and configs.

**Environment Setup**
Use the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate GPG
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

Core dependencies include `transformers`, `trl`, `datasets`, `deepspeed`, `accelerate`. Math evaluation uses `math-verify` and `latex2sympy2-extended`. Custom evaluation tasks live in `src/open_r1/evaluate.py` (requires `lighteval`).

**Quick Start (Single Node Example)**
Adjust paths and configs for your setup:
```bash
accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  src/open_r1/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
  --output_dir outputs/grpo_demo
```

**SLURM Training**
Templates are provided; update absolute paths to match your environment:
```bash
sbatch train_grpo/train_grpo_Qwen_instruct.sh
```

**Run In Background (CLI Suspend)**
You can detach training jobs from your terminal using one of the options below.

Option A: `nohup` (simple, no session management)
```bash
nohup accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  src/open_r1/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
  --output_dir outputs/grpo_demo \
  > logs/grpo_demo.out 2>&1 &
```

Option B: `tmux` (recommended for long runs)
```bash
tmux new -s grpo_run
# run your command inside tmux
accelerate launch ... 
# detach with Ctrl+b then d
```

Option C: `screen`
```bash
screen -S grpo_run
accelerate launch ...
# detach with Ctrl+a then d
```

**Evaluation**
Custom tasks are defined in `src/open_r1/evaluate.py`. Run evaluation with `lighteval` per your experiment setup.

**Reproducibility Pointers**
- Training entry points: `src/open_r1/grpo.py`, `src/open_r1/grpo_pure.py`, `src/open_r1/grpo_wo_entropy.py`, `src/open_r1/grpo_wo_length.py`
- Configs: `recipes/*/grpo/*.yaml`
- Launch scripts: `train_*/`

**Citation**
If you use this project, please cite our paper:
```
xxx
```

**Notes**
- W&B logging is enabled by default in training scripts; adjust `report_to` or related code if needed.
- Some scripts are cluster-specific and require path/environment adaptation.
