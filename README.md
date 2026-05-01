# Long Chain-of-Thought Compression via Fine-Grained Group Policy Optimization

This repository is an implementation of our paper: `Long Chain-of-Thought Compression via Fine-Grained Group Policy Optimization`.

## Poster

[![Poster preview](poster.png)](poster.png)

**1. Repository Layout**
- `src/open_r1/`: core training and evaluation code.
- `recipes/`: model and training configs (including `accelerate` configs).
- `train_grpo/` `train_fgo/` `train_wo_entropy/` `train_wo_length/`: SLURM launch scripts.
- `slurm/`: training/serving/generation SLURM templates and notes.
- `scripts/`: data generation, evaluation, upload helpers.
- `tests/`: tests and configs.

**2. Environment Setup**
Use the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate FGO
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```


**3. Quick Start**
Adjust paths and configs for your setup:
```bash
accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  src/open_r1/grpo.py \
  --config recipes/Qwen2.5-1.5B-Math/grpo/config_demo.yaml \
  --output_dir outputs/grpo_demo
```

**4. SLURM Training**
Templates are provided; update absolute paths to match your environment:
```bash
sbatch train_fgo/train_grpo_Qwen_math.sh
```

**6. Reproducibility Pointers**
- Training entry points: `src/open_r1/grpo.py`, `src/open_r1/grpo_pure.py`, `src/open_r1/grpo_wo_entropy.py`, `src/open_r1/grpo_wo_length.py`
- Configs: `recipes/*/grpo/*.yaml`
- Launch scripts: `train_*/`

**7. Citation**
If you use this project, please cite our paper:
```
@misc{han2026longchainofthoughtcompressionfinegrained,
      title={Long Chain-of-Thought Compression via Fine-Grained Group Policy Optimization}, 
      author={Xinchen Han and Hossam Afifi and Michel Marot and Xilu Wang and Lu Yin},
      year={2026},
      eprint={2602.10048},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.10048}, 
}
```

**8. Notes**
- W&B logging is enabled by default in training scripts; adjust `report_to` or related code if needed.
- Some scripts are cluster-specific and require path/environment adaptation.
