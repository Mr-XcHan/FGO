# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import sys

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards_01 import get_reward_funcs
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from open_r1.utils.data_utils import custom_loading_dataset
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
import wandb
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

from datetime import datetime
now = datetime.now()
timestr = now.strftime("%Y_%m_%d_%H_%M_%S")


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    project_name = "cewe_" + model_args.model_name_or_path.replace("/", "_")
    wandb.init(project=project_name, name="GRPO_" + timestr)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # handle dataset
    # Load the dataset
    if 'simplelr_qwen_level3to5' in script_args.dataset_name:
        dataset = custom_loading_dataset(script_args.dataset_name, max_length=training_args.max_prompt_length,
                                         tokenizer=tokenizer)

    else:
        # this is for mathlighteval
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)['train'].select(range(2000))
        dataset = dataset.map(lambda x:x, remove_columns=['level', 'type'])

    # Load the dataset
    # dataset_paths = {
    #     "math_test": "/mnt/beegfs/home/han/offline_rl/code/O1-Pruner/data/dataset/math_test.json",
    #     "math_train_hard": "/mnt/beegfs/home/han/offline_rl/code/O1-Pruner/data/dataset/math_train_hard.json",
    #     "math_train": "/mnt/beegfs/home/han/offline_rl/code/O1-Pruner/data/dataset/math_train.json",
    #     "gsm8k": "/mnt/beegfs/home/han/offline_rl/code/O1-Pruner/data/dataset/gsm8k.json",
    #     "gaokao": "/mnt/beegfs/home/han/offline_rl/code/O1-Pruner/data/dataset/gaokao.json"
    # }
    # dataset = load_dataset("json", data_files=dataset_paths["math_train"])['train'].select(range(training_args.max_steps))
    # dataset = dataset.map(lambda x:x, remove_columns=['ground_truth_solution', 'pre_generated_steps', 'pre_generated_answer', 'pre_generated_verifier_score'])

        # this is for open-r1
        # dataset = load_dataset(script_args.dataset_name) 
        # dataset = dataset.map(lambda x:x, remove_columns=['solution', 'source', 'messages'])

    # print("dataset", dataset)

    # if isinstance(dataset, DatasetDict):
    #     for split in dataset:
    #         if "messages" in dataset[split].column_names:
    #             dataset[split] = dataset[split].remove_columns("messages")
    # elif isinstance(dataset, Dataset):
    #     if "messages" in dataset.column_names:
    #         dataset = dataset.remove_columns("messages")

    # Format into conversation
    def make_conversation(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        # prompt.append({"role": "user", "content": example["question"]})
        prompt.append({"role": "user", "content": example["problem"]})
        prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt_str}

    def make_conversation_math35(example):
        prompt = []
        prompt.append({"role": "user", "content": example["instruction"][0]['content']})
        # prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    if 'simplelr_qwen_level3to5' in script_args.dataset_name:
        dataset = dataset.map(make_conversation_math35)
    else:
        dataset = dataset.map(make_conversation)
        # eval_dataset = eval_dataset.map(make_conversation)

    # for split in dataset:
    #     if "messages" in dataset[split].column_names:
    #         dataset[split] = dataset[split].remove_columns("messages")

    reward_funcs = get_reward_funcs(script_args)

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    # if training_args.eval_strategy == "no":
    #     eval_dataset = None
    # else:
    #     # if training_args.weighted_sample:
    #     #     eval_dataset = dataset[script_args.dataset_train_split]
    #     # else:
    #     #     eval_dataset = dataset[script_args.dataset_test_split]
    #     eval_dataset = dataset[script_args.dataset_test_split]
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    ) # eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,

    ###############
    # Training loop
    ###############
    def get_latest_checkpoint(output_dir):
        # 找到所有名为 checkpoint-* 的子目录
        checkpoints = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if re.match(r"checkpoint-\d+", d)
        ]
        if not checkpoints:
            return None
        # 提取步数并排序
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        return checkpoints[-1]  # 返回最后一个 checkpoint
    
    logger.info("*** Train ***")
    checkpoint = get_latest_checkpoint(training_args.output_dir)
    print("checkpoint:", checkpoint)
    
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.model.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
