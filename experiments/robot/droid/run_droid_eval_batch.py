"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.
Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import draccus
import numpy as np
import tqdm
import wandb
import time

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.droid.droid_utils import (
    get_droid_env, 
    get_droid_image, 
    get_droid_observation,
    save_rollout_video,
    save_rollot_reasoning,
)

from experiments.robot.openvla_utils import get_processor, hf_to_vllm, PromptManager
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    unnorm_key: str = "custom_droid_rlds_dataset"                       # Key for action un-normalization (for OpenVLA only)
    #################################################################################################################
    # Droid environment-specific parameters
    #################################################################################################################
    max_steps: int = 90                            # Maximum number of steps to run
    instruction: str = "place banana on the plate"         # Instruction for the task
    num_steps_wait: int = 4  
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 1                                    # Random Seed (for reproducibility)
    use_vllm: bool = False                           # Use VLLM for action generation
    reasoning: bool = True                            # Use reasoning for action generation
    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # if "image_aug" in cfg.pretrained_checkpoint:
    #     assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        if cfg.use_vllm:
            model = hf_to_vllm(model, processor, cfg)
            model.use_vllm = True
        else: model.use_vllm = False

    # Initialize local logging

    run_id = cfg.unnorm_key
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Get expected image dimensions
    # resize_size = get_image_resize_size(cfg)
    resize_size = (224, 224)
    env = get_droid_env()
    
    # Start evaluation
    replay_images = []
    replay_reasoning = []
    inference_times = []
    t = 0
    total_episodes = 0
    # M: batch preprations
    prompt_manager = PromptManager()

    while t < cfg.max_steps + cfg.num_steps_wait:
        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
        # and we need to wait for them to fall
        if t < cfg.num_steps_wait:
            # obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue
        
        obs = get_droid_observation(env)
        # Get preprocessed image
        img = get_droid_image(obs, resize_size)
        
        replay_images.append(img)
        obs = {"full_image": img}
        # Query model to get action
        
        if t == cfg.num_steps_wait:
            inference_time, action, generated_ids = get_action(
                cfg,
                model,
                obs,
                cfg.instruction,
                processor=processor,
            )

            # M: Update prompt history
            generated_text = processor.batch_decode(generated_ids)[0]
            prompt_manager.update_history(generated_text)
        else:
            prompts = prompt_manager.generate_prompts(cfg.instruction)
            # Query model to get action
            inference_time, action, generated_ids = get_action(
                cfg,
                model,
                obs,
                cfg.instruction,
                processor=processor,
                prompts=prompts, 
                max_new_tokens=60,
            )
            generated_texts = processor.batch_decode(generated_ids)
            for i, generated_text in enumerate(generated_texts[:-1]):
                # print("\033[32m" + f"Generated texts: {generated_text}" + "\033[0m")
                prompt_manager.update_history(generated_text+" ", i) # since text ends with :
            generated_text = generated_texts[-1]

        replay_reasoning.append(generated_text)
        print(f"\nStep: {t}\n", generated_text)
        print(f"Inference time: {inference_time:.4f} seconds")
        inference_times.append(inference_time)
        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        if cfg.model_family == "openvla":
            action[-1] = 1 - action[-1]
        
        print(f"Step {t} Action: {action}")
        # Execute action in environment
        env.step(action)
        t += 1

    # Save a replay video of the episode
    save_rollout_video(
        replay_images, total_episodes, success=True, task_description=cfg.instruction, log_file=log_file
    )
    save_rollot_reasoning(
        replay_reasoning, replay_images, total_episodes, success=True, task_description=cfg.instruction, log_file=log_file
    )

    # profile times 
    if inference_times:
        total_inference_time = sum(inference_times)
        num_steps = len(inference_times)
        average_inference_time = total_inference_time / num_steps
        throughput = num_steps / total_inference_time
        print(f"Average inference time: {average_inference_time:.4f} seconds")
        print(f"Throughput: {throughput:.2f} steps per second")
        log_file.write(f"Average inference time: {average_inference_time:.4f} seconds\n")
        log_file.write(f"Throughput: {throughput:.2f} steps per second\n")
        if cfg.use_wandb:
            wandb.log({
                "average_inference_time": average_inference_time,
                "throughput": throughput,
        })
            
    # Save local log file
    log_file.close()

if __name__ == "__main__":
    eval_libero()
