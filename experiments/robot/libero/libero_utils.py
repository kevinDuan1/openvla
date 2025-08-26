"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import enum
import cv2
from PIL import Image, ImageDraw, ImageFont
import textwrap
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)
import matplotlib.pyplot as plt


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    rollout_dir = f"./rollouts/{DATE}/{processed_task_description}"
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

def save_rollot_reasoning(rollout_reasoning, rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    rollout_dir = f"./rollouts/{DATE}/{processed_task_description}"
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}/"
    os.makedirs(mp4_path, exist_ok=True)
    for i in range(len(rollout_reasoning)):
        draw_reasoning(rollout_images[i], rollout_reasoning[i]).save(mp4_path + f"reasoning_{i}.png")


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

#=======================================================

#Define some utils.
def draw_reasoning(image, generated_text):
    tags = [f" {tag}" for tag in get_cot_tags_list()]
    reasoning = split_reasoning(generated_text, tags)
    text = [tag + reasoning[tag] for tag in [' TASK:',' PLAN:',' SUBTASK REASONING:',' SUBTASK:',
                                            ' MOVE REASONING:',' MOVE:', ' VISIBLE OBJECTS:', ' GRIPPER POSITION:'] if tag in reasoning]
    metadata = get_metadata(reasoning)
    bboxes = {}
    for k, v in metadata["bboxes"].items():
        if k[0] == ",":
            k = k[1:]
        bboxes[k.lstrip().rstrip()] = v

    caption = ""
    for t in text:
        wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False) 
        word_list = wrapper.wrap(text=t) 
        caption_new = ''
        for ii in word_list[:-1]:
            caption_new = caption_new + ii + '\n      '
        caption_new += word_list[-1]

        caption += caption_new.lstrip() + "\n\n"

    base = Image.fromarray(np.ones((480, 640, 3), dtype=np.uint8) * 255)
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default(size=14) # big text
    color = (0,0,0) # RGB
    draw.text((30, 30), caption, color, font=font)
    # rescale image to 480 640 3
    # image = image.copy()
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS).copy()
    img_arr = np.array(resized_image)

    # img_arr = np.array(image)
    draw_gripper(img_arr, metadata["gripper"])
    draw_bboxes(img_arr, bboxes)

    text_arr = np.array(base)
    reasoning_img = Image.fromarray(np.concatenate([img_arr, text_arr], axis=1))

    return reasoning_img

def split_reasoning(text, tags):
    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
                # print(tag, s)
            else:
                new_parts[k] = v

    return new_parts

class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]

def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]


def draw_gripper(img, pos_list, img_size=(640, 480)):
    for i, pos in enumerate(reversed(pos_list)):
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)

# def get_metadata(reasoning):
#     metadata = {"gripper": [[0, 0]], "bboxes": dict()}

#     if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
#         gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
#         gripper_pos = gripper_pos.split("[")[-1]
#         gripper_pos = gripper_pos.split("]")[0]
#         gripper_pos = [int(x) for x in gripper_pos.split(",")]
#         gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
#         metadata["gripper"] = gripper_pos

#     if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
#         for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
#             obj = sample.split("[")[0]
#             if obj == "":
#                 continue
#             coords = [int(n) for n in sample.split("[")[-1].split(",")]
#             metadata["bboxes"][obj] = coords

#     return metadata

def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        # Filter and validate integers
        gripper_pos = [x.strip() for x in gripper_pos.split(",") if x.strip().isdigit()]
        gripper_pos = [int(x) for x in gripper_pos]
        gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
        metadata["gripper"] = gripper_pos

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            obj = sample.split("[")[0]
            if obj == "":
                continue
            try:
                coords = [x.strip() for x in sample.split("[")[-1].split(",") if x.strip().isdigit()]
                coords = [int(n) for n in coords]
                metadata["bboxes"][obj] = coords
            except (ValueError, IndexError):
                # Handle malformed data gracefully
                print(f"Warning: Skipping malformed data for object {obj}: {sample}")
                continue

    return metadata

def resize_pos(pos, img_size):
    return [(x * size) // 256 for x, size in zip(pos, img_size)]

def draw_bboxes(img, bboxes, img_size=(640, 480)):
    for name, bbox in bboxes.items():
        show_name = name
        # show_name = f'{name}; {str(bbox)}'
        if len(bbox) != 4:
            continue
        cv2.rectangle(
            img,
            resize_pos((bbox[0], bbox[1]), img_size),
            resize_pos((bbox[2], bbox[3]), img_size),
            name_to_random_color(name),
            1,
        )
        cv2.putText(
            img,
            show_name,
            resize_pos((bbox[0], bbox[1] + 6), img_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

def save_action_plots(rollout_actions, idx, success, task_description, log_file=None):
    """Saves action plots for temporal stability analysis."""
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_")
        .replace(".", "_")[:50]
    )
    rollout_dir = f"./rollouts/{DATE}/{processed_task_description}"
    os.makedirs(rollout_dir, exist_ok=True)
    
    # Convert actions to numpy array if it's a list
    actions_array = np.array(rollout_actions)
    
    # Create figure with subplots for each action dimension
    num_dims = actions_array.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(12, 3*num_dims))
    if num_dims == 1:
        axes = [axes]
    
    # Action dimension names (assuming 7-D action space)
    dim_names = [
        'X Position', 'Y Position', 'Z Position', 
        'X Rotation', 'Y Rotation', 'Z Rotation', 'Gripper'
    ]
    
    # Plot each action dimension
    for i in range(num_dims):
        ax = axes[i]
        timesteps = np.arange(len(actions_array))
        ax.plot(timesteps, actions_array[:, i], 'b-', linewidth=2, 
                label=f'{dim_names[i]}')
        
        # Calculate and plot action differences
        if len(actions_array) > 1:
            action_diffs = np.diff(actions_array[:, i])
            ax_twin = ax.twinx()
            ax_twin.plot(timesteps[1:], np.abs(action_diffs), 'r--', alpha=0.7, 
                        label='|Action Diff|')
            ax_twin.set_ylabel('|Action Difference|', color='r')
            ax_twin.tick_params(axis='y', labelcolor='r')
            
            # Add statistics
            mean_diff = np.mean(np.abs(action_diffs))
            std_diff = np.std(action_diffs)
            ax_twin.axhline(y=mean_diff, color='orange', linestyle=':', alpha=0.8, 
                           label=f'Mean |Diff|: {mean_diff:.4f}')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'{dim_names[i]} Value')
        ax.set_title(f'{dim_names[i]} Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        if i < num_dims - 1:
            ax_twin.legend(loc='upper right')
    
    # Add overall statistics
    if len(actions_array) > 1:
        all_diffs = np.diff(actions_array, axis=0)
        overall_mean_diff = np.mean(np.abs(all_diffs))
        overall_std_diff = np.std(all_diffs)
        
        fig.suptitle(
            f'Action Temporal Stability Analysis\n'
            f'Episode {idx} - Success: {success}\n'
            f'Overall Mean |Action Diff|: {overall_mean_diff:.4f}, '
            f'Std: {overall_std_diff:.4f}', 
            fontsize=14
        )
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}"
        f"--task={processed_task_description}--actions.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved action plot at path {plot_path}")
    if log_file is not None:
        log_file.write(f"Saved action plot at path {plot_path}\n")
    
    # Return statistics for logging
    if len(actions_array) > 1:
        all_diffs = np.diff(actions_array, axis=0)
        return {
            'mean_action_diff': np.mean(np.abs(all_diffs)),
            'std_action_diff': np.std(all_diffs),
            'max_action_diff': np.max(np.abs(all_diffs)),
            'min_action_diff': np.min(np.abs(all_diffs))
        }
    return {}

def save_l1_action_plots(rollout_actions, idx, success, task_description, log_file=None):
    """Saves L1 distance action plots for temporal stability analysis."""
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_")
        .replace(".", "_")[:50]
    )
    rollout_dir = f"./rollouts/{DATE}/{processed_task_description}"
    os.makedirs(rollout_dir, exist_ok=True)
    
    # Convert actions to numpy array if it's a list
    actions_array = np.array(rollout_actions)
    
    # Create figure for L1 distance analysis
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate L1 distances between consecutive actions
    if len(actions_array) > 1:
        l1_distances = np.sum(np.abs(np.diff(actions_array, axis=0)), axis=1)
        timesteps = np.arange(1, len(actions_array))
        
        # Plot L1 distances
        ax.plot(timesteps, l1_distances, 'b-', linewidth=2, label='L1 Distance')
        
        # Add statistics
        mean_l1 = np.mean(l1_distances)
        std_l1 = np.std(l1_distances)
        max_l1 = np.max(l1_distances)
        min_l1 = np.min(l1_distances)
        
        # Plot mean line
        ax.axhline(y=mean_l1, color='orange', linestyle=':', alpha=0.8, 
                   label=f'Mean L1: {mean_l1:.4f}')
        
        # Plot std bands
        ax.fill_between(timesteps, mean_l1 - std_l1, mean_l1 + std_l1, 
                       alpha=0.2, color='orange', label=f'±1 Std: {std_l1:.4f}')
        
        # Add statistics text
        stats_text = f'Mean: {mean_l1:.4f}\nStd: {std_l1:.4f}\nMax: {max_l1:.4f}\nMin: {min_l1:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('L1 Distance')
        ax.set_title(f'L1 Distance Between Consecutive Actions\nEpisode {idx} - Success: {success}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add overall statistics to title
        fig.suptitle(
            f'L1 Distance Action Stability Analysis\n'
            f'Episode {idx} - Success: {success}\n'
            f'Mean L1 Distance: {mean_l1:.4f}, Std: {std_l1:.4f}', 
            fontsize=14
        )
    else:
        ax.text(0.5, 0.5, 'Not enough actions for L1 analysis', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f'L1 Distance Analysis\nEpisode {idx} - Success: {success}')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}"
        f"--task={processed_task_description}--l1_actions.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved L1 action plot at path {plot_path}")
    if log_file is not None:
        log_file.write(f"Saved L1 action plot at path {plot_path}\n")
    
    # Return statistics for logging
    if len(actions_array) > 1:
        all_diffs = np.sum(np.abs(np.diff(actions_array, axis=0)), axis=1)
        # all_diffs = np.diff(actions_array, axis=0)
        return {
            'mean_action_diff': np.mean(np.abs(all_diffs)),
            'std_action_diff': np.std(all_diffs),
            'max_action_diff': np.max(np.abs(all_diffs)),
            'min_action_diff': np.min(np.abs(all_diffs))
        }
    return {}