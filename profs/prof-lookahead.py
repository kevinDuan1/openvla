import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import time
import numpy as np
import cv2
import textwrap
from PIL import Image, ImageDraw, ImageFont
import enum
import json
import os 

#Define some utils.
#=============================================================================
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

def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        gripper_pos = [int(x) for x in gripper_pos.split(",")]
        gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
        metadata["gripper"] = gripper_pos

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            obj = sample.split("[")[0]
            if obj == "":
                continue
            coords = [int(n) for n in sample.split("[")[-1].split(",")]
            metadata["bboxes"][obj] = coords

    return metadata

def resize_pos(pos, img_size):
    return [(x * size) // 256 for x, size in zip(pos, img_size)]

def draw_bboxes(img, bboxes, img_size=(640, 480)):
    for name, bbox in bboxes.items():
        show_name = name
        # show_name = f'{name}; {str(bbox)}'

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

#=============================================================================

device = "cuda:1"
# Load Processor & VLA
path_to_converted_ckpt = "/home/zhekai/models/ecot-libero-object-r32"
processor = AutoProcessor.from_pretrained(path_to_converted_ckpt, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    path_to_converted_ckpt,
    # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(device)

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
t = CotTag.TASK.value
def get_openvla_prompt(instruction: str, task) -> str:
    return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: {task}"

INSTRUCTION = "place the watermelon on the towel"
prompt = get_openvla_prompt(INSTRUCTION, t)
image = Image.open("test.png")
print(prompt.replace(". ", ".\n"))
# print("Image size:", image.size)

inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
dataset_statistics_path = os.path.join(path_to_converted_ckpt, "dataset_statistics.json")
if os.path.isfile(dataset_statistics_path):
    with open(dataset_statistics_path, "r") as f:
        norm_stats = json.load(f)
    vla.norm_stats = norm_stats

#=============================================================================
# change the language model use LOOKAHEAD decoding
#=============================================================================
from transformers import AutoModelForCausalLM
if hasattr(vla, "language_model"):
    del vla.language_model

os.environ["USE_LADE"]="1"
import lade
lade.augment_all()
lade.config_lade(LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7, DEBUG=0) 
#LEVEL, WINDOW_SIZE and GUESS_SET_SIZE are three important configurations (N,W,G) in lookahead decoding, please refer to our blog!
#You can obtain a better performance by tuning LEVEL/WINDOW_SIZE/GUESS_SET_SIZE on your own device.

vla.language_model = AutoModelForCausalLM.from_pretrained("tmp/_home_zhekai_models_ecot-libero-object-r32-vllm", torch_dtype=torch.bfloat16).to(device)


# sync prompts
prompts = []
for t in CotTag:
    prompt_b = f"{SYSTEM_PROMPT} USER: What action should the robot take to {INSTRUCTION.lower()}? ASSISTANT: {t.value}"
    prompts.append(prompt_b)

# async prompts
async_prompts = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What action should the robot take to place the watermelon on the towel? ASSISTANT: TASK: The task is to place the watermelon on the towel. The first step is to move the robotic arm towards the towel. PLAN: 1. Move to the right and forward. 2. Move down and grip the towel. 3. Move backward and up. 4. Move to the left. VISIBLE OBJECTS: the robot task [100, 1, 153, 105], the towel [160, 99, 220, 164], the towel [160, 99, 221, 165], table [20, 39, 239, 249], the robot task [100, 1, 154, 106] SUBTASK REASONING: The towel is to the right and slightly forward from the current robotic arm position. The robotic arm needs to move forward and up to reach the towel and grip it. SUBTASK: Move forward and up. MOVE REASONING: The robotic arm needs to move forward and up to reach the towel and grip it. MOVE: Move forward up. GRIPPER POSITION: [121, 91, 130, 87, 142, 87, 153, 88, 169, 95] ACTION: 塔瀬ܝĦ越ਿŸ"
# break async_prompts with CotTag keep value before the tag

prompts = []
for t in CotTag:
    # if t == CotTag.VISIBLE_OBJECTS:
    #     break
    prompts.append(async_prompts.split(t.value)[0] + t.value)
    # print(prompts[-1])

# for p in prompts:
#     print(p)
# left padding
# processor.tokenizer.padding_side = 'left'
inputs = processor(prompts[0], image, padding=True).to(device, dtype=torch.bfloat16)


# use cuda timmer 
times = []
for i in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    action, generated_ids = vla.predict_action(**inputs, unnorm_key='libero_object_no_noops', do_sample=False, max_new_tokens=1024, use_cache=True)
    end.record()
    torch.cuda.synchronize()
    print(f'times:  {start.elapsed_time(end)}ms')
    generated_text = processor.batch_decode(generated_ids)
    for i in range(len(generated_text)):
        # print(f"Prompt: {prompts[i]}")
        print(f"Generated: {generated_text[i]}")
        print("")

    times.append(start.elapsed_time(end))

print(f"Average time: {np.mean(times)}ms")