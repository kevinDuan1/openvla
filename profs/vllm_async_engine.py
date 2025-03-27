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
import asyncio
from uuid import uuid4


# vla.language_model.save_pretrained("logs/llama-bridge")
from vllm import LLM, SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from vllm.inputs import TokensPrompt


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

device = "cuda:0"
# Load Processor & VLA
path_to_converted_ckpt = "Embodied-CoT/ecot-openvla-7b-oxe"
# path_to_converted_ckpt = "/home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0"
processor = AutoProcessor.from_pretrained(path_to_converted_ckpt, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    path_to_converted_ckpt,
    torch_dtype=torch.bfloat16,
    # low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(device)


vla.input_embds = vla.language_model.get_input_embeddings()

# load language model with VLLM
if hasattr(vla, "language_model"):
    del vla.language_model

sampling_params = SamplingParams(temperature=0, max_tokens=60, stop_token_ids=[29901])
async_engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model="logs/llama-bridge",
            gpu_memory_utilization=0.64,
            preemption_mode="swap",
            swap_space=10,
            disable_log_requests=True,
        )
)

async def engine_inference(
    model,
    engine,
    input_ids = None,
    pixel_values = None,
    sampling_params = None,
):
    # Visual Feature Extraction (shared across batched lanuaged inputs)
    patch_features = model.vision_backbone(pixel_values)
    projected_patch_embeddings = model.projector(patch_features)
    embds = model.input_embds
    input_embeddings = [embds(ids) for ids in input_ids]
    
    # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
    multimodal_embeddings = [torch.cat([inemb[:, :1, :], projected_patch_embeddings, inemb[:, 1:, :]], dim=1).squeeze(0) for inemb in input_embeddings]#[0] 
    prompt = [[32000] * emb.shape[-2] for emb in multimodal_embeddings]
    inputs = [{"prompt_token_ids": p, "multi_modal_data": {"image":m}} for p, m in zip(prompt, multimodal_embeddings)]
    tasks = [asyncio.create_task(run_query(TokensPrompt(**q), engine, sampling_params)) for q in inputs]
    results = []
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
    return results

async def run_query(query, engine, params):
    request_id = uuid4()
    outputs = engine.generate(query, params, request_id)
    async for output in outputs:
        final_output = output
    responses = []
    for output in final_output.outputs:
        responses.append(output.text)
    return responses

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

# print("Image size:", image.size)
dataset_statistics_path = os.path.join(path_to_converted_ckpt, "dataset_statistics.json")
if os.path.isfile(dataset_statistics_path):
    with open(dataset_statistics_path, "r") as f:
        norm_stats = json.load(f)
    vla.norm_stats = norm_stats

async_prompts = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What action should the robot take to place the watermelon on the towel? ASSISTANT: TASK: The task is to place the watermelon on the towel. The first step is to move the robotic arm towards the towel. PLAN: 1. Move to the right and forward. 2. Move down and grip the towel. 3. Move backward and up. 4. Move to the left. VISIBLE OBJECTS: the robot task [100, 1, 153, 105], the towel [160, 99, 220, 164], the towel [160, 99, 221, 165], table [20, 39, 239, 249], the robot task [100, 1, 154, 106] SUBTASK REASONING: The towel is to the right and slightly forward from the current robotic arm position. The robotic arm needs to move forward and up to reach the towel and grip it. SUBTASK: Move forward and up. MOVE REASONING: The robotic arm needs to move forward and up to reach the towel and grip it. MOVE: Move forward up. GRIPPER POSITION: [121, 91, 130, 87, 142, 87, 153, 88, 169, 95] ACTION: 塔瀬ܝĦ越ਿŸ"
# break async_prompts with CotTag keep value before the tag

prompts = []
for t in CotTag:
    # if t == CotTag.PLAN:
    #     break
    prompts.append(async_prompts.split(t.value)[0] + t.value)
    # print(prompts[-1]) 
from transformers.utils import TensorType
prompts_reason = prompts[:-1]
prompts_action = prompts[-1]

inputs_reason = [processor.tokenizer(p, return_tensors=TensorType.PYTORCH)['input_ids'].to(device) for p in prompts_reason]
inputs_action = [processor.tokenizer(prompts_action, return_tensors=TensorType.PYTORCH)['input_ids'].to(device)]
pixel_values = processor.image_processor(image, return_tensors=TensorType.PYTORCH)["pixel_values"].to(device, dtype=torch.bfloat16)


def get_outputs(model, engine, inputs, pixel_values, sampling_params):    
    start = time.perf_counter()
    result = asyncio.run(engine_inference(vla, async_engine, inputs_reason, pixel_values, sampling_params))
    print("Inference time:", time.perf_counter() - start)
    print(result)


# Background task for sending action requests
async def action_request_task(sampling_params):
   for _ in range(10):
        start = time.perf_counter()
        result = await engine_inference(vla, async_engine, inputs_action, pixel_values, sampling_params)
        print(f"Action Inference time: {time.perf_counter() - start}")
        print("Action result:", result)
        # await asyncio.sleep(1)  # Wait 1 second before the next request

# Background task for sending reasoning requests
async def reasoning_request_task(sampling_params):
    for _ in range(3):
        start = time.perf_counter()
        result = await engine_inference(vla, async_engine, inputs_reason, pixel_values, sampling_params)
        print("Reasoning Inference time:", time.perf_counter() - start)
        print("Reasoning result:", result)
        # await asyncio.sleep(0.1)

# Start the background tasks
async def update_reason_action():
    action_task = asyncio.create_task(action_request_task(sampling_params))
    reasoning_task = asyncio.create_task(reasoning_request_task(sampling_params))
    await asyncio.sleep(10)
    action_task.cancel()
    reasoning_task.cancel()

asyncio.run(update_reason_action())