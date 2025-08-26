import torch
import requests
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json
import os 

if int(os.environ.get("LOAD_LADE", 0)):
    import lade 
    lade.augment_all()
    lade.config_lade(LEVEL=5, WINDOW_SIZE=15, GUESS_SET_SIZE=15, DEBUG=1, )# POOL_FROM_PROMPT=True)

device = "cuda:0"
# Load Processor & VLA
model_name = "Embodied-CoT/ecot-openvla-7b-oxe"
model_name_vllm = "../tmp/_home_zhekai_models_ecot-libero-object-r32-vllm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
language_model = AutoModelForCausalLM.from_pretrained(model_name_vllm, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)

# Create prompt
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
def get_openvla_prompt(instruction: str) -> str:
    return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
INSTRUCTION = "place the watermelon on the towel"
prompt = get_openvla_prompt(INSTRUCTION)
inputs = tokenizer(prompt, return_tensors='pt').to(device)

# Load image from Github
url = 'https://raw.githubusercontent.com/MichalZawalski/embodied-CoT/main/test_obs.png'
page = requests.get(url)
image = Image.open(BytesIO(page.content))

# warmup
input_emb = torch.load('../tmp/multimodal_embeddings.pt')[0].to(device)
language_model.generate(inputs_embeds=input_emb, max_new_tokens=512, use_cache=True)


# use cuda timmer 
times = []
for i in range(3):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    greedy_outputs = language_model.generate(inputs_embeds=input_emb, max_new_tokens=512, use_cache=True)
    end.record()
    torch.cuda.synchronize()
    print(f'times:  {start.elapsed_time(end)}ms')
    
    print(len(greedy_outputs[0]))
    times.append(start.elapsed_time(end))
print(f'Generated text: {tokenizer.decode(greedy_outputs[0])}')
print(f"Average time: {np.mean(times)}ms")