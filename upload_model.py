from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

model_path = '/home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0'
# Load your model and tokenizer···
model = AutoModelForVision2Seq.from_pretrained(model_path,  torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Upload directly to the Hub
token = "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"
model.push_to_hub("ecot-libero-object-r32", token=token)
tokenizer.push_to_hub("ecot-libero-object-r32", token=token)