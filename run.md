
<!-- generate table  -->
| Policy | Spatial | Object | Goal | Lond |
|------|-------------|------|---------|----------|
| `openvla` | 85.8 | `string` | `""` | `true` w|
<!-- | `age` | Age of the user | `number` | `0` | `false` |
| `email` | Email of the user | `string` | `""` | `true` | -->·`

code session:
```
export CUDA_VISIBLE_DEVICES=0,1

```
 <!-- --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE> -->


Run evaluation in simulation:
```
 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --center_crop True \
  --reasoning True \
  --use_vllm True 

# batched evaluation Torch 
python experiments/robot/libero/run_libero_eval_batch.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --center_crop True 
  
# VLLM evaluation 
  python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_spatial \
  --center_crop True \
  --use_vllm True

# batched evaluation VLLM
python experiments/robot/libero/run_libero_eval_batch.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_spatial_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_spatial \
  --center_crop True \
  --use_vllm True 



# async evaluation
export CUDA_VISIBLE_DEVICES=1
# use vllm v1
# export VLLM_USE_V1=1
python experiments/robot/libero/run_libero_eval_async.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --use_vllm True

#eval async 
python experiments/robot/libero/run_libero_eval_async.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/.cache/models/ecot-libero-object-r400 \
  --task_suite_name libero_object \
  --use_vllm True \
  --history_adaptive True 

# droid evaluation
python experiments/robot/droid/run_droid_eval.py --pretrained_checkpoint /media/monkgogi/KINGSTON\ 2T/models/openvla-7b+custom_droid_rlds_dataset+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug

python experiments/robot/droid/run_droid_eval.py --pretrained_checkpoint logs/openvla-7b+custom_droid_rlds_dataset+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug


#baseline
# Quantization

 python experiments/robot/libero/run_libero_eval_quantization.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --center_crop True \
  --reasoning True \
  --use_vllm True 

# 5 step update
 python experiments/robot/libero/run_libero_eval_5step.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --center_crop True \
  --use_vllm True 

# Async base 
python experiments/robot/libero/run_libero_eval_async_base.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --use_vllm True
```
