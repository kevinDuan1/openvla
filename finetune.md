
<!-- generate table  -->
| Policy | Spatial | Object | Goal | Lond |
|------|-------------|------|---------|----------|
| `openvla` | 85.8 | `string` | `""` | `true` w|
<!-- | `age` | Age of the user | `number` | `0` | `false` |
| `email` | Email of the user | `string` | `""` | `true` | -->·`

code session:
```
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "Embodied-CoT/ecot-openvla-7b-oxe" \
  --data_root_dir dataset/libero/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir logs/ \
  --adapter_tmp_dir tmp/ \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 16 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OpenVLA \
  --wandb_entity zhekaiduan2312 
```
 <!-- --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE> -->


Run evaluation in simulation:
```
 python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --center_crop True \
  --reasoning True

# batched evaluation Torch 
python experiments/robot/libero/run_libero_eval_batch.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --center_crop True 
  
# batched evaluation VLLM
python experiments/robot/libero/run_libero_eval_batch.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --center_crop True \
  --use_vllm True 

# VLLM evaluation 
  python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --center_crop True \
  --use_vllm True

# async evaluation
# use vllm v1
# export VLLM_USE_V1=1
python experiments/robot/libero/run_libero_eval_async.py \
  --model_family openvla \
  --pretrained_checkpoint /home/zhekai/code/embodied-CoT/outputs/ecot-openvla-7b-oxe+libero_object_no_noops+b1+lr-0.0005+lora-r32+dropout-0.0 \
  --task_suite_name libero_object \
  --use_vllm True 


# droid evaluation
python experiments/robot/droid/run_droid_eval.py --pretrained_checkpoint /media/monkgogi/KINGSTON\ 2T/models/openvla-7b+custom_droid_rlds_dataset+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug

python experiments/robot/droid/run_droid_eval.py --pretrained_checkpoint logs/openvla-7b+custom_droid_rlds_dataset+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug
```
