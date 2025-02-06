
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
  --pretrained_checkpoint "logs/ecot-openvla-7b-oxe+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"\
  --task_suite_name libero_spatial \
  --center_crop True
```