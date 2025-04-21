

```
# start the server  
conda activate polymetis-local
python ~/code/droid/scripts/server/run_server.py


conda activate openvla
cd /home/monkgogi/code/openvla
python experiments/robot/droid/run_droid_eval.py \
--pretrained_checkpoint /home/monkgogi/models/openvla-7b+custom_droid_rlds_dataset+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug

cd /home/monkgogi/code/openvla
python experiments/robot/droid/run_droid_eval.py \
--pretrained_checkpoint 

#batched 
python experiments/robot/droid/run_droid_eval_batch.py \
--pretrained_checkpoint /home/monkgogi/models/ecot-openvla-7b-oxe+custom_droid_rlds_dataset+b1+lr-0.0005+lora-r32+dropout-0.0 \
--use_vllm True

```