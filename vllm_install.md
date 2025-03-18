install vllm
```
conda create -n openvla-vllm python=3.11 -y
conda activate openvla-vllm

pip install -e ../openvla/
pip install timm==0.9.10  --no-build-isolation

git clone https://github.com/kevinDuan1/vllm.git
cd vllm
git checkout v0.7
VLLM_USE_PRECOMPILED=1 pip install -e ../vllm/


pip install -e ../LIBERO/
pip install -r ../openvla/experiments/robot/libero/libero_requirements.txt

```


