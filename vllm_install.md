install vllm
```
conda create -n vllm python=3.11 -y
conda activate vllm

git clone https://github.com/kevinDuan1/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e .


pip install -e ../openvla/
pip install timm==0.9.10  --no-build-isolation

pip install -e ../LIBERO/
pip install -r ../openvla/experiments/robot/libero/libero_requirements.txt

```


