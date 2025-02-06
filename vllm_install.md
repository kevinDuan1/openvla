install vllm
```
conda create -n vllm-v1 python=3.12 -y
conda activate vllm-v1

git clone https://github.com/kevinDuan1/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .


pip install -e ../openvla/
pip install timm==0.9.10  --no-build-isolation
```


