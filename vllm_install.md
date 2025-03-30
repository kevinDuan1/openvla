install vllm
```
conda create -n openvla-vllm python=3.11 -y
conda activate openvla-vllm

pip install -e ../openvla/
pip install timm==0.9.10  --no-build-isolation

pip install -e ../LIBERO/
pip install -r ../openvla/experiments/robot/libero/libero_requirements.txt

git clone https://github.com/kevinDuan1/vllm.git
cd vllm
git checkout v0.7

export VLLM_USE_PRECOMPILED=1

export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/c4/9d/64e107313a19327b049a2267871cceb9b0415f79ee5c00dc360099f929e8/vllm-0.8.1-cp38-abi3-manylinux1_x86_64.whl
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/15/77/7beca2061aadfdfd2d81411102e6445b459bcfedfc46671d4712de6a00fb/vllm-0.8.0-cp38-abi3-manylinux1_x86_64.whl
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/8d/cf/9b775a1a1f5fe2f6c2d321396ad41b9849de2c76fa46d78e6294ea13be91/vllm-0.7.3-cp38-abi3-manylinux1_x86_64.whl
pip install --editable ../vllm


```


