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
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/8d/cf/9b775a1a1f5fe2f6c2d321396ad41b9849de2c76fa46d78e6294ea13be91/vllm-0.7.3-cp38-abi3-manylinux1_x86_64.whl
pip install --editable ../vllm


```


install droid
```
pip install pyzed

cd droid
pip install -e ./droid/oculus_reader
pip install -e .


# Done like this to avoid dependency issues
pip install dm-robotics-moma==0.5.0 --no-deps
pip install dm-robotics-transformations==0.5.0 --no-deps
pip install dm-robotics-agentflow==0.5.0 --no-deps
pip install dm-robotics-geometry==0.5.0 --no-deps
pip install dm-robotics-manipulation==0.5.0 --no-deps
pip install dm-robotics-controllers==0.5.0 --no-deps

```