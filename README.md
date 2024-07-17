# Code
## setup
```bash
conda create -n llfm python=3.8
conda activate llfm
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope
pip install "transformers==4.37.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate
# install pytorch according to your device.
```

## run experiment
```bash
./scripts/run.sh
```
