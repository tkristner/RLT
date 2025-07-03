#!/bin/bash

python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --break-system-packages
python3 -m pip install vllm==0.8.3 tensorboard packaging --break-system-packages
python3 -m pip install flash-attn --no-build-isolation --break-system-packages
python3 -m pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/ --break-system-packages

python3 -m pip install --upgrade -r requirements_08.txt --break-system-packages