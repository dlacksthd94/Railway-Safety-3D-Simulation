#!/usr/bin/bash

#################### create miniconda env
ENV_NAME="rail"

set -e
source ~/miniconda3/etc/profile.d/conda.sh

##### rw_scraping #####
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    conda deactivate
    conda env remove -n $ENV_NAME -y
fi
conda create -n $ENV_NAME python=3.12 -y

conda activate $ENV_NAME

#################### install packages
pip cache purge
conda clean --all -y

pip install pandas json5 scikit-learn
pip install streamlit
pip install openai pytesseract google-genai # for transcribing report form in json format
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # torch for cuda 12.6
pip install transformers accelerate bitsandbytes # for labeling news & populating the report form
pip install sentencepiece # for running VLMs from Hugging Face
pip install py360convert ultralytics diffusers peft # for image preprocessing
pip install open3d plotly gsplat # for 3d reconstruction
