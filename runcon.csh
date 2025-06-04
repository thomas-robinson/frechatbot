#!/bin/csh
mkdir -p models
mkdir -p faiss_index
if (! -f models/ggml-model-q4_0.gguf) then
  wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf -O ggml-model-q4_0.gguf
  if (! -d models) mkdir models
  mv ggml-model-q4_0.gguf models/
endif
if (! -d fre-cli) then
  git clone --recursive https://github.com/NOAA-GFDL/fre-cli.git
endif
module load miniforge
conda deactivate
conda remove -n local-chatbot --all --yes
conda env create -f environment.yml
conda activate local-chatbot

python chat.py
