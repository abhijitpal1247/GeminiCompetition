import os
import wget


os.makedirs("cola_weights")
wget.download(url="https://civitai.com/api/download/models/267807?token=$CIVITAI_API_KEY",
              out="cola_weights/pytorch_lora.safetensors")

wget.download(url="https://civitai.com/api/download/models/26680?token=$CIVITAI_API_KEY",
              out="celebrity_weights/pytorch_lora.safetensors")
