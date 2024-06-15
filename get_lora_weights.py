import os
import wget


os.makedirs("cola_weights", exist_ok=True)
os.makedirs("celebrity_weights", exist_ok=True)
civitai_api_key = os.getenv('CIVITAI_API_KEY')
wget.download(url=f"https://civitai.com/api/download/models/267807?token={civitai_api_key}",
              out="cola_weights/pytorch_lora.safetensors")

wget.download(url=f"https://civitai.com/api/download/models/26680?token={civitai_api_key}",
              out="celebrity_weights/pytorch_lora.safetensors")
