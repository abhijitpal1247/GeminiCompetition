import os
import subprocess

import wget

os.makedirs("cola_weights", exist_ok=True)
os.makedirs("celebrity_weights", exist_ok=True)

subprocess.run(["wget", "-q", "https://civitai.com/api/download/models/267807?token=$CIVITAI_API_KEY", "-O",
                "cola_weights/pytorch_lora.safetensors"])
subprocess.run(["wget", "-q", "https://civitai.com/api/download/models/26680?token=$CIVITAI_API_KEY", "-O",
                "celebrity_weights/pytorch_lora.safetensors"])

