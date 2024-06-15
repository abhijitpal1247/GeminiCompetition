import subprocess

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda"
dtype = torch.float16

step = 8  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"  # Choose to your favorite base model.

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.load_lora_weights("celebrity_weights/", adapter_name="celebrity-lora")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-tilt-up", adapter_name="tilt-up")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing",
                                                    beta_schedule="linear")
pipe.set_adapters(["tilt-up", "celebrity-lora"],
                  [0.8, 0.9])
output = pipe(prompt="th3r0ck smiling", guidance_scale=1.0, num_inference_steps=step, num_frames=24)
export_to_video(output.frames[0], "animation.mp4")

subprocess.run(["python", "inference_video.py", "--exp=1", "--video=animation.mp4", "--scale=0.5"])
