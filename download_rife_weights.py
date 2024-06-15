import os
import gdown
from zipfile import ZipFile

os.makedirs("ECCV2022-RIFE-main/train_log", exist_ok=True)
gdown.download(id="1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_", output="ECCV2022-RIFE-main/train_log/")
with ZipFile("ECCV2022-RIFE-main/train_log/RIFE_trained_model_v3.6.zip", 'r') as zObject:
    zObject.extractall(path="arXiv2020-RIFE/train_log/")

gdown.download(id="1i3xlKb7ax7Y70khcTcuePi6E7crO_dFc", output="ECCV2022-RIFE-main/")



