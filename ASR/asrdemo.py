!pip install wget
!apt-get install sox libsndfile1 ffmpeg libsox-fmt-mp3
!pip install text-unidecode
!pip install matplotlib>=3.3.2
BRANCH = 'main'
!python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]

!pip install jiwer

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager
import os
import glob
import subprocess
import wget
import torch
import torch.nn as nn
import requests
import copy
from omegaconf import OmegaConf, open_dict

download_model = requests.get("https://raw.githubusercontent.com/dunjadakovic/EloquentiaBWKI/d05d20b99014fd3718d6d9a83adf43f34a2119af/ASR/model.nemo")#modell aus github als tempfile schreiben
with open("model.nemo", "wb") as file:
    file.write(download_model.content)
model = nemo_asr.models.ASRModel.restore_from(restore_path = "/content/model.nemo")#path zum gewünschten path ändern, falls in colab genutzt wird, bleibt alles gleich

model.eval()

import json
import pandas as pd
df = pd.read_json('https://raw.githubusercontent.com/dunjadakovic/EloquentiaBWKI/cf29438e732a88f154ba83578bca7ee69b53eff2/ASR/testfiles/testing_files.json', lines=True)
print(df)
#path aus file extrahieren
audio_filepaths = []
fileNum = 1
pathToRoot = "/content/"
for audio_filepath in df['audio_filepath']:
  print(audio_filepath)
  wavFile = requests.get(audio_filepath)
  file_path = "files" + str(fileNum) + ".wav"
  with open(file_path, "wb") as file:
    file.write(wavFile.content)
  new_path = pathToRoot + file_path
  audio_filepaths.append(new_path)
  fileNum += 1
print(audio_filepaths)
i = 0
print(len(audio_filepaths))
while(i < len(audio_filepaths)):
  try:
    #transkript
    print("Transcribing"+ audio_filepaths[i])
    transcriptionModel = model.transcribe(audio_filepaths[i])
    print(transcriptionModel)
  except:
    print("Please recheck filepath!")
    print(audio_filepath[i])
  i+=1



