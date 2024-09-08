
# Install dependencies
!pip install wget
!apt-get install sox libsndfile1 ffmpeg libsox-fmt-mp3
!pip install text-unidecode
!pip install matplotlib>=3.3.2

## Install NeMo
BRANCH = 'main'
!python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager
import os
import glob
import subprocess
import tarfile
import wget
import copy
from omegaconf import OmegaConf, open_dict
import torch
import torch.nn as nn

char_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_citrinet_256", map_location='cpu') #citrinet zeigt im vergleich zu z.B. quarznet bessere performance,
#insbesondere bei kleinen Datensätzen (https://csit.am/2023/proceedings/AIML/AIML_1). Zwischen citrinet 256 und citrinet 1024 war citrinet 256 die bessere wahl, da die 1024
#Filter bei diesem Datensatz zu viel gewesen wären.
#für originelles tutorial siehe: https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb
cfg = copy.deepcopy(char_model.cfg)

freeze_encoder = True #@param ["False", "True"] {type:"raw"}
freeze_encoder = bool(freeze_encoder)
def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)
if freeze_encoder:
  char_model.encoder.freeze()
  char_model.encoder.apply(enable_bn_se)
  logging.info("Model encoder has been frozen, and batch normalization has been unfrozen")
else:
  char_model.encoder.unfreeze()
  logging.info("Model encoder has been un-frozen")

# Setup train, validation, test configs
with open_dict(cfg):
  # Train dataset  (Concatenate train manifest cleaned and dev manifest cleaned)
  cfg.train_ds.manifest_filepath = "/content/trainingmanifest (2).json"
  cfg.train_ds.labels = ["audio_filepath", "duration", "text"]
  cfg.train_ds.normalize_transcripts = False
  cfg.train_ds.batch_size = 32
  cfg.train_ds.num_workers = 11
  cfg.train_ds.pin_memory = True
  cfg.train_ds.trim_silence = True
  cfg.train_ds.is_tarred = False

  # Validation dataset  (Use test dataset as validation, since we train using train + dev)
  cfg.validation_ds.manifest_filepath = "/content/validationmanifest.json"
  cfg.validation_ds.labels = ["audio_filepath", "duration", "text"]
  cfg.validation_ds.normalize_transcripts = False
  cfg.validation_ds.batch_size = 8
  cfg.validation_ds.num_workers = 11
  cfg.validation_ds.pin_memory = True
  cfg.validation_ds.trim_silence = True
  cfg.train_ds.is_tarred = False
char_model.setup_training_data(cfg.train_ds)
char_model.setup_multiple_validation_data(cfg.validation_ds)
print(char_model.cfg.spec_augment)

with open_dict(char_model.cfg.optim):
  char_model.cfg.optim.lr = 0.001
  char_model.cfg.optim.betas = [0.9, 0.999] #default für adam
  char_model.cfg.optim.weight_decay = 0.001  # gegen overfitting
  char_model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
  char_model.cfg.optim.sched.warmup_ratio = 0.05  # 5 % warmup
  char_model.cfg.optim.sched.min_lr = 1e-4
  char_model.cfg.spec_augment.freq_masks = 2
  char_model.cfg.spec_augment.freq_width = 25
  char_model.cfg.spec_augment.time_masks = 2
  char_model.cfg.spec_augment.time_width = 0.05
  #spectrogram augmentation in kleinem Ausmaß hilft hier, overfitting zu präventieren
char_model.spec_augmentation = char_model.from_config_dict(char_model.cfg.spec_augment)

use_cer = True #@param ["False", "True"] {type:"raw"}
log_prediction = True #@param ["False", "True"] {type:"raw"}

char_model.wer.use_cer = use_cer
char_model.wer.log_prediction = log_prediction

from pytorch_lightning.callbacks import ModelCheckpoint

# Create an instance of the ModelCheckpoint callback
checkpoint_val_callback = ModelCheckpoint(
    monitor= 'val_loss',
    mode='min',
    save_top_k=1,
    verbose=True,
)
checkpoint_train_callback = ModelCheckpoint(
    monitor= 'train_loss',
    mode='min',
    save_top_k=1,
    verbose=True,
)

import torch
import pytorch_lightning as ptl


EPOCHS = 100

trainer = ptl.Trainer(devices=1,
                      accelerator="gpu",
                      max_epochs=EPOCHS,
                      accumulate_grad_batches=1,
                      enable_checkpointing=False,
                      logger=False,
                      log_every_n_steps=5,
                      check_val_every_n_epoch=10,
                      )

# Setup model with the trainer
char_model.set_trainer(trainer)

# Finally, update the model's internal config
char_model.cfg = char_model._cfg

config = exp_manager.ExpManagerConfig(
    exp_dir=f'/content/',
    name=f"finetunedasrcitrinet3",
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
    ),
)

config = OmegaConf.structured(config)

logdir = exp_manager.exp_manager(trainer, config)
trainer.fit(char_model)

char_model.save_to("/content/model.nemo")