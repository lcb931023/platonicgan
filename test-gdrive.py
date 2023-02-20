import torch
from scripts.utils.config import load_config, make_dirs
import os, sys
import cv2
import scripts.renderer.transform as dt
import scripts.utils.io as dh
import numpy as np
import scripts.utils.utils as utils
import glob
import scripts.utils.logger as log
from scripts.trainer import TrainerPlatonic, TrainerPlatonic3D, Trainer3D

checkpoint_path = '/content/drive/MyDrive/AI/platonicgan/stats/checkpoint.pkl'
config_path = 'scripts/configs/config_tree.yaml'
output_path = '/content/drive/MyDrive/AI/platonicgan/output'

param = load_config(config_path)

# training trainer
if param.mode == 'platonic':
    trainer = TrainerPlatonic(param, None, True)
elif param.mode == 'platonic_3D':
    trainer = TrainerPlatonic3D(param, None, True)
elif param.mode == '3D':
    trainer = Trainer3D(param, None, True)
else:
    raise NotImplementedError

checkpoint = torch.load(checkpoint_path)

for idx, model in enumerate(trainer.models):
    if model is not None:
        model.load_state_dict(checkpoint['model_{}'.format(str(idx))])

encoder = trainer.models[0]
generator = trainer.models[1]

n_outputs_max = 400
n_outputs_counter = 0

object_list = []

with torch.no_grad():
    for idx, (image, volume, vector, base_path, object_id, view_index, class_name) in enumerate(
            trainer.data_loader_test):

        print('{}/{}'.format(n_outputs_counter + 1,
                                    int(trainer.dataset_test.__len__() / param.training.batch_size)))

        x_input = image.to(param.device)
        volume = volume.to(param.device)
        vector = vector.to(param.device)

        if not param.task == 'reconstruction':
            z = encoder(x_input)
        else:
            z = utils.generate_z(param, param.training.z_size)

        output_volume = generator(z)
        output_name = '{}-{}'.format(param.task, object_id[0])
        dh.volume_to_raw(output_volume, output_path, output_name)
