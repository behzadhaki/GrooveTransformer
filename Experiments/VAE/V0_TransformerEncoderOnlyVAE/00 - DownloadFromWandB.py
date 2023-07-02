import wandb
import os
os.environ["WANDB_SILENT"] = "true"

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "../../..")

from helpers import load_variational_mgt_model

import json

json_dict = dict()

run = wandb.init()
links = {
    "GOOD_AVERAGE_glamorous-sweep-62_epoch_100": "mmil_vae_g2d/voice_distribution_and_genre_distribution_imbalance/model_epoch_100:v283",
    "GOOD_azure-sweep-54_epoch_100": "mmil_vae_g2d/voice_distribution_and_genre_distribution_imbalance/model_epoch_100:v279",
    "GOOD_apricot-sweep-17_epoch_100": "mmil_vae_g2d/voice_distribution_and_genre_distribution_imbalance/model_epoch_100:v242",
    "GOOD_hearty-sweep-60_epoch_100": "mmil_vae_g2d/voice_distribution_and_genre_distribution_imbalance/model_epoch_100:v280",
    "GOOD_worldly-sweep-22_epoch_100": "mmil_vae_g2d/voice_distribution_and_genre_distribution_imbalance/model_epoch_100:v245",
    "GOOD_legendary-sweep-5_epoch_100": "mmil_vae_g2d/voice_distribution_and_genre_distribution_imbalance/model_epoch_100:v230",
    "drawn_river_6_epoch_100": "mmil_vae_g2d/beta_annealing_study/model_epoch_100:v2",
    "worldly-firebrand-5_epoch_100": "mmil_vae_g2d/beta_annealing_study/model_epoch_100:v1",
    "noble-field-7_epoch_100": "mmil_vae_g2d/beta_annealing_study/model_epoch_100:v3",
    "young-violet-12_epoch_200": "mmil_vae_g2d/beta_annealing_study/model_epoch_200:v0",
    "kind-gorge-14_epoch_500": "mmil_vae_g2d/beta_annealing_study/model_epoch_500:v1"
}


for run_name, link in links.items():
    artifact = run.use_artifact(links[run_name], type='model')
    artifact_dir = artifact.download()
    if run_name not in json_dict:
        json_dict[run_name] = dict()
    artifact_name = artifact_dir.split("/")[-1]
    model_name = artifact_name.split("model_epoch_")[-1].split(":")[0]
    model = load_variational_mgt_model(os.path.join(artifact_dir, f"{model_name}.pth"))
    json_dict[run_name] = {
        "wandb_link": link,
        "artifact_name": artifact_name,
        "epoch": model_name,
        "load_path": os.path.join(artifact_dir, f"{model_name}.pth"),
    }
    print(json_dict)

with open("downloaded_artifacts.json", "w") as f:
    json.dump(json_dict, f, sort_keys=True, indent=4, separators=(',', ': '))
