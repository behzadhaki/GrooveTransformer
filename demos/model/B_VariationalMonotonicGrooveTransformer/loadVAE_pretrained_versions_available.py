# Available models:
#   "misunderstood_bush_246 --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/vxuuth1y
#   "rosy_durian_248        --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/2cgu9h6u
#   "hopeful_gorge_252"     --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/v7p0se6e
#   "solar_shadow_247"      --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/35c9fysk

from helpers import load_variational_mgt_model
import torch

model_name = "misunderstood_bush_246"
model_path = f"model/saved/{vae_direction}/{model_name}.pth"

# 1. LOAD MODEL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GrooveTransformer = load_variational_mgt_model(model_path, device=device)
