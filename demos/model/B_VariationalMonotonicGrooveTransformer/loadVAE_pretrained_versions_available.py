# Available models:


from helpers import load_variational_mgt_model
from model import GrooveTransformerEncoderVAE
import torch

model_name = f"save_dommie_version"
model_path = f"demos/model/B_VariationalMonotonicGrooveTransformer/save_model/{model_name}.pth"

# 1. LOAD MODEL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
groove_transformer_vae = load_variational_mgt_model(model_path, device=device)

