# Available models:
#   "misunderstood_bush_246 --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/vxuuth1y
#   "rosy_durian_248        --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/2cgu9h6u
#   "hopeful_gorge_252"     --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/v7p0se6e
#   "solar_shadow_247"      --> https://wandb.ai/mmil_tap2drum/transformer_groove_tap2drum/runs/35c9fysk

from model.Base.modelLoadersSamplers import load_groove_transformer_encoder_model

model_name = "misunderstood_bush_246"
model_path = f"model/saved/monotonic_groove_transformer_v1/latest/{model_name}.pth"
GrooveTransformer = load_groove_transformer_encoder_model(model_path)