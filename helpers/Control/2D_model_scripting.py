from helpers import load_density_2d_model
import wandb
import torch
import os

if __name__ == "__main__":

        # download model from wandb and load it
        run = wandb.init()

        epoch = 290
        version = 134
        run_name = "summer-sweep-13" #"rosy-sweep-51"       # f"apricot-sweep-56_ep{epoch}"

        artifact_path = f"mmil_julian/Control 1D/model_epoch_{epoch}:v{version}"
        epoch = artifact_path.split("model_epoch_")[-1].split(":")[0]

        artifact = run.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        model = load_density_2d_model(os.path.join(artifact_dir, f"{epoch}.pth"))
        
        model.serialize("../../eval/Control/serialized/2D")

        # scripted_model = torch.jit.script(model)
        #
        # test_input = torch.rand((1, 32, 27))
        # test_density = [0.5]
        #
        # hvo, _, _, _ = scripted_model.forward(test_input, test_density)
        # print(hvo)