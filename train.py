import wandb
import torch
from statistics import mean
from model.src.BasicGrooveTransformer_VAE import GrooveTransformerEncoderVAE
from helpers.BasicGrooveTransformer_train_VAE import calculate_loss_VAE
# Load dataset as torch.utils.data.Dataset
from data.src.dataLoaders import MonotonicGrooveDataset
from torch.utils.data import DataLoader
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# logger.info("MAKE SURE YOU DO THIS")
# logger.warning("this is a warning!")

hyperparameter_defaults = dict(
    nhead_enc=2,
    nhead_dec=2,
    d_model_enc=16,
    d_model_dec=16,
    embedding_size_src=27,
    embedding_size_tgt=27,
    dim_feedforward=32,
    dropout=0.3,
    loss_hit_penalty_multiplier=0.5,
    num_encoder_layers=2,
    num_decoder_layers=2,
    max_len=32,
    device="cpu",
    latent_dim=32,
    epochs=10,
    batch_size=16,
    lr=1e-3,
    bce=True,
    dice=True)  # TODO add dice loss to log

# sweep_config['parameters'] = parameters_dict
#wandb_run = wandb.init(config=hyperparameter_defaults, project="sweeps_small", anonymous="allow", entity="mmil_vae_g2d",
#                       settings=wandb.Settings(code_dir="."))

wandb_run = wandb.init(config=hyperparameter_defaults, project="sweeps_small_save", anonymous="allow",
                       settings=wandb.Settings(code_dir="train.py"))

# this config will be set by Sweep Controller
config = wandb.config
run_name = wandb_run.name
run_id = wandb_run.id




if __name__ == "__main__":
    # load dataset as torch.utils.data.Dataset
    training_dataset = MonotonicGrooveDataset(
        dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
        subset_tag="train",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=False)
    test_dataset = MonotonicGrooveDataset(
        dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
        subset_tag="test",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=False)


    ## train function

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # If called by wandb.agent, as below,

    # BCE used for hit loss
    bce_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    # MSE used for velocities and offsets losses
    mse_fn = torch.nn.MSELoss(reduction='none')
    hit_loss_penalty = config.loss_hit_penalty_multiplier

    groove_transformer_cpu = GrooveTransformerEncoderVAE(config.d_model_enc, config.d_model_dec,
                                                     config.embedding_size_src,
                                                     config.embedding_size_tgt, config.nhead_enc,
                                                     config.nhead_dec,
                                                     config.dim_feedforward, config.dropout,
                                                     config.num_encoder_layers,
                                                     config.latent_dim, config.num_decoder_layers,
                                                     config.max_len,
                                                     device,
                                                     config.bce)

    groove_transformer = groove_transformer_cpu.to(device)#groove_transformer_cpu.cuda() if torch.cuda.is_available() else groove_transformer_cpu
    optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=1e-4)
    batch_size = config.batch_size
    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    metrics = dict()
    for epoch in range(config.epochs):

        #train loop
        groove_transformer.train()  # train mode
        for batch_count, (inputs, outputs, indices) in enumerate(train_dataloader):
            if batch_count % 40 == 0:
                print(f"Epoch {epoch} - Batch #{batch_count} - inputs.shape {inputs.shape} - "
                      f"outputs.shape {outputs.shape} - indices.shape {indices.shape} ")
            #logger.warning(f"model device is {groove_transformer.device}")
            # inputs = inputs.clone().detach()#torch.tensor(inputs.float())
            inputs = inputs.to(device)
            #logger.warning(f"inputs device is {inputs.device}")
            # outputs = torch.tensor(outputs.float())
            outputs = outputs.to(device)
            #logger.warning(f"output device is {outputs.device}")

            # run one epoch

            # forward + backward + optimize
            output_net = groove_transformer(inputs)
            # loss = calculate_loss_VAE(outputs, labels)

            loss, losses = calculate_loss_VAE(output_net, outputs, bce_fn, mse_fn,
                                              hit_loss_penalty, config.bce, config.dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = {"train/loss_total": loss.cpu().detach().numpy(),
                       "train/epoch": epoch,
                       "train/loss_h": losses['loss_h'],
                       "train/loss_v": losses['loss_v'],
                       "train/loss_o": losses['loss_o'],
                       "train/loss_KL": losses['loss_KL'],
                       }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        # test loop
        groove_transformer.eval()
        loss_total = np.array([])
        loss_h = np.array([])
        loss_v = np.array([])
        loss_o = np.array([])
        loss_KL = np.array([])

        for batch_count, (inputs, outputs, indices) in enumerate(test_dataloader):

            output_test = inputs.to(device)
            inputs_test = outputs.to(device)

            output_net_test = groove_transformer(inputs_test)
            val_loss, val_losses = calculate_loss_VAE(output_net_test, output_test, bce_fn, mse_fn,
                                                      hit_loss_penalty, config.bce, config.dice)

            loss_total = np.append(loss_total, val_loss.cpu().detach().numpy())
            loss_h = np.append(loss_h, val_losses['loss_h'])
            loss_v = np.append(loss_v, val_losses['loss_v'])
            loss_o = np.append(loss_o, val_losses['loss_o'])
            loss_KL = np.append(loss_KL, val_losses['loss_KL'])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_metrics = {"val/loss_total": loss_total.mean(),
                       "val/loss_h": loss_h.mean(),
                       "val/loss_v":  loss_v.mean(),
                       "val/loss_o":  loss_o.mean(),
                       "val/loss_KL":  loss_KL.mean()
                       }
        wandb.log({**metrics, **val_metrics})

        print(f"Epoch {epoch} Finished with total loss of {loss_total.mean()}")



        if epoch % 5 == 0:
            model_artifact = wandb.Artifact(f'model_epoch{epoch}', type='model')

            model_path = f"misc/sweeps/{run_name}_{run_id}/{epoch}.pth"
            groove_transformer.save(model_path)
            model_artifact.add_file(model_path)
            wandb_run.log_artifact(model_artifact)

    wandb.finish()

