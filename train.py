import wandb
import torch
from statistics import mean
from model import GrooveTransformerEncoderVAE
from helpers.VAE.train_utils import calculate_loss_VAE, hits_accuracy
from data.src.dataLoaders import MonotonicGrooveDataset
from torch.utils.data import DataLoader
import numpy as np
import logging
import yaml
import argparse
from eval.GrooveEvaluator import load_evaluator
from helpers.VAE.eval_utils import get_logging_media_for_vae_model_wandb

logger = logging.getLogger("train.py")
logger.setLevel(logging.DEBUG)

# logger.info("MAKE SURE YOU DO THIS")
# logger.warning("this is a warning!")

parser = argparse.ArgumentParser()

parser.add_argument("--wandb", help="log to wandb", default=True)

# wandb parameters
parser.add_argument(
    "--config",
    help="Yaml file for configuration. If available, the rest of the arguments will be ignored",
    default=None,
)
parser.add_argument("--wandb_project", help="WANDB Project Name", default="sweeps_small")

# model parameters
parser.add_argument("--d_model_enc", help="Dimension of the encoder model", default=16)
parser.add_argument("--d_model_dec", help="Dimension of the decoder model", default=16)
parser.add_argument("--embedding_size_src", help="Dimension of the source embedding", default=27)
parser.add_argument("--embedding_size_tgt", help="Dimension of the target embedding", default=27)
parser.add_argument("--nhead_enc", help="Number of attention heads for the encoder", default=2)
parser.add_argument("--nhead_dec", help="Number of attention heads for the decoder", default=2)
parser.add_argument("--dim_feedforward_enc", help="Dimension of the feedforward layer for the encoder", default=32)
parser.add_argument("--dim_feedforward_dec", help="Dimension of the feedforward layer for the decoder", default=32)
parser.add_argument("--num_encoder_layers", help="Number of encoder layers", default=2)
parser.add_argument("--num_decoder_layers", help="Number of decoder layers", default=2)
parser.add_argument("--dropout", help="Dropout", default=0.4)
parser.add_argument("--latent_dim", help="Dimension of the latent space", default=32)
parser.add_argument("--max_len_enc", help="Maximum length of the encoder", default=32)
parser.add_argument("--max_len_dec", help="Maximum length of the decoder", default=32)          # FIXME: check the model tester, if this is different from the encoder, it will fail
parser.add_argument("--device", help="Device to use", default="cpu")
parser.add_argument("--o_activation", help="Offset activation function - either 'sigmoid' or 'tanh'",
                    default="tanh")
parser.add_argument("--hit_loss_function", help="hit_loss_function - either 'bce' or 'dice' loss", default='bce')   # FIXME: WHAT DOES THIS DO? FI (check)

# HParams for the model, to use if no config file is provided
parser.add_argument("--loss_hit_penalty_multiplier",
                    help="loss values corresponding to correctly predicted silences will be weighted with this factor",
                    default=0.5)
parser.add_argument("--epochs", help="Number of epochs", default=50)
parser.add_argument("--batch_size", help="Batch size", default=16)
parser.add_argument("--lr", help="Learning rate", default=1e-4)

parser.add_argument("--is_testing", help="Use testing dataset (1% of full date) for testing the script", default=False)  # FIXME: Default should be False before merging
parser.add_argument("--optimizer", help="optimizer to use - either 'sgd' or 'adam' loss", default="sgd")
parser.add_argument("--dataset_json_dir",
                    help="Path to the folder hosting the dataset json file",
                    default="data/dataset_json_settings")
parser.add_argument("--dataset_json_fname",
                    help="filename of the data (USE 4_4_Beats_gmd.jsom for only beat sections,"
                         " and 4_4_BeatsAndFills_gmd.json for beats and fill samples combined",
                    default="4_4_Beats_gmd.json")

parser.add_argument("--save_model", help="Save model", default=True)
parser.add_argument("--save_model_dir", help="Path to save the model", default="misc/VAE")
parser.add_argument("--save_model_frequency", help="Save model every n epochs", default=10)

parser.add_argument("--generate_audio_samples", help="Generate audio samples", default=True)
parser.add_argument("--piano_roll_frequency", help="Frequency of piano roll generation", default=10)

# --------------------------------------------------------------------
# Dummy arguments for running the script in pycharm's python console
# --------------------------------------------------------------------
parser.add_argument("--mode", help="IGNORE THIS PARAM", default="client")
parser.add_argument("--port", help="IGNORE THIS PARAM", default="config.yaml")
# --------------------------------------------------------------------

args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")

if args.config is not None:
    with open(args.config, "r") as f:
        hparams = yaml.safe_load(f)
else:
    hparams = dict(
        d_model_enc=args.d_model_enc,
        d_model_dec=args.d_model_dec,
        embedding_size_src=args.embedding_size_src,
        embedding_size_tgt=args.embedding_size_tgt,
        nhead_enc=args.nhead_enc,
        nhead_dec=args.nhead_dec,
        dim_feedforward_enc=args.dim_feedforward_enc,
        dim_feedforward_dec=args.dim_feedforward_dec,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        max_len_enc=args.max_len_enc,
        max_len_dec=args.max_len_dec,
        device=args.device,
        o_activation=args.o_activation,
        hit_loss_function=args.hit_loss_function,
        loss_hit_penalty_multiplier=args.loss_hit_penalty_multiplier,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        is_testing=args.is_testing,
        dataset_json_dir=args.dataset_json_dir,
        dataset_json_fname=args.dataset_json_fname,
    )




# config files without wandb_project specified
if args.wandb_project is not None:
    hparams["wandb_project"] = args.wandb_project

assert "wandb_project" in hparams.keys(), "wandb_project not specified"


if __name__ == "__main__":

    # Initialize wandb
    # ----------------------------------------------------------------------------------------------------------
    wandb_run = wandb.init(
        config=hparams,                         # either from config file or CLI specified hyperparameters
        project=hparams["wandb_project"],          # name of the project
        anonymous="allow",
        entity="mmil_vae_g2d",                          # saves in the mmil_vae_g2d team account
        settings=wandb.Settings(code_dir="train.py")    # for code saving
    )

    # Reset config to wandb.config (in case of sweeping with YAML necessary)
    # ----------------------------------------------------------------------------------------------------------
    config = wandb.config
    run_name = wandb_run.name
    run_id = wandb_run.id

    # Load Training and Testing Datasets and Wrap them in torch.utils.data.Dataloader
    # ----------------------------------------------------------------------------------------------------------
    # only 1% of the dataset is used for testing
    training_dataset = MonotonicGrooveDataset(
        dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
        subset_tag="train",
        max_len=int(args.max_len_enc),
        tapped_voice_idx=2,
        collapse_tapped_sequence=False,
        down_sampled_ratio=0.1 if args.is_testing is True else None)
    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = MonotonicGrooveDataset(
        dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
        subset_tag="test",
        max_len=int(args.max_len_enc),
        tapped_voice_idx=2,
        collapse_tapped_sequence=False,
        down_sampled_ratio=0.1 if args.is_testing is True else None)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the model
    # ------------------------------------------------------------------------------------------------------------
    groove_transformer_cpu = GrooveTransformerEncoderVAE(config)

    groove_transformer = groove_transformer_cpu.to(config.device)

    # Instantiate the loss Criterion and Optimizer
    # ------------------------------------------------------------------------------------------------------------
    bce_fn = torch.nn.BCEWithLogitsLoss(reduction='none')       # used for hit loss
    mse_fn = torch.nn.MSELoss(reduction='none')                 # used for velocities and offsets losses
    optimizer = torch.optim.Adam(groove_transformer.parameters(), lr=1e-4)    # FIXME !!!!!!!!! WHY JUST ADAM?????

    # Iterate over epochs
    # ------------------------------------------------------------------------------------------------------------
    metrics = dict()
    for epoch in range(config.epochs):

        # Ensure Train Mode
        # ------------------------------------------------------------------------------------------
        groove_transformer.train()

        loss_total = np.array([])
        loss_h = np.array([])
        loss_v = np.array([])
        loss_o = np.array([])
        loss_KL = np.array([])

        # Iterate over batches
        # ------------------------------------------------------------------------------------------
        for batch_count, (inputs, outputs, indices) in enumerate(train_dataloader):

            # Move data to GPU if available
            # ---------------------------------------------------------------------------------------
            inputs = inputs.to(config.device)
            outputs = outputs.to(config.device)

            # Forward pass
            # ---------------------------------------------------------------------------------------
            output_net = groove_transformer(inputs)

            # Compute losses
            # ---------------------------------------------------------------------------------------
            loss, losses = calculate_loss_VAE(
                prediction=output_net,
                targets=outputs,  # TODO rename y to targets (chek)
                bce_fn=bce_fn,
                mse_fn=mse_fn,
                hit_loss_penalty=config.loss_hit_penalty_multiplier,
                hit_loss_function=config.hit_loss_function,  # TODO rename to use_dice (check)
                offset_activation=config.offset_activation)  # TODO rename to use_bce (check)

            # Backward pass
            # ---------------------------------------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics
            # ---------------------------------------------------------------------------------------
            loss_total = np.append(loss_total, val_loss.cpu().detach().numpy())
            loss_h = np.append(loss_h, val_losses['loss_h'])
            loss_v = np.append(loss_v, val_losses['loss_v'])
            loss_o = np.append(loss_o, val_losses['loss_o'])
            loss_KL = np.append(loss_KL, val_losses['loss_KL'])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # FIXME!!!! You're only logging the last batch of the epoch!!! (check)
        # Log metrics
        wandb.log({"train/loss_total": loss_total.mean(),
                   "train/loss_h": loss_h.mean(),
                   "train/loss_v": loss_v.mean(),
                   "train/loss_o": loss_o.mean(),
                   "train/loss_KL": loss_KL.mean()}, commit=False)

        # ---------------------------------------------------------------------------------------------------
        # After each epoch, evaluate the model on the test set
        #     - To ensure not overloading the GPU, we evaluate the model on the test set also in batche
        #           rather than all at once
        # ---------------------------------------------------------------------------------------------------
        groove_transformer.eval()

        accuracy_h = np.array([])
        loss_total = np.array([])
        loss_h = np.array([])
        loss_v = np.array([])
        loss_o = np.array([])
        loss_KL = np.array([])

        for batch_count, (inputs, outputs, indices) in enumerate(test_dataloader):

            # FIXME: ALL EVALUATIONS SO FAR ARE INCORRECT! check comments below
            # EVALUATION: Move data to GPU if available
            # ---------------------------------------------------------------------------------------
            inputs_test = inputs.to(config.device)      # FIXME THERE WAS A BUG HERE, YOU WERE USING THE INPUTS AS OUTPUTS
            output_test = outputs.to(config.device)

            # EVALUATION: Forward pass
            # ---------------------------------------------------------------------------------------
            output_net_test = groove_transformer(inputs_test)       # FIXME: WHy isn't there a predict call here?
            val_loss, val_losses = calculate_loss_VAE(
                prediction=output_net_test,
                y=output_test,
                bce_fn=bce_fn,
                mse_fn=mse_fn,
                hit_loss_penalty=config.loss_hit_penalty_multiplier,
                bce=config.use_bce,
                dice=config.use_dice)       # FIXME THE order of the bce and dice was reversed ---> check if affects the test results

            # EVALUATION: Track Per Batch Metrics to be logged
            # ---------------------------------------------------------------------------------------
            accuracy_h = np.append(accuracy_h, hits_accuracy(output_net_test, output_test))
            loss_total = np.append(loss_total, val_loss.cpu().detach().numpy())
            loss_h = np.append(loss_h, val_losses['loss_h'])
            loss_v = np.append(loss_v, val_losses['loss_v'])
            loss_o = np.append(loss_o, val_losses['loss_o'])
            loss_KL = np.append(loss_KL, val_losses['loss_KL'])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Generate PianoRolls and KL/OA PLots if Needed
        # ---------------------------------------------------------------------------------------------------
        if args.generate_audio_samples:
            if epoch % args.piano_roll_frequency == 0:
                media = get_logging_media_for_vae_model_wandb(
                    groove_transformer_vae=groove_transformer,
                    device=config.device,
                    dataset_setting_json_path=config.dataset_setting_json_path,
                    subset_name='test',
                    down_sampled_ratio=0.005,
                    cached_folder="eval/GrooveEvaluator/templates",
                    divide_by_genre=True,
                    need_piano_roll=True,
                    need_kl_plot=False,
                    need_audio=False
                )
                wandb.log(media, commit=False)

        # EVALUATION: Log Mean of Per Batch Metrics
        # ---------------------------------------------------------------------------------------------------
        wandb.log({
            "test/accuracy_h": accuracy_h.mean(),
            "test/loss_total": loss_total.mean(),
            "test/loss_h": loss_h.mean(),
            "test/loss_v":  loss_v.mean(),
            "test/loss_o":  loss_o.mean(),
            "test/loss_KL":  loss_KL.mean()},
            commit=True)

        logger.info(f"Epoch {epoch} Finished with total loss of {loss_total.mean()} and acc of {accuracy_h.mean()}")

        # Save the model if needed
        # ---------------------------------------------------------------------------------------------------
        if args.save_model:
            if epoch % args.save_model_frequency == 0 and epoch > 0:
                if epoch < 10:
                    ep_ = f"00{epoch}"
                elif epoch < 100:
                    ep_ = f"0{epoch}"
                else:
                    ep_ = epoch
                model_artifact = wandb.Artifact(f'model_epoch_{ep_}', type='model')
                model_path = f"{args.save_model_dir}/{args.wandb_project}/{run_name}_{run_id}/{ep_}.pth"
                groove_transformer.save(model_path)
                model_artifact.add_file(model_path)
                wandb_run.log_artifact(model_artifact)
                logger.info(f"Model saved to {model_path}")

    wandb.finish()

