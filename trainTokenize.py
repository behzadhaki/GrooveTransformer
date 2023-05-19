import os

import wandb
import torch
from model import TokenizedTransformerEncoder
from helpers import tokenize_train_utils, tokenize_eval_utils
from data.src.dataLoaders import MonotonicGrooveTokenizedDataset, load_gmd_hvo_sequences
from torch.utils.data import DataLoader
from logging import getLogger, DEBUG
import yaml
import argparse
import numpy as np

logger = getLogger("train.py")
logger.setLevel(DEBUG)

# logger.info("MAKE SURE YOU DO THIS")
# logger.warning("this is a warning!")

parser = argparse.ArgumentParser()

# ----------------------- Set True When Testing ----------------
parser.add_argument("--is_testing", help="Use testing dataset (1% of full date) for testing the script", type=bool,
                    default=False)

# ----------------------- WANDB Settings -----------------------
parser.add_argument("--wandb", type=bool, help="log to wandb", default=True)
# wandb parameters
parser.add_argument(
    "--config",
    help="Yaml file for configuration. If available, the rest of the arguments will be ignored", default=None)
parser.add_argument("--wandb_project", type=str, help="WANDB Project Name", default="tokenized dataset")

# ----------------------- Model Parameters -----------------------
# d_model_dec_ratio denotes the ratio of the dec relative to enc size
parser.add_argument("--d_model", type=int, help="Dimension of the encoder model", default=32)
parser.add_argument("--n_head", type=int, help="Number of attention heads", default=2)
parser.add_argument("--dim_feedforward", type=int, help="Feed forward dimensions", default=64)
parser.add_argument("--num_encoder_layers", type=int, help="Number of encoder layers", default=3)
parser.add_argument("--max_len", type=int, help="Maximum length of the encoder", default=400)
parser.add_argument("--n_voices", type=int, help="Number of drum voices", default=9)
parser.add_argument("--token_embedding_ratio", type=float, help="Ratio of token embedding size to HV linear layer", default=0.8)
parser.add_argument("--token_type_loc", type=int, help="Index of token type", default=0)
parser.add_argument("--padding_idx", type=int, help="Padding value in token embedding", default=0)

# ----------------------- Loss Parameters -----------------------
# Removed all
# ----------------------- Training Parameters -----------------------
parser.add_argument("--dropout", type=float, help="Dropout", default=0.4)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=100)
parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
parser.add_argument("--optimizer", type=str, help="optimizer to use - either 'sgd' or 'adam' loss", default="sgd",
                    choices=['sgd', 'adam'])

# ----------------------- Data Parameters -----------------------
parser.add_argument("--dataset_json_dir", type=str,
                    help="Path to the folder hosting the dataset json file",
                    default="data/dataset_json_settings")
parser.add_argument("--dataset_json_fname", type=str,
                    help="filename of the data (USE 4_4_Beats_gmd.jsom for only beat sections,"
                         " and 4_4_BeatsAndFills_gmd.json for beats and fill samples combined",
                    default="BeatsAndFills_gmd_96.json")
parser.add_argument("--evaluate_on_subset", type=str,
                    help="Using test or evaluation subset for evaluating the model", default="test",
                    choices=['test', 'evaluation'] )

# ----------------------- Evaluation Params -----------------------
parser.add_argument("--calculate_hit_scores_on_train", type=bool,
                    help="Evaluates the quality of the hit models on training set",
                    default=True)
parser.add_argument("--calculate_hit_scores_on_test", type=bool,
                    help="Evaluates the quality of the hit models on test/evaluation set",
                    default=True)
parser.add_argument("--piano_roll_samples", type=bool, help="Generate audio samples", default=True)
parser.add_argument("--piano_roll_frequency", type=int, help="Frequency of piano roll generation", default=1)
parser.add_argument("--hit_score_frequency", type=int, help="Frequency of hit score generation", default=1)

# ----------------------- Misc Params -----------------------
parser.add_argument("--save_model", type=bool, help="Save model", default=True)
parser.add_argument("--save_model_dir", type=str, help="Path to save the model", default="misc/tokenize")
parser.add_argument("--save_model_frequency", type=int, help="Save model every n epochs", default=50)


args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")

# Disable wandb logging in testing mode
if args.is_testing:
    os.environ["WANDB_MODE"] = "disabled"

if args.config is not None:
    with open(args.config, "r") as f:
        hparams = yaml.safe_load(f)
else:
    hparams = dict(
        d_model=args.d_model,
        n_head=args.n_head,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.num_encoder_layers,
        max_len=args.max_len,
        n_voices = args.n_voices,
        #n_token_types = args.n_token_types,
        token_embedding_ratio=args.token_embedding_ratio,
        token_type_loc=args.token_type_loc,
        padding_idx = args.padding_idx,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        is_testing=args.is_testing,
        dataset_json_dir=args.dataset_json_dir,
        dataset_json_fname=args.dataset_json_fname,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


# config files without wandb_project specified
if args.wandb_project is not None:
    hparams["wandb_project"] = args.wandb_project

assert "wandb_project" in hparams.keys(), "wandb_project not specified"


if __name__ == "__main__":

    # go to terminal, python3 trainTokenize.py

    # Initialize wandb
    # ----------------------------------------------------------------------------------------------------------
    wandb_run = wandb.init(
        config=hparams,                         # either from config file or CLI specified hyperparameters
        project=hparams["wandb_project"],          # name of the project
        anonymous="allow",
        entity="mmil_julian",
        settings=wandb.Settings(code_dir="train.py")    # for code saving
    )

    # Reset config to wandb.config (in case of sweeping with YAML necessary)
    # ----------------------------------------------------------------------------------------------------------
    config = wandb.config
    run_name = wandb_run.name
    run_id = wandb_run.id

    # Load Training and Testing Datasets and Wrap them in torch.utils.data.Dataloader
    # ----------------------------------------------------------------------------------------------------------
    # only 1% of the dataset is used if we are testing the script (is_testing==True)
    train_subset = load_gmd_hvo_sequences(dataset_setting_json_path="data/dataset_json_settings/BeatsAndFills_gmd_96.json",
                                          subset_tag="train",
                                          force_regenerate=False)

    training_dataset = MonotonicGrooveTokenizedDataset(subset=train_subset)
    vocab = training_dataset.vocab
    config["n_token_types"] = len(vocab)
    hit_value = vocab["hit"]
    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

    test_subset = load_gmd_hvo_sequences(dataset_setting_json_path="data/dataset_json_settings/BeatsAndFills_gmd_96.json",
                                         subset_tag="test",
                                         force_regenerate=False)
    test_dataset = MonotonicGrooveTokenizedDataset(vocab=vocab, subset=test_subset)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the model
    # ------------------------------------------------------------------------------------------------------------
    tokenized_model = TokenizedTransformerEncoder(config)

    tokenized_model = tokenized_model.to(config.device)
    wandb.watch(tokenized_model, log="all", log_freq=1)

    # Instantiate the loss Criterion and Optimizer
    # ------------------------------------------------------------------------------------------------------------

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(tokenized_model.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(tokenized_model.parameters(), lr=config.lr)

    # Iterate over epochs
    # ------------------------------------------------------------------------------------------------------------
    metrics = dict()
    step_ = 0

    for epoch in range(config.epochs):
        print(f"Epoch {epoch} of {config.epochs}, steps so far {step_}")

        # Run the training loop (trains per batch internally)
        # ------------------------------------------------------------------------------------------
        tokenized_model.train()
        logger.info("***************************Training...")

        train_log_metrics, step_ = tokenize_train_utils.train_loop(
            train_dataloader=train_dataloader,
            model=tokenized_model,
            optimizer=optimizer,
            device=config.device,
            starting_step=step_)

        wandb.log(train_log_metrics, commit=False)

        # ---------------------------------------------------------------------------------------------------
        # After each epoch, evaluate the model on the test set
        #     - To ensure not overloading the GPU, we evaluate the model on the test set also in batche
        #           rather than all at once
        # ---------------------------------------------------------------------------------------------------
        tokenized_model.eval()       # DON'T FORGET TO SET THE MODEL TO EVAL MODE (check torch no grad)
        logger.info("***************************Testing...")

        test_log_metrics = tokenize_train_utils.test_loop(
            test_dataloader=test_dataloader,
            model=tokenized_model,
            device=config.device)

        wandb.log(test_log_metrics, commit=False)
        logger.info(f"Epoch {epoch} Finished with total train loss of {train_log_metrics['train/loss_total']} "
                    f"and test loss of {test_log_metrics['test/loss_total']}")

        # Todo: Removing media generation because there are issues with the resolution consistency (4 vs. 96)
        # Generate PianoRolls if needed
        # ---------------------------------------------------------------------------------------------------
        # if args.piano_roll_samples:
        #     if epoch % args.piano_roll_frequency == 0:
        #         media = tokenize_eval_utils.get_logging_media_for_tokenize_model_wandb(
        #             model=tokenized_model,
        #             device=config.device,
        #             dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
        #             subset_name='test',
        #             down_sampled_ratio=0.005,
        #             vocab=vocab,
        #             cached_folder="eval/GrooveEvaluator/templates",
        #             divide_by_genre=True,
        #             max_length=args.max_len,
        #             need_piano_roll=True,
        #             need_kl_plot=False,
        #             need_audio=False
        #         )
        #         wandb.log(media, commit=False)

        # Commit the metrics to wandb
        # ---------------------------------------------------------------------------------------------------
        wandb.log({"epoch": epoch}, step=epoch)

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
                tokenized_model.save(model_path)
                model_artifact.add_file(model_path)
                wandb_run.log_artifact(model_artifact)
                logger.info(f"Model saved to {model_path}")

    wandb.finish()

