import os

import wandb
import json
import torch
from model import GrooveControl_VAE, Control_components, GAN_components

from helpers.Control.density_eval import *
from data.src.dataLoaders import GrooveDataSet_Control
from helpers import vae_train_utils, control_train_utils, control_loss_functions
from helpers import density_eval
from helpers.Control.loss_functions import generate_theta_rise


from torch.utils.data import DataLoader
from logging import getLogger, DEBUG
import yaml
import argparse
from distutils.util import strtobool

logger = getLogger("train.py")
logger.setLevel(DEBUG)

parser = argparse.ArgumentParser()

# ----------------------- Set True When Testing ----------------
parser.add_argument("--is_testing", help="Use testing dataset (1% of full date) for testing the script", type=strtobool,
                    default=False)

# ----------------------- WANDB Settings -----------------------
parser.add_argument("--wandb", type=bool, help="log to wandb", default=True)
parser.add_argument(
    "--config",
    help="Yaml file for configuration. If available, the rest of the arguments will be ignored", default=None)
parser.add_argument("--wandb_project", type=str, help="WANDB Project Name",
                    default="ControlAdversarial")

# ----------------------- Model Parameters -----------------------
# d_model_dec_ratio denotes the ratio of the dec relative to enc size
parser.add_argument("--d_model_enc", type=int, help="Dimension of the encoder model", default=128)
parser.add_argument("--d_model_dec_ratio", type=int, help="Dimension of the decoder model as a ratio of d_model_enc",
                    default=2)
parser.add_argument("--embedding_size_src", type=int, help="Dimension of the source embedding", default=3)
parser.add_argument("--embedding_size_tgt", type=int, help="Dimension of the target embedding", default=27)
parser.add_argument("--nhead_enc", type=int, help="Number of attention heads for the encoder", default=4)
parser.add_argument("--nhead_dec", type=int, help="Number of attention heads for the decoder", default=8)
# d_ff_enc_to_dmodel denotes the ratio of the feed_forward ratio in encoder relative to the encoder dim (d_model_enc)
parser.add_argument("--d_ff_enc_to_dmodel", type=float,
                    help="ration of the dimension of enc feed-frwrd layer relative to "
                         "enc dmodel", default=1)
# d_ff_dec_to_dmodel denotes the ratio of the feed_forward ratio in encoder relative to the encoder dim (d_model_enc)
parser.add_argument("--d_ff_dec_to_dmodel", type=float,
                    help="ration of the dimension of dec feed-frwrd layer relative to decoder dmodel", default=4)
# n_dec_lyrs_ratio denotes the ratio of the dec relative to n_enc_lyrs
parser.add_argument("--n_enc_lyrs", type=int, help="Number of encoder layers", default=2)
parser.add_argument("--n_dec_lyrs_ratio", type=float, help="Number of decoder layers as a ratio of "
                                                           "n_enc_lyrs as a ratio of d_ff_enc", default=5)
parser.add_argument("--max_len_enc", type=int, help="Maximum length of the encoder", default=32)
parser.add_argument("--max_len_dec", type=int, help="Maximum length of the decoder", default=32)
parser.add_argument("--latent_dim", type=int, help="Overall Dimension of the latent space", default=128)

# ----------------------- Dropout Training Parameters -----------------------
parser.add_argument("--dropout", type=float, help="Dropout", default=0.2)
parser.add_argument("--velocity_dropout", type=float, help="Velocity Dropout", default=0.2)
parser.add_argument("--offset_dropout", type=float, help="Offset Dropout", default=0.2)

# ----------------------- Control Parameters -----------------------
parser.add_argument("--use_in_attention", type=strtobool, help="In Attention Decoder", default=False)
parser.add_argument("--n_continuous_params", type=int, help="Number of controllable continuous parameters", default=2)
parser.add_argument("--train_density", type=strtobool, help="Include density parameter", default=True)
parser.add_argument("--train_intensity", type=strtobool, help="Include intensity parameter", default=True)
parser.add_argument("--train_genre", type=strtobool, help="Include genre parameter", default=True)
parser.add_argument("--balance_param_loss_weights", type=strtobool,
                    help="Weight losses according to data", default=False)

# ----------------------- Training Loss Parameters -----------------------
parser.add_argument("--balance_vo", type=strtobool, help="Whether to make vel/off loss proportional to h",
                    default=False)
parser.add_argument("--hit_loss_balancing_beta", type=float, help="beta parameter for hit loss balancing", default=0.0)
parser.add_argument("--genre_loss_balancing_beta", type=float, help="beta parameter for genre loss balancing",
                    default=0.0)
parser.add_argument("--hit_loss_function", type=str, help="hit_loss_function - only bce supported for now",
                    default="bce")
parser.add_argument("--velocity_loss_function", type=str, help="velocity_loss_function - either 'bce' or 'mse' loss",
                    default='mse', choices=['bce', 'mse'])
parser.add_argument("--offset_loss_function", type=str, help="offset_loss_function - either 'bce' or 'mse' loss",
                    default='mse', choices=['bce', 'mse'])
parser.add_argument("--beta_annealing_activated", help="Use cyclical annealing on KL beta term", type=strtobool,
                    default=True)
parser.add_argument("--beta_level", type=float, help="Max level of beta term on KL", default=0.1)
parser.add_argument("--beta_annealing_per_cycle_rising_ratio", type=float,
                    help="rising ratio in each cycle to anneal beta", default=0.75)
parser.add_argument("--beta_annealing_per_cycle_period", type=int,
                    help="Number of epochs for each cycle of Beta annealing", default=100)
parser.add_argument("--beta_annealing_start_first_rise_at_epoch", type=int,
                    help="Warm up epochs (KL = 0) before starting the first cycle", default=30)

# ----------------------- Training Parameters -----------------------
parser.add_argument("--force_data_on_cuda", type=bool, help="places all training data on cude", default=True)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=300)
parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
parser.add_argument("--lr", type=float, help="Learning rate", default=0.0003)
parser.add_argument("--optimizer", type=str, help="optimizer to use - either 'sgd' or 'adam' loss", default="adam",
                    choices=['sgd', 'adam'])
parser.add_argument("--reduce_loss_by_sum", type=int, help="reduce loss by summing over all dimensions", default=0)

# ----------------------- Data Parameters -----------------------
parser.add_argument("--dataset_json_dir", type=str,
                    help="Path to the folder hosting the dataset json file",
                    default="data/dataset_json_settings")
parser.add_argument("--dataset_json_fname", type=str,
                    help="filename of the data (USE 4_4_Beats_gmd.jsom for only beat sections,"
                         " and 4_4_BeatsAndFills_gmd.json for beats and fill samples combined",
                    default="4_4_Beats_gmd.json")
parser.add_argument("--evaluate_on_subset", type=str,
                    help="Using test or evaluation subset for evaluating the model", default="test",
                    choices=['test', 'evaluation'])

# ----------------------- Evaluation Params -----------------------
parser.add_argument("--calculate_hit_scores_on_train", type=strtobool,
                    help="Evaluates the quality of the hit models on training set",
                    default=False)
parser.add_argument("--calculate_hit_scores_on_test", type=strtobool,
                    help="Evaluates the quality of the hit models on test/evaluation set",
                    default=False)
parser.add_argument("--piano_roll_samples", type=strtobool, help="Generate piano rolls", default=False)
parser.add_argument("--piano_roll_frequency", type=int, help="Frequency of piano roll generation", default=20)
parser.add_argument("--hit_score_frequency", type=int, help="Frequency of hit score generation", default=10)

# ----------------------- Misc Params -----------------------
parser.add_argument("--save_model", type=bool, help="Save model", default=True)
parser.add_argument("--save_model_dir", type=str, help="Path to save the model", default="misc/Control_VAE")
parser.add_argument("--save_model_frequency", type=int, help="Save model every n epochs", default=10)

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
    d_model_dec = int(float(args.d_model_enc) * float(args.d_model_dec_ratio))
    dim_feedforward_enc = int(float(args.d_ff_enc_to_dmodel) * float(args.d_model_enc))
    dim_feedforward_dec = int(float(args.d_ff_dec_to_dmodel) * d_model_dec)
    num_decoder_layers = int(float(args.n_enc_lyrs) * float(args.n_dec_lyrs_ratio))

    hparams = dict(
        d_model_enc=args.d_model_enc,
        d_model_dec=d_model_dec,
        embedding_size_src=args.embedding_size_src,
        embedding_size_tgt=args.embedding_size_tgt,
        nhead_enc=args.nhead_enc,
        nhead_dec=args.nhead_dec,
        dim_feedforward_enc=dim_feedforward_enc,
        dim_feedforward_dec=dim_feedforward_dec,
        num_encoder_layers=int(args.n_enc_lyrs),
        num_decoder_layers=num_decoder_layers,
        max_len_enc=args.max_len_enc,
        max_len_dec=args.max_len_dec,
        latent_dim=args.latent_dim,

        dropout=args.dropout,
        velocity_dropout=args.velocity_dropout,
        offset_dropout=args.offset_dropout,

        use_in_attention=args.use_in_attention,
        n_continuous_params=args.n_continuous_params,
        train_density=args.train_density,
        train_intensity=args.train_intensity,
        train_genre=args.train_genre,

        device="cuda" if torch.cuda.is_available() else "cpu",

        o_activation="tanh" if args.offset_loss_function == "mse" else "sigmoid",
        hit_loss_function=args.hit_loss_function,
        velocity_loss_function=args.velocity_loss_function,
        offset_loss_function=args.offset_loss_function,

        hit_loss_balancing_beta=float(args.hit_loss_balancing_beta),
        genre_loss_balancing_beta=float(args.genre_loss_balancing_beta),

        beta_annealing_activated=args.beta_annealing_activated,
        beta_level=args.beta_level,
        beta_annealing_per_cycle_rising_ratio=float(args.beta_annealing_per_cycle_rising_ratio),
        beta_annealing_per_cycle_period=args.beta_annealing_per_cycle_period,
        beta_annealing_start_first_rise_at_epoch=args.beta_annealing_start_first_rise_at_epoch,

        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        reduce_loss_by_sum=True if args.reduce_loss_by_sum == 1 else False,
        is_testing=args.is_testing,
        dataset_json_dir=args.dataset_json_dir,
        dataset_json_fname=args.dataset_json_fname
    )

# config files without wandb_project specified
if args.wandb_project is not None:
    hparams["wandb_project"] = args.wandb_project

assert "wandb_project" in hparams.keys(), "wandb_project not specified"

if __name__ == "__main__":

    # Initialize wandb
    # ----------------------------------------------------------------------------------------------------------
    wandb_run = wandb.init(
        config=hparams,  # either from config file or CLI specified hyperparameters
        project=hparams["wandb_project"],  # name of the project
        entity="mmil_julian",  # saves in the mmil_vae_cntd team account
        settings=wandb.Settings(code_dir="train.py")  # for code saving
    )

    # Reset config to wandb.config (in case of sweeping with YAML necessary)
    # ----------------------------------------------------------------------------------------------------------
    config = wandb.config
    run_name = wandb_run.name
    run_id = wandb_run.id
    collapse_tapped_sequence = (args.embedding_size_src == 3)
    # Load Training and Testing Datasets and Wrap them in torch.utils.data.Dataloader
    # ----------------------------------------------------------------------------------------------------------
    # only 1% of the dataset is used if we are testing the script (is_testing==True)
    should_place_all_data_on_cuda = args.force_data_on_cuda and torch.cuda.is_available()

    # Get our definition of genres for one-hot encoding
    with open('data/control/gmd_genre_dict.json', 'r') as f:
        genre_dict = json.load(f)
        config['genre_dict'] = genre_dict

    training_dataset = GrooveDataSet_Control(
        dataset_setting_json_path="data/dataset_json_settings/4_4_BeatsAndFills_gmd.json",
        subset_tag="train",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=collapse_tapped_sequence,
        down_sampled_ratio=0.1 if args.is_testing else None,
        move_all_to_gpu=False,
        hit_loss_balancing_beta=0,
        genre_loss_balancing_beta=0,
        custom_genre_mapping_dict=genre_dict
    )

    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = GrooveDataSet_Control(
        dataset_setting_json_path="data/dataset_json_settings/4_4_BeatsAndFills_gmd.json",
        subset_tag="test",
        max_len=32,
        tapped_voice_idx=2,
        collapse_tapped_sequence=collapse_tapped_sequence,
        down_sampled_ratio=0.1 if args.is_testing else None,
        move_all_to_gpu=False,
        hit_loss_balancing_beta=0,
        genre_loss_balancing_beta=0,
        custom_genre_mapping_dict=genre_dict
    )

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the VAEmodel
    # ------------------------------------------------------------------------------------------------------------
    model = GrooveControl_VAE(config)

    groovecontrol_model = model.to(config.device)
    wandb.watch(groovecontrol_model, log="all", log_freq=1)

    # Regressors and Classifiers for latent space disentanglement
    if args.balance_param_loss_weights:
        density_weights, intensity_weights, genre_weights = training_dataset.get_parameter_weights()
        density_loss_fn = torch.nn.BCELoss(weight=density_weights)
        intensity_loss_fn = torch.nn.BCELoss(weight=intensity_weights)
        genre_loss_fn = torch.nn.BCELoss(weight=genre_weights)
    else:
        density_loss_fn = torch.nn.BCELoss()
        intensity_loss_fn = torch.nn.BCELoss()
        genre_loss_fn = torch.nn.BCELoss()


    adversarial_models = {"density": {"active": False},
                          "intensity": {"active": False},
                          "genre": {"active": False}}


    print("\n\nHELLO\n\n")

    if args.train_density:
        model = GAN_components.LatentClassifier(latent_dim=args.latent_dim,
                                                                  n_classes=10,
                                                                  loss_function=density_loss_fn)
        density_regressor_model = model.to(config.device)
        optimizer = torch.optim.Adam(density_regressor_model.parameters(), lr=config.lr)
        adversarial_models["density"] = {"active": True, "model": density_regressor_model, "optimizer": optimizer}
        wandb.watch(density_regressor_model, log="all", log_freq=1)

    if args.train_intensity:
        model = GAN_components.LatentClassifier(latent_dim=args.latent_dim,
                                                                    n_classes=10,
                                                                    loss_function=intensity_loss_fn)
        intensity_regressor_model = model.to(config.device)
        optimizer = torch.optim.Adam(intensity_regressor_model.parameters(), lr=config.lr)
        adversarial_models["intensity"] = {"active": True, "model": intensity_regressor_model, "optimizer": optimizer}
        wandb.watch(intensity_regressor_model, log="all", log_freq=1)

    if args.train_genre:
        model = GAN_components.LatentClassifier(latent_dim=args.latent_dim,
                                                                 n_classes=len(genre_dict),
                                                                 loss_function=genre_loss_fn)
        genre_classifier_model = model.to(config.device)
        optimizer = torch.optim.Adam(genre_classifier_model.parameters(), lr=config.lr)
        adversarial_models["genre"] = {"active": True, "model": genre_classifier_model, "optimizer": optimizer}
        wandb.watch(genre_classifier_model, log="all", log_freq=1)

    # Instantiate the loss Criterion and Optimizer
    # ------------------------------------------------------------------------------------------------------------

    if config.hit_loss_function == "bce":
        hit_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        raise NotImplementedError(f"hit_loss_function {config.hit_loss_function} not implemented")

    if config.velocity_loss_function == "bce":
        velocity_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        velocity_loss_fn = torch.nn.MSELoss(reduction='none')

    if config.offset_loss_function == "bce":
        offset_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        offset_loss_fn = torch.nn.MSELoss(reduction='none')

    if config.optimizer == 'adam':
        vae_optimizer = torch.optim.Adam(groovecontrol_model.parameters(), lr=config.lr)
    else:
        vae_optimizer = torch.optim.SGD(groovecontrol_model.parameters(), lr=config.lr)

    beta_np_cyc = control_loss_functions.generate_beta_curve(
        n_epochs=config.epochs,
        period_epochs=config.beta_annealing_per_cycle_period,
        rise_ratio=config.beta_annealing_per_cycle_rising_ratio,
        start_first_rise_at_epoch=config.beta_annealing_start_first_rise_at_epoch)

    # Iterate over epochs
    # ------------------------------------------------------------------------------------------------------------
    metrics = dict()
    step_ = 0

    for epoch in range(config.epochs):
        print(f"Epoch {epoch} of {config.epochs}, steps so far {step_}")

        # Run the training loop (trains per batch internally)
        # ------------------------------------------------------------------------------------------
        groovecontrol_model.train()
        for model_, component in adversarial_models.items():
            if component["active"]:
                component["model"].train()

        logger.info("***************************Training...")

        if config.beta_annealing_activated:
            beta = float(args.beta_level * beta_np_cyc[epoch])
        else:
            beta = args.beta_level

        adversarial_loss_modifier = generate_theta_rise(epoch, theta_level=0.1,
                                                        epochs_to_reach_theta=50,
                                                        start_first_rise_at_epoch=50)

        train_log_metrics, step_ = control_train_utils.train_loop(
            train_dataloader=train_dataloader,
            vae_model=groovecontrol_model,
            adversarial_models=adversarial_models,
            vae_optimizer=vae_optimizer,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            device=config.device,
            starting_step=step_,
            kl_beta=beta,
            adversarial_loss_modifier=adversarial_loss_modifier,
            reduce_by_sum=config.reduce_loss_by_sum,
            balance_vo=args.balance_vo,
        )

        wandb.log(train_log_metrics, commit=False)
        wandb.log({"kl_beta": beta}, commit=False)

        # ---------------------------------------------------------------------------------------------------
        # After each epoch, evaluate the model on the test set
        #     - To ensure not overloading the GPU, we evaluate the model on the test set also in batche
        #           rather than all at once
        # ---------------------------------------------------------------------------------------------------
        groovecontrol_model.eval()  # DON'T FORGET TO SET THE MODEL TO EVAL MODE (check torch no grad)
        for model, component in adversarial_models.items():
            if component["active"]:
                component["model"].eval()
        logger.info("***************************Testing...")

        test_log_metrics = control_train_utils.test_loop(
            test_dataloader=test_dataloader,
            vae_model=groovecontrol_model,
            adversarial_models=adversarial_models,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            device=config.device,
            kl_beta=beta,
            adversarial_loss_modifier=adversarial_loss_modifier,
            reduce_by_sum=config.reduce_loss_by_sum,
            balance_vo=args.balance_vo
        )

        wandb.log(test_log_metrics, commit=False)
        logger.info(f"Epoch {epoch} Finished with total train loss of {train_log_metrics['train/vae_loss_total']} "
                    f"and test loss of {test_log_metrics['test/vae_loss_total']}")

        # Generate PianoRolls and UMAP Plots  and KL/OA PLots if Needed
        # ---------------------------------------------------------------------------------------------------
        if args.piano_roll_samples:
            if epoch % args.piano_roll_frequency == 0:
                piano_rolls = get_piano_rolls_for_density_model_wandb(vae_model=groovecontrol_model,
                                                                      device=config.device,
                                                                      test_dataset=test_dataset,
                                                                      normalizing_fn=training_dataset.normalize_density
                                                                      if args.normalize_densities else None,
                                                                      reduce_dim=collapse_tapped_sequence)

                wandb.log(piano_rolls, commit=False)

                media = generate_umap_for_density_model_wandb(
                    model=groovecontrol_model,
                    device=config.device,
                    test_dataset=test_dataset,
                    subset_name='test',
                    collapse_tapped_sequence=collapse_tapped_sequence,
                )
                wandb.log(media, commit=False)

        # Get Hit Scores for the entire train and the entire test set
        # ---------------------------------------------------------------------------------------------------
        if args.calculate_hit_scores_on_train:
            if epoch % args.hit_score_frequency == 0:
                logger.info("________Calculating Hit Scores on Train Set...")
                train_set_hit_scores = get_hit_scores_for_density_model(
                    model=groovecontrol_model,
                    device=config.device,
                    dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
                    subset_name='train',
                    down_sampled_ratio=0.1,
                    collapse_tapped_sequence=collapse_tapped_sequence,
                    normalizing_fn=training_dataset.normalize_density if args.normalize_densities else None,
                    cached_folder="eval/GrooveEvaluator/templates",
                    divide_by_genre=False
                )
                wandb.log(train_set_hit_scores, commit=False)

                densities_predictions = get_density_prediction_averages(model=groovecontrol_model,
                                                                        test_dataset=test_dataset,
                                                                        device=config.device,
                                                                        normalizing_fn=training_dataset.normalize_density
                                                                        if args.normalize_densities else None,
                                                                        reduce_dim=collapse_tapped_sequence)
                wandb.log(densities_predictions, commit=False)

        if args.calculate_hit_scores_on_test:
            if epoch % args.hit_score_frequency == 0:
                logger.info("________Calculating Hit Scores on Test Set...")
                test_set_hit_scores = get_hit_scores_for_density_model(
                    model=groovecontrol_model,
                    device=config.device,
                    dataset_setting_json_path=f"{config.dataset_json_dir}/{config.dataset_json_fname}",
                    subset_name=args.evaluate_on_subset,
                    down_sampled_ratio=None,
                    collapse_tapped_sequence=collapse_tapped_sequence,
                    normalizing_fn=training_dataset.normalize_density if args.normalize_densities else None,
                    cached_folder="eval/GrooveEvaluator/templates",
                    divide_by_genre=False
                )
                wandb.log(test_set_hit_scores, commit=False)

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
                groovecontrol_model.save(model_path)
                model_artifact.add_file(model_path)
                wandb_run.log_artifact(model_artifact)
                logger.info(f"Model saved to {model_path}")

    wandb.finish()
