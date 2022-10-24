# Import model
import torch
import sys
sys.path.insert(1, "../..")
sys.path.insert(1, "..")
from model.src.BasicGrooveTransformer import GrooveTransformerEncoder
from model.saved.monotonic_groove_transformer_v1 import params
from eval.GrooveEvaluator.src.back_compatible_loader import load_evaluator

import pickle


def load_model(model_name, model_path):

    # load model parameters from params.py file
    params_ = params.model_params[model_name]

    # load checkpoint
    checkpoint = torch.load(model_path, map_location=params_['device'])

    # Initialize model
    groove_transformer = GrooveTransformerEncoder(params_['d_model'],
                                                  params_['embedding_sz'],
                                                  params_['embedding_sz'],
                                                  params_['n_heads'],
                                                  params_['dim_ff'],
                                                  params_['dropout'],
                                                  params_['n_layers'],
                                                  params_['max_len'],
                                                  params_['device'])

    # Load model and put in evaluation mode
    groove_transformer.load_state_dict(checkpoint['model_state_dict'])
    groove_transformer.eval()

    return groove_transformer


if __name__ == '__main__':
    # ------------------ Load Trained Model  ------------------ #
    model_name = "colorful_sweep_41"
    model_path = f"../../model/saved/monotonic_groove_transformer_v1/{model_name}.model"

    groove_transformer = load_model(model_name, model_path)

    # load and existing evaluator to ensure consistency

    gmd_eval = load_evaluator(
        f"../../eval/saved/monotonic_groove_transformer_v1/"
        f"validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval")

    gt_hvo_sequences = gmd_eval.get_ground_truth_hvo_sequences()

    src = torch.empty((len(gt_hvo_sequences), 32, 27))
    for i, hvo in enumerate(gt_hvo_sequences):
        src[i, :, :] = torch.tensor(hvo.flatten_voices(voice_idx=2, velocity_aggregator_modes=3))


    prediction_hvos = torch.empty((len(gt_hvo_sequences), 32, 27))

    h, v, o = groove_transformer.predict(src)

    prediction_hvos[:, :, :9] = h
    prediction_hvos[:, :, 9:18] = v
    prediction_hvos[:, :, 18:] = o

    gmd_eval.add_predictions(prediction_hvos.detach().numpy())

    gmd_eval.dump(f"eval/saved/monotonic_groove_transformer_v1/post_training_validation_set_evaluator_run_{model_name}.Eval")


















