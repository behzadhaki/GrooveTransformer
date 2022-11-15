if __name__ == "__main__":
    from model import load_groove_transformer_encoder_model
    from model.saved.monotonic_groove_transformer_v1.old.params import model_params

    for model_name in model_params.keys():
        old_model_path = f"model/saved/monotonic_groove_transformer_v1/old/{model_name}.model"
        latest_model_path = f"model/saved/monotonic_groove_transformer_v1/latest/{model_name}.pth"
        params_dict = model_params[model_name]
        GrooveTransformer = load_groove_transformer_encoder_model(old_model_path, params_dict)
        GrooveTransformer.save(latest_model_path)

