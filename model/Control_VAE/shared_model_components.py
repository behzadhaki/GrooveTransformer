import torch
from copy import deepcopy


def create_layers(module, N):
    # User to stack multiple Encoder layers for In-Attention mechanism
    return torch.nn.ModuleList([deepcopy(module) for i in range(N)])


class InAttentionEncoder(torch.nn.Module):
    """
    This function is a modified version of the In-Attention Transformer from Musemorphose paper
    https://arxiv.org/abs/2105.04090
    We create a stack of num_encoder_layers transformer encoders. Separately, we expect control parameters to be
    a vector of num_params length. This vector is expanded to d_model with a learnable Linear layer.
    Then, at each Layer, it is summed with the source (which initially comes from sampling VAE latent space)
    and fed through the next iteration of multi-head self-attention.
    """

    def __init__(self, d_model, n_head, dim_feedforward, dropout, num_encoder_layers, num_params):
        super(InAttentionEncoder, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.layers = create_layers(encoder_layer, num_encoder_layers)
        self.parameter_projection = torch.nn.Linear(num_params, d_model)

    def forward(self, x, parameters):
        parameters = self.parameter_projection(parameters)
        for mod in self.layers:
            x += parameters
            x = mod(x)
        return x


class LatentRegressor(torch.nn.Module):
    """
    Regressor to predict a single continuous value from the latent space. For example,
    if latent space has 256 values, it will look at 255 of these and predict a single value.
    """
    def __init__(self, latent_dim, activate_output=True):
        super(LatentRegressor, self).__init__()
        latent_dim = latent_dim - 1
        self.fc_layer_1 = torch.nn.Linear(latent_dim, latent_dim, bias=True)
        self.fc_layer_2 = torch.nn.Linear(latent_dim, latent_dim, bias=True)
        self.fc_output_layer = torch.nn.Linear(latent_dim, 1)
        self.fc_activation_1 = torch.nn.ReLU()
        self.fc_activation_2 = torch.nn.ReLU()
        self.fc_output_activation = torch.nn.ReLU() if activate_output else None

    def forward(self, input):
        x = self.fc_layer_1(input)
        x = self.fc_activation_1(x)
        x = self.fc_layer_2(x)
        x = self.fc_activation_2(x)
        output = self.fc_output_layer(x).squeeze()  # squeeze to match the single-dimensionality of control vector
        if self.fc_output_activation is not None:
            output = self.fc_output_activation(output)

        return output


class LatentClassifier(torch.nn.Module):
    def __init__(self, latent_dim, n_classes, class_labels_list):
        super(LatentClassifier, self).__init__()
        assert n_classes == len(class_labels_list)
        assert all([isinstance(x, str) for x in class_labels_list])
        self.class_labels_list = class_labels_list

        latent_dim = latent_dim - n_classes
        self.fc_layer_1 = torch.nn.Linear(latent_dim, latent_dim, bias=True)
        self.fc_layer_2 = torch.nn.Linear(latent_dim, latent_dim, bias=True)
        self.fc_output_layer = torch.nn.Linear(latent_dim, n_classes)
        self.fc_activation_1 = torch.nn.ReLU()
        self.fc_activation_2 = torch.nn.ReLU()

    def forward(self, input):
        x = self.fc_layer_1(input)
        x = self.fc_activation_1(x)
        x = self.fc_layer_2(x)
        x = self.fc_activation_2(x)
        logits = self.fc_output_layer(x)

        return logits
