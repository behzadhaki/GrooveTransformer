import torch


class LatentRegressor(torch.nn.Module):
    """
    Regressor to predict a single continuous value from the latent space. For example,
    if latent space has 256 values, it will look at 255 of these and predict a single value.
    """
    def __init__(self, latent_dim, activate_output=True):
        super(LatentRegressor, self).__init__()
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
    def __init__(self, latent_dim, n_classes, loss_function):
        super(LatentClassifier, self).__init__()

        self.fc_layer_1 = torch.nn.Linear(latent_dim, latent_dim, bias=True)
        self.fc_layer_2 = torch.nn.Linear(latent_dim, latent_dim, bias=True)
        self.fc_output_layer = torch.nn.Linear(latent_dim, n_classes)
        self.fc_activation_1 = torch.nn.Tanh()
        self.fc_activation_2 = torch.nn.Tanh()
        self.fc_output_activation = torch.nn.Sigmoid()
        self.loss_function = loss_function

    def forward(self, latent_z):
        x = self.fc_layer_1(latent_z)
        x = self.fc_activation_1(x)
        x = self.fc_layer_2(x)
        x = self.fc_activation_2(x)
        x = self.fc_output_layer(x)
        logits = self.fc_output_activation(x)

        return logits

    def calculate_loss(self, predictions, targets):
        return self.loss_function(predictions, targets)


class LatentContinuousClassifier(torch.nn.Module):
    # Potentially deprecated; was trying to recreate a paper's model for continuous classification
    def __init__(self, latent_dim, loss_function):
        super(LatentContinuousClassifier, self).__init__()
        latent_dim -= 1
        self.fc_layer_1 = torch.nn.Linear(latent_dim, latent_dim)
        self.fc_layer_2 = torch.nn.Linear(latent_dim, latent_dim)
        self.output_layer = torch.nn.Linear(latent_dim, 10)

        self.activation_layer_1 = torch.nn.Tanh()
        self.activation_layer_2 = torch.nn.Tanh()
        self.output_activation = torch.nn.Sigmoid()

        self.loss_function = loss_function

    def forward(self, latent_z):
        x = self.fc_layer_1.forward(latent_z)
        x = self.activation_layer_1(x)
        x = self.fc_layer_2.forward(x)
        x = self.activation_layer_2(x)
        output = self.output_layer.forward(x)
        output = self.output_activation(output)
        return output

    def calculate_loss(self, predictions, targets):
        return self.loss_function(predictions, targets)



