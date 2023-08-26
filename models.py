import torch.nn as nn
import torchvision


def model_list():
    """
    Just a bunch of possible models to use for transfer learning
    # Comments are Top1 Acc, Top5 Acc, Params, GFLOPS
    """
    # Comments are Top1 Acc, Top5 Acc, Params, GFLOPS
    # 67.668, 87.402, 2.5M, 0.06
    mobilenetv3_small = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    # 75.274, 92.566, 5.5M, 0.22
    mobilenetv3_large = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2")
    # 72.996, 91.086, 3.5M, 0.3
    shufflenet_v2_x0 = torchvision.models.shufflenet_v2_x1_5(weights="IMAGENET1K_V1")
    # 75.804, 92.742, 4.3M, 0.4
    regnet400f = torchvision.models.regnet_x_400mf(weights="IMAGENET1K_V1")
    # 78.828, 94.502, 6.4M, 0.83
    regnet800f = torchvision.models.regnet_x_800mf(weights="IMAGENET1K_V1")
    # 69.758, 89.078, 11.7M, 1.81
    resnet18 = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    # 80.858, 95.434, 25.6M, 4.09
    resnet50 = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    # 77.294, 93.45, 27.2M, 5.71
    inceptionv3 = torchvision.models.inception_v3(weights="IMAGENET1K_V1")
    # 84.228, 96.878, 21.5M, 8.37
    efficient_netv2_s = torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1")


def make_mobilenetv3(n_output=112, small=True):
    """
    Loads a pretrained mobilenet v3 (small or large), and replaces the
    last linear layer with a linear layer to `n_output`.
    Use as the x -> c (input to concept) layer of a model.
    Sets every layer, except the last one, to not trainable.

    Args:
        subset (int, optional): Set to the amount of attributes to use.
            Will just be used as the amount of output nodes to use.
        small (bool, optional): If small, will load the small mobilenet v3
            model, if not, will load the large one. The small has about 2.5m
            parameters while the large has about 5.5m. Defaults to True.

    Returns:
        model: The mobilenet model.
    """
    if small:
        mobilenetv3 = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    else:
        mobilenetv3 = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2")

    for param in mobilenetv3.parameters():  # Freeze all layers
        param.requires_grad = False

    in_features = mobilenetv3.classifier[3].in_features  # Last layer of model
    mobilenetv3.classifier[3] = nn.Linear(in_features, n_output)

    for param in mobilenetv3.classifier[3].parameters():
        param.requires_grad = True  # Make last layer trainable

    return mobilenetv3


def make_attr_model_single(n_input=112, n_output=200):
    """
    A simple single layer output layer, to be extended on the pretrained model.

    Args:
        n_input (int, optional): The inputs (attributes). Defaults to 112.
        n_output (int, optional): The outputs (classes). Defaults to 200.

    Returns:
        model: The model.
    """
    model = nn.Sequential(
        nn.Linear(n_input, n_output)
    )
    return model


def make_attr_model_double(n_input=112, n_hidden=200, n_output=200):
    """
    A simple single layer output layer, to be extended on the pretrained model.

    Args:
        n_input (int, optional): The inputs (attributes). Defaults to 112.
        n_hidden (int, optional): The amount of nodes int he hidden layer.
            Defaults to 200.
        n_output (int, optional): The outputs (classes). Defaults to 200.

    Returns:
        model: The model.
    """
    model = nn.Sequential(
        nn.Linear(n_input, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_output)
    )
    return model


class CombineModels(nn.Module):
    """
    Combines two models. The forward function will output both the final outputs,
    and the intermediary outputs from the first model.
    This is meant to be used as a Concept Bottleneck Model, where the first model is
    pretrained and ouputs the amount of attributes (112 for CUB dataset), and the
    second model is the top layer(s) and takes 112 as input and output the amount
    of classes (200).
    """
    def __init__(self, first_model, second_model, activation_func="sigmoid"):
        super(CombineModels, self).__init__()
        self.first_model = first_model
        self.second_model = second_model
        self.activation_func = activation_func
        if activation_func is None:
            self.actication_func = None
        elif activation_func.lower() == "relu":
            self.actication_func = nn.ReLU()
        elif activation_func.lower() == "sigmoid":
            self.activation_func = nn.Sigmoid()

    def forward(self, inputs):
        attr_outputs = self.first_model(inputs)
        if self.activation_func is not None:
            attr_outputs = self.activation_func(attr_outputs)
        outputs = self.second_model(attr_outputs)
        return outputs, attr_outputs
