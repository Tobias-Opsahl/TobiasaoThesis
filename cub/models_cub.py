"""
Models for the CUB dataset.
NOTE: mobilenetv3s weight does not work for torch version 1, which is ran at ml.hpc.uio.
It should be swapped for another pretrained model, or examined in more detail.
"""
import torch
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


class LinearWithInput(nn.Module):
    """
    Custom Linear Layer that also returns the input. Used for skip
    connections on pretrained models.
    """

    def __init__(self, in_features, out_features):
        super(LinearWithInput, self).__init__()
        self.linear_layer = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        output = self.linear_layer(x)
        return output, x


class Identity(nn.Module):
    """
    Identity function, in order to `remove` a layer of a pytorch model.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CBMWithSkip(nn.Module):
    """
    CBM model with skip connection.
    """
    def __init__(self, n_attr, n_output, n_hidden, sigmoid, concat=False, small=True):
        """
        Model looks as follows:
        Pretrained -> x -> attribute_layer -> y + x -> class_layer
        Constructor. Can make residual or concatinated skip connections.

        Args:
            n_attr (int): Nodes in the attribute layer.
            n_output (int): Nodes in the output layer
            n_hidden (int): Nodes in the hidden layer (both before attribute
                layer and class layer).
            sigmoid (bool): Wether or not sigmoid is used in the attribute layer.
            concat (bool, optional): Wether or not the skip connection is concatinated.
                If not, they are added (like in ResNet). Defaults to False.
            small (bool, optional): Wether or not to use small mobilenetv3 model.
                Defaults to True.
        """
        super(CBMWithSkip, self).__init__()
        self.concat = concat
        # Load pretrained model
        if small:
            mobilenetv3 = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        else:
            mobilenetv3 = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2")

        for param in mobilenetv3.parameters():  # Freeze all layers
            param.requires_grad = False

        in_features = mobilenetv3.classifier[3].in_features  # Last layer of model
        mobilenetv3.classifier[3] = nn.Linear(in_features, n_hidden)

        for param in mobilenetv3.classifier[3].parameters():
            param.requires_grad = True  # Make last layer trainable
        self.pretrained = mobilenetv3

        self.attribute_classifier = nn.Linear(n_hidden, n_attr)
        self.sigmoid = None
        if sigmoid:  # Sigmoid on attribute layer
            self.sigmoid = nn.Sigmoid()
        self.class_classifier1 = nn.Linear(n_attr, n_hidden)
        self.class_activation = nn.ReLU()
        if concat:  # Concatinate skip connection (DenseNet style)
            self.class_classifier2 = nn.Linear(2 * n_hidden, n_output)
        else:  # Residual skip connection (ResNet Style)
            self.class_classifier2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        pretrained_output = self.pretrained(x)
        attr_output = self.attribute_classifier(pretrained_output)
        if self.sigmoid is not None:
            attr_output = self.sigmoid(attr_output)
        if self.concat:  # Densenet skip connection
            y = self.class_classifier1(attr_output)
            x = torch.concat((y, pretrained_output), dim=1)
        else:  # Residual connection
            x = self.class_classifier1(attr_output) + pretrained_output

        x = self.class_activation(x)
        x = self.class_classifier2(x)
        return x, attr_output


class CombineModels(nn.Module):
    """
    Combines two models. The forward function will output both the final outputs,
    and the intermediary outputs from the first model.
    This is meant to be used as a Concept Bottleneck Model, where the first model is
    pretrained and ouputs the amount of attributes (112 for CUB dataset), and the
    second model is the top layer(s) and takes 112 as input and output the amount
    of classes (200).
    """
    def __init__(self, first_model, second_model, sigmoid=True):
        """
        Constructer of the model described above.

        Args:
            first_model (model): The first model (pretrained to attributes)
            second_model (model): Second model (attributes to classes)
            sigmoid (bool, optional): Wether or not to use sigmoid activation
                function at the attribute layer. Defaults to True.
        """
        super(CombineModels, self).__init__()
        self.first_model = first_model
        self.second_model = second_model
        self.activation_func = None
        if sigmoid is not None:
            self.activation_func = nn.Sigmoid()

    def forward(self, inputs):
        attr_outputs = self.first_model(inputs)
        if self.activation_func is not None:
            attr_outputs = self.activation_func(attr_outputs)
        outputs = self.second_model(attr_outputs)
        return outputs, attr_outputs


def make_cbm(n_classes, n_attr=112, sigmoid=True, double_top=True, n_hidden=256, small=True):
    """
    A normal CBM model, made with a pretrained model, classifier layers
    and then combined with CombineModels.
    Uses Mobilenetv3 (small or large) as the pretrained model

    Args:
        n_classes (int): Amount of output classes.
        n_attr (int, optional): Amount of attributes. Defaults to 112.
        sigmoid (bool): Wether or not to use sigmoid activation function at the
            attribute layer.
        double_top (bool, optional): If True, will use double Linear layer
            at the top. Defaults to True.
        n_hidden (int, optional): Number of hidden nodes to use in hidden layer
            if `double_top` is True. Defaults to 256.
        small (bool, optional): Wether to use small or large mobilenetv3. Defaults to True.

    Returns:
        model: The model
    """
    mobilenetv3 = make_mobilenetv3(n_output=n_attr, small=small)
    if double_top:
        top_model = make_attr_model_double(n_input=n_attr, n_hidden=n_hidden, n_output=n_classes)
    else:
        top_model = make_attr_model_single(n_input=n_attr, n_output=n_classes)
    model = CombineModels(mobilenetv3, top_model, sigmoid=sigmoid)
    return model


class SequentialConceptModel(nn.Module):
    """
    Model based on CBM that trains on concepts, but concepts are distributed
    along layers, instead of all in one layer.
    """
    def __init__(self, n_attr, n_hidden, n_classes, small=True):
        super(SequentialConceptModel, self).__init__()
        self.n_attr = n_attr
        # Load pretrained model
        if small:
            mobilenetv3 = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        else:
            mobilenetv3 = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2")

        for param in mobilenetv3.parameters():  # Freeze all layers
            param.requires_grad = False

        in_features = mobilenetv3.classifier[3].in_features  # Last layer of model
        mobilenetv3.classifier[3] = nn.Linear(in_features, n_hidden)

        for param in mobilenetv3.classifier[3].parameters():
            param.requires_grad = True  # Make last layer trainable
        self.pretrained = mobilenetv3

        self.linear_layers = []
        self.activation_functions = []
        self.batch_norms = []
        self.output_layers = []
        for _ in range(n_attr):
            self.linear_layers.append(nn.Linear(n_hidden, n_hidden))
            self.batch_norms.append(nn.BatchNorm1d(n_hidden))
            self.activation_functions.append(nn.ReLU())
            self.output_layers.append(nn.Linear(n_hidden, 1))
        self.classifier = nn.Linear(n_hidden + n_attr, n_classes)

    def forward(self, x):
        x = self.pretrained(x)
        pretrained_output = x  # TODO: Check that this is not inplace
        concepts = []
        for i in range(self.n_attr):
            x = self.linear_layers[i](x)
            x = self.batch_norms[i](x)
            x = self.activation_functions[i](x)
            concept_output = self.output_layers[i](x)
            concepts.append(concept_output)
        concepts = torch.concat(concepts, dim=1)  # Turn list into tensor
        x = torch.concat((x, concepts), dim=1)
        x = self.classifier(x)
        return x, concepts
