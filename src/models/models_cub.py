"""
Note that a lot of this code is similar to Shapes, and could easily be generic instead. However, they where under
constant parallel development, so it was easier to keep them separate.
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from src.constants import USE_XAVIER_INIT_IN_BOTTLENECK


def get_base_model_cub(n_output=256):
    """
    Resnet18 base model for CUB dataset.

    Args:
        n_output (int, optional): The amount of output nodes from the last linear layer. Defaults to 256.

    Returns:
        model: The PyTorch model.
    """
    resnet18 = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    num_in_features = resnet18.fc.in_features  # Input to last layer
    resnet18.fc = nn.Linear(num_in_features, n_output)
    for param in resnet18.parameters():
        param.requires_grad = False
    for param in resnet18.fc.parameters():  # Make new linear layer trainable
        param.requires_grad = True
    return resnet18

class CubCNN(nn.Module):
    """
    A simple feed forward CNN. Uses the base model, and then two more linear layers.
    Does not perform a softmax, it should be infered in the loss-function.
    """
    def __init__(self, n_classes=200, n_linear_output=256, n_hidden=256, dropout_probability=0.2):
        """
        Args:
            n_classes (int): The amount of classes (output nodes for this model).
            n_linear_output (int, optional): The amount of output nodes from the base model's linear layer.
                Defaults to 256.
            n_hidden (int): The amount of nodes in the fully connected linear layer before the output.
            dropout_probability (float): Probability of dropout in layer after the base-model. Probability equal to
                0 means no dropout.
        """
        super(CubCNN, self).__init__()
        self.base_model = get_base_model_cub(n_output=n_linear_output)
        self.name = "CubCNN"
        self.short_name = "cnn"

        self.dropout = nn.Dropout(dropout_probability)
        self.intermediary_classifier = nn.Linear(n_linear_output, n_hidden)
        self.final_classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = self.base_model.forward(x)  # Base model, Out: N x n_linear_output
        x = F.relu(x)
        x = self.dropout(x)
        x = self.intermediary_classifier(x)  # Out: N x n_hidden
        x = F.relu(x)
        x = self.final_classifier(x)  # Out: N x n_classes

        return x

class CubCBM(nn.Module):
    """
    A Join Concept Bottleneck Model, as from the Concept Bottleneck Models paper. https://arxiv.org/abs/2007.04612

    First gets the output from the basemodel, then passes that into a bottleneck layer. The bottleneck layer consist
    of as many nodes as there are attributes/concepts. This is then given into one or two more layers, and then it
    is classified on.

    The concepts are returned with the final output, so that one can train on both.
    No softmax or sigmoid are done to the output of forward, (altough a sigmoid or relu may be applied to the
    bottleneck-layer before it is passed to the next layers) so be sure to infer this in the loss function.
    """
    def __init__(self, n_classes=200, n_attr=112, n_linear_output=256, attribute_activation_function="sigmoid",
                 hard=False, two_layers=True, dropout_probability=0.2):
        """
        Args:
            n_classes (int): The amount of classes (output nodes for this model).
            n_attr (int): The amount of attributes in the dataset. This will be the number of nodes in the
                bottleneck layer.
            n_linear_output (int, optional): The amount of output nodes from the base model's linear layer.
                Defaults to 256.
            attribute_activation_function (str): If and what activation function to use at the concept-bottleneck layer.
                Should be in ["relu", "simgoid", None]. Concept outputs from `forward()` will still be without
                activation function, but activation function may be applied before passing to next layer.
            hard (bool): If True, will pass forward hard concepts. This is concepts that are rounded off to 0 or 1.
                Should be used only with "activation_function" equal to "sigmoid".
            two_layers (bool): If True, will have two linear layers after the bottleneck layer. If False, will
                only use one (which corresponds to logistic regression on the bottleneck layer).
            dropout_probability (float): Probability of dropout in layer after the base-model. Probability equal to
                0 means no dropout.
        """
        super(CubCBM, self).__init__()
        self.name = "CubCBM"
        self.short_name = "cbm"
        self.base_model = get_base_model_cub(n_output=n_linear_output)

        self.use_sigmoid = False
        self.use_relu = False
        if isinstance(attribute_activation_function, str):
            attribute_activation_function = attribute_activation_function.strip().lower()
        if hard and attribute_activation_function != "sigmoid":
            message = f"Attribute activation function must be `sigmoid` when `hard` is True. "
            message += f"Was {attribute_activation_function}. "
            raise ValueError(message)
        if attribute_activation_function == "sigmoid":
            self.use_sigmoid = True
        elif attribute_activation_function == "relu":
            self.use_relu = True
        self.hard = hard
        self.two_layers = two_layers

        self.dropout = nn.Dropout(dropout_probability)

        self.attribute_classifier = nn.Linear(n_linear_output, n_attr)
        if self.two_layers:
            self.intermediary_classifier = nn.Linear(n_attr, n_attr)
        self.final_classifier = nn.Linear(n_attr, n_classes)

        if self.use_sigmoid and USE_XAVIER_INIT_IN_BOTTLENECK:  # Use xavier / glorot intialization for bottlneck
            nn.init.xavier_uniform_(self.attribute_classifier.weight)
            if self.attribute_classifier.bias is not None:
                nn.init.zeros_(self.attribute_classifier.bias)

    def forward(self, x):
        x = self.base_model.forward(x) # Base model, Out: N x n_linear_output
        x = F.relu(x)
        x = self.dropout(x)
        concepts = self.attribute_classifier(x)  # Out: N x n_attr
        if self.use_sigmoid:
            x = F.sigmoid(concepts)
        elif self.use_relu:
            x = F.relu(concepts)
        else:
            x = concepts
        if self.hard:
            x = torch.round(x)
        if self.two_layers:
            x = self.intermediary_classifier(x)  # Out: N x n_attr
            x = F.relu(x)
        x = self.final_classifier(x)  # Out: N x n_classes

        return x, concepts


class CubCBMWithResidual(nn.Module):
    """
    A Concept Bottleneck Model with a residual skip connection, as in ResNet. https://arxiv.org/abs/1512.03385

    This first gets the linear output from the base model. This output is then given to the bottleneck layer, and then
    to a layer with as many nodes as the output from the basemodel. This is then added up and sent to the last
    linear classification layer.

    The concepts are returned with the final output, so that one can train on both.
    No softmax or sigmoid are done to the output of forward, (altough a sigmoid or relu may be applied to the
    bottleneck-layer before it is passed to the next layers) so be sure to infer this in the loss function.
    """
    def __init__(self, n_classes=200, n_attr=112, n_linear_output=256, attribute_activation_function="sigmoid",
                 hard=False, dropout_probability=0.2):
        """
        Args:
            n_classes (int): The amount of classes (output nodes for this model).
            n_attr (int): The amount of attributes in the dataset. This will be the number of nodes in the
                bottleneck layer.
            n_linear_output (int, optional): The amount of output nodes from the base model's linear layer.
                Defaults to 64.
            attribute_activation_function (str): If and what activation function to use at the concept-bottleneck layer.
                Should be in ["relu", "simgoid", None]. Concept outputs from `forward()` will still be without
                activation function, but activation function may be applied before passing to next layer.
            hard (bool): If True, will pass forward hard concepts. This is concepts that are rounded off to 0 or 1.
                Should be used only with "activation_function" equal to "sigmoid".
            dropout_probability (float): Probability of dropout in layer after the base-model. Probability equal to
                0 means no dropout.
        """
        super(CubCBMWithResidual, self).__init__()
        self.name = "CubCBMWithResidual"
        self.short_name = "cbm_res"
        self.base_model = get_base_model_cub(n_output=n_linear_output)

        self.use_sigmoid = False
        self.use_relu = False
        if isinstance(attribute_activation_function, str):
            attribute_activation_function = attribute_activation_function.strip().lower()
        if hard and attribute_activation_function != "sigmoid":
            message = f"Attribute actiation function must be `sigmoid` when `hard` is True. "
            message += f"Was {attribute_activation_function}. "
            raise ValueError(message)
        if attribute_activation_function == "sigmoid":
            self.use_sigmoid = True
        elif attribute_activation_function == "relu":
            self.use_relu = True
        self.hard = hard

        self.dropout = nn.Dropout(dropout_probability)

        self.attribute_classifier = nn.Linear(n_linear_output, n_attr)
        self.intermediary_classifier = nn.Linear(n_attr, n_linear_output)
        self.final_classifier = nn.Linear(n_linear_output, n_classes)

        if self.use_sigmoid and USE_XAVIER_INIT_IN_BOTTLENECK:  # Use xavier / glorot intialization for bottlneck
            nn.init.xavier_uniform_(self.attribute_classifier.weight)
            if self.attribute_classifier.bias is not None:
                nn.init.zeros_(self.attribute_classifier.bias)

    def forward(self, x):
        x = self.base_model.forward(x)  # Base model, Out: N x n_linear_output
        x = F.relu(x)
        x = self.dropout(x)
        identity = x
        concepts = self.attribute_classifier(x)  # Out: N x n_attr
        if self.use_sigmoid:
            x = F.sigmoid(concepts)
        elif self.use_relu:
            x = F.relu(concepts)
        else:
            x = concepts
        if self.hard:
            x = torch.round(x)
        x = self.intermediary_classifier(x)  # Out: N x n_linear_output
        x = F.relu(x)
        x = x + identity  # Residual connection
        x = self.final_classifier(x)  # Out: N x n_classes

        return x, concepts


class CubCBMWithSkip(nn.Module):
    """
    A Concept Bottleneck Model with a skip connection. The skip connection concatinates, like in DenseNets.

    This model first gets output form the basemodel. Then that output is bassed through the bottleneck layer,
    then to a layer with `n_hidden` nodes, and finally this output is concatinated with the basemodel output and
    put into the last linear classification layer.

    The concepts are returned with the final output, so that one can train on both.
    No softmax or sigmoid are done to the output of forward, (altough a sigmoid or relu may be applied to the
    bottleneck-layer before it is passed to the next layers) so be sure to infer this in the loss function.
    """
    def __init__(self, n_classes=200, n_attr=112, n_linear_output=256, attribute_activation_function="sigmoid",
                 n_hidden=128, hard=False, dropout_probability=0.2):
        """
        Args:
            n_classes (int): The amount of classes (output nodes for this model).
            n_attr (int): The amount of attributes in the dataset. This will be the number of nodes in the
                bottleneck layer.
            n_linear_output (int, optional): The amount of output nodes from the base model's linear layer.
                Defaults to 64.
            attribute_activation_function (str): If and what activation function to use at the concept-bottleneck layer.
                Should be in ["relu", "simgoid", None]. Concept outputs from `forward()` will still be without
                activation function, but activation function may be applied before passing to next layer.
            hard (bool): If True, will pass forward hard concepts. This is concepts that are rounded off to 0 or 1.
                Should be used only with "activation_function" equal to "sigmoid".
            n_hidden (int): The amount of nodes used in the linear layer after the bottleneck layer.
                This layer will be concatinated on top of the output from the base-model.
            dropout_probability (float): Probability of dropout in layer after the base-model. Probability equal to
                0 means no dropout.
        """
        super(CubCBMWithSkip, self).__init__()
        self.name = "CubCBMWithSkip"
        self.short_name = "cbm_skip"
        self.base_model = get_base_model_cub(n_output=n_linear_output)

        self.use_sigmoid = False
        self.use_relu = False
        if isinstance(attribute_activation_function, str):
            attribute_activation_function = attribute_activation_function.strip().lower()
        if hard and attribute_activation_function != "sigmoid":
            message = f"Attribute actiation function must be `sigmoid` when `hard` is True. "
            message += f"Was {attribute_activation_function}. "
            raise ValueError(message)
        if attribute_activation_function == "sigmoid":
            self.use_sigmoid = True
        elif attribute_activation_function == "relu":
            self.use_relu = True
        self.hard = hard

        self.dropout = nn.Dropout(dropout_probability)

        self.attribute_classifier = nn.Linear(n_linear_output, n_attr)
        self.intermediary_classifier = nn.Linear(n_attr, n_hidden)

        self.final_classifier = nn.Linear(n_hidden + n_linear_output, n_classes)
        if self.use_sigmoid and USE_XAVIER_INIT_IN_BOTTLENECK:  # Use xavier / glorot intialization for bottlneck
            nn.init.xavier_uniform_(self.attribute_classifier.weight)
            if self.attribute_classifier.bias is not None:
                nn.init.zeros_(self.attribute_classifier.bias)

    def forward(self, x):
        x = self.base_model.forward(x)  # Base model, Out: N x n_linear_output
        x = F.relu(x)
        x = self.dropout(x)
        identity = x
        concepts = self.attribute_classifier(x)  # Out: N x n_attr
        if self.use_sigmoid:
            x = F.sigmoid(concepts)
        elif self.use_relu:
            x = F.relu(concepts)
        else:
            x = concepts
        if self.hard:
            x = torch.round(x)
        x = self.intermediary_classifier(x)  # Out: N x n_hidden
        x = F.relu(x)
        x = torch.concat((x, identity), dim=1)
        x = self.final_classifier(x)  # Out: N x n_classes

        return x, concepts

class CubLogisticOracle(nn.Module):
    """
    Logistic regression from attributes to classes.
    This model is designed to recieve concepts labels at testing time, and is therefore considered an oracle.
    """
    def __init__(self, n_classes, n_attr):
        """
        Args:
            n_classes (int): The amount of classes to predict.
            n_attr (int): The amount of input concepts.
        """
        super(CubLogisticOracle, self).__init__()
        self.name = "CubLogisticOracle"
        self.short_name = "lr_oracle"

        self.n_classes = n_classes
        self.n_attr = n_attr

        self.layer = nn.Linear(n_attr, n_classes)

    def forward(self, x):
        x = self.layer(x)
        return x


class CubNNOracle(nn.Module):
    """
    Two layer neural network from attributes to classes.
    The hidden layer has as many nodes as the input layer, and uses ReLU activation function.
    This model is designed to recieve concepts labels at testing time, and is therefore considered an oracle.
    """
    def __init__(self, n_classes, n_attr):
        """
        Args:
            n_classes (int): The amount of classes to predict.
            n_attr (int): The amount of input concepts.
        """
        super(CubNNOracle, self).__init__()
        self.name = "CubNNOracle"
        self.short_name = "nn_oracle"

        self.n_classes = n_classes
        self.n_attr = n_attr

        self.hidden_layer = nn.Linear(n_attr, n_attr)
        self.output_layer = nn.Linear(n_attr, n_classes)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x
