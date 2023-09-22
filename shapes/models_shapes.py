import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapesBaseModel(nn.Module):
    """
    A basemodel for many of the Shapes-models.
    Expects an input size of (3 x 64 x 64).

    This model contains three convolutional layers with ReLU and maxpooling, followed by
    one linear layer with an optional amount of output nodes, and a final ReLU.
    This is useful since both the normal CNN and many of the CBMs use this as a base model.
    """
    def __init__(self, n_output=64):
        """
        Args:
            n_output (int, optional): The amount of output nodes from the linear layer. Defaults to 64.
        """
        super(ShapesBaseModel, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2048, n_output)  # 32 * 8 * 8 = 2048

    def forward(self, x):
        # Input size: [batch_size, channels, height, width],  N x 3 x 64 x 64
        x = self.conv1(x)  # Out: N x 8 x 64 x 64
        x = F.relu(x)
        x = self.pool(x)  # Out: N x 8 x 32 x 32
        x = self.conv2(x)  # Out: N x 16 x 32 x 32
        x = F.relu(x)
        x = self.pool(x)  # Out: N x 16 x 16 x 16
        x = self.conv3(x)  # Out N x 32 x 16 x 16
        x = F.relu(x)
        x = self.pool(x)  # Out: N x 32 x 8 x 8
        # Flatten all dimensions except batch-size
        x = torch.flatten(x, 1)  # Out: N x 2048 (N x (32 * 8 * 8))
        x = self.fc1(x)  # Out: N x n_output
        x = F.relu(x)
        return x


class ShapesCNN(ShapesBaseModel):
    """
    A simple feed forward CNN. Uses the base model, and then two more linear layers.
    Does not perform a softmax, it should be infered in the loss-function.
    """
    def __init__(self, n_classes, n_linear_output=64):
        """
        Args:
            n_classes (int): The amount of classes (output nodes for this model).
            n_linear_output (int, optional): The amount of output nodes from the base model's linear layer.
                Defaults to 64.
        """
        super(ShapesCNN, self).__init__(n_linear_output)
        self.name = "ShapesCNN"

        self.intermediary_classifier = nn.Linear(n_linear_output, 32)
        self.final_classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        x = super(ShapesCNN, self).forward(x)  # Base model, Out: N x n_linear_output
        x = self.intermediary_classifier(x)  # Out: N x 32
        x = F.relu(x)
        x = self.final_classifier(x)  # Out: N x n_classes

        return x


class ShapesCBM(ShapesBaseModel):
    """
    A Join Concept Bottleneck Model, as from the Concept Bottleneck Models paper. https://arxiv.org/abs/2007.04612

    First gets the output from the basemodel, then passes that into a bottleneck layer. The bottleneck layer consist
    of as many nodes as there are attributes/concepts. This is then given into one or two more layers, and then it
    is classified on.

    The concepts are returned with the final output, so that one can train on both.
    No softmax or sigmoid are done to the output of forward, (altough a sigmoid or relu may be applied to the
    bottleneck-layer before it is passed to the next layers) so be sure to infer this in the loss function.
    """
    def __init__(self, n_classes, n_attr, n_linear_output=64, attribute_activation_function=None, two_layers=True):
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
            two_layers (bool): If True, will have two linear layers after the bottleneck layer. If False, will
                only use one (which corresponds to logistic regression on the bottleneck layer).
        """
        super(ShapesCBM, self).__init__(n_linear_output)
        self.name = "ShapesCBM"
        self.use_sigmoid = False
        self.use_relu = False
        attribute_activation_function = attribute_activation_function.strip().lower()
        if attribute_activation_function == "sigmoid":
            self.use_sigmoid = True
        elif attribute_activation_function == "relu":
            self.use_relu = True
        self.two_layers = two_layers

        self.attribute_classifier = nn.Linear(n_linear_output, n_attr)
        if self.two_layers:
            self.intermediary_classifier = nn.Linear(n_attr, n_attr)
        self.final_classifier = nn.Linear(n_attr, n_classes)

    def forward(self, x):
        x = super(ShapesCBM, self).forward(x)  # Base model, Out: N x n_linear_output
        concepts = self.attribute_classifier(x)  # Out: N x n_attr
        if self.use_sigmoid:
            x = F.sigmoid(concepts)
        elif self.use_relu:
            x = F.relu(concepts)
        else:
            x = concepts
        if self.two_layers:
            x = self.intermediary_classifier(x)  # Out: N x n_attr
            x = F.relu(x)
        x = self.final_classifier(x)  # Out: N x n_classes

        return x, concepts


class ShapesCBMWithResidual(ShapesBaseModel):
    """
    A Concept Bottleneck Model with a residual skip connection, as in ResNet. https://arxiv.org/abs/1512.03385

    This first gets the linear output from the base model. This output is then given to the bottleneck layer, and then
    to a layer with as many nodes as the output from the basemodel. This is then added up and sent to the last
    linear classification layer.

    The concepts are returned with the final output, so that one can train on both.
    No softmax or sigmoid are done to the output of forward, (altough a sigmoid or relu may be applied to the
    bottleneck-layer before it is passed to the next layers) so be sure to infer this in the loss function.
    """
    def __init__(self, n_classes, n_attr, n_linear_output=64, attribute_activation_function=None):
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
        """
        super(ShapesCBMWithResidual, self).__init__(n_linear_output)
        self.name = "ShapesCBMWithResidual"
        self.use_sigmoid = False
        self.use_relu = False
        attribute_activation_function = attribute_activation_function.strip().lower()
        if attribute_activation_function == "sigmoid":
            self.use_sigmoid = True
        elif attribute_activation_function == "relu":
            self.use_relu = True

        self.attribute_classifier = nn.Linear(n_linear_output, n_attr)
        self.intermediary_classifier = nn.Linear(n_attr, n_linear_output)
        self.final_classifier = nn.Linear(n_linear_output, n_classes)

    def forward(self, x):
        # Input size: [batch_size, channels, height, width],  N x 3 x 64 x 64
        x = super(ShapesCBMWithResidual, self).forward(x)  # Base model, Out: N x n_linear_output
        identity = x
        concepts = self.attribute_classifier(x)  # Out: N x n_attr
        if self.use_sigmoid:
            x = F.sigmoid(concepts)
        elif self.use_relu:
            x = F.relu(concepts)
        else:
            x = concepts
        x = self.intermediary_classifier(x)  # Out: N x n_linear_output
        x = F.relu(x)
        x = x + identity  # Residual connection
        x = self.final_classifier(x)  # Out: N x n_classes

        return x, concepts


class ShapesCBMWithSkip(ShapesBaseModel):
    """
    A Concept Bottleneck Model with a skip connection. The skip connection concatinates, like in DenseNets.

    This model first gets output form the basemodel. Then that output is bassed through the bottleneck layer,
    then to a layer with `n_hidden` nodes, and finally this output is concatinated with the basemodel output and
    put into the last linear classification layer.

    The concepts are returned with the final output, so that one can train on both.
    No softmax or sigmoid are done to the output of forward, (altough a sigmoid or relu may be applied to the
    bottleneck-layer before it is passed to the next layers) so be sure to infer this in the loss function.
    """
    def __init__(self, n_classes, n_attr, n_linear_output=64, attribute_activation_function=None, n_hidden=16):
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
            n_hidden (int): The amount of nodes used in the linear layer after the bottleneck layer.
                This layer will be concatinated on top of the output from the base-model.
        """
        super(ShapesCBMWithSkip, self).__init__(n_linear_output)
        self.name = "ShapesCBMWithSkip"
        self.use_sigmoid = False
        self.use_relu = False
        attribute_activation_function = attribute_activation_function.strip().lower()
        if attribute_activation_function == "sigmoid":
            self.use_sigmoid = True
        elif attribute_activation_function == "relu":
            self.use_relu = True

        self.attribute_classifier = nn.Linear(n_linear_output, n_attr)
        self.intermediary_classifier = nn.Linear(n_attr, n_hidden)
        self.final_classifier = nn.Linear(n_hidden + n_linear_output, n_classes)

    def forward(self, x):
        x = super(ShapesCBMWithSkip, self).forward(x)  # Base model, Out: N x n_linear_output
        identity = x
        concepts = self.attribute_classifier(x)  # Out: N x n_attr
        if self.use_sigmoid:
            x = F.sigmoid(concepts)
        elif self.use_relu:
            x = F.relu(concepts)
        else:
            x = concepts
        x = self.intermediary_classifier(x)  # Out: N x n_hidden
        x = F.relu(x)
        x = torch.concat((x, identity), dim=1)
        x = self.final_classifier(x)  # Out: N x n_classes

        return x, concepts
