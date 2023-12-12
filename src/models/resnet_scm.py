import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet18SCM(nn.Module):
    def __init__(self, n_classes=200, n_attr=112, n_linear_output=256, n_hidden=256, dropout_probability=0.2,
                 attribute_activation_function=None, hard=False):
        super(ResNet18SCM, self).__init__()
        self.name = "CubSCM"
        self.short_name = "scm"

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

        # self.resnet18 = models.resnet18(pretrained=True)
        # Overwrite last layer
        self.resnet18 = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        num_in_features = self.resnet18.fc.in_features  # Input to last layer
        self.resnet18.fc = nn.Linear(num_in_features, n_linear_output)
        for param in self.resnet18.parameters():
            param.requires_grad = False
        for param in self.resnet18.fc.parameters():  # Make new linear layer trainable
            param.requires_grad = True

        self.layers = list(self.resnet18.children())

        n_concept_output_nodes = [1, 1, 1, 1, 1]
        for i in range(n_attr - 5):  # Gradually add concept output nodes to match the amount of attributes
            index = i % 5
            n_concept_output_nodes[index] += 1

        # Pooling layers to reduce the intermediary dimensions
        self.adaptive_pool1 = nn.AdaptiveAvgPool2d((8, 8))
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.adaptive_pool3 = nn.AdaptiveAvgPool2d((2, 2))

        self.concept_layer1 = nn.Linear(8192, n_concept_output_nodes[0])  # 8 * 32 * 32
        self.concept_layer2 = nn.Linear(4096, n_concept_output_nodes[1])  # 16 * 16 * 16
        self.concept_layer3 = nn.Linear(2048, n_concept_output_nodes[2])  # 32 * 8 * 8
        self.concept_layer4 = nn.Linear(n_linear_output, n_concept_output_nodes[3])
        self.concept_layer5 = nn.Linear(n_hidden, n_concept_output_nodes[4])
        self.dropout = nn.Dropout(dropout_probability)
        self.intermediary_classifier = nn.Linear(n_linear_output, n_hidden)
        self.final_classifier = nn.Linear(n_hidden + n_attr, n_classes)

    def forward(self, x):
        concepts = []
        # Forward pass through initial layers
        x = self.layers[0](x)  # Conv1
        x = self.layers[1](x)  # BatchNorm1
        x = self.layers[2](x)  # ReLU
        x = self.layers[3](x)  # MaxPool
        x = self.layers[4](x)  # BasicBlock
        

        x = self.layers[5](x)  # BasicBlock

        intermediate_output = self.adaptive_pool1(x)
        intermediate_output = torch.flatten(intermediate_output, 1)
        intermediate_output = self.dropout(intermediate_output)
        concept1 = self.concept_layer1(intermediate_output)
        concepts.append(concept1)

        x = self.layers[6](x)  # BasicBlock
        intermediate_output = self.adaptive_pool2(x)
        intermediate_output = torch.flatten(intermediate_output, 1)
        intermediate_output = self.dropout(intermediate_output)
        concept2 = self.concept_layer2(intermediate_output)
        concepts.append(concept2)

        x = self.layers[7](x)  # BasicBlock
        intermediate_output = self.adaptive_pool3(x)
        intermediate_output = torch.flatten(intermediate_output, 1)
        intermediate_output = self.dropout(intermediate_output)
        concept3 = self.concept_layer3(intermediate_output)
        concepts.append(concept3)

        x = self.layers[8](x)  # Adaptive Average Pool 2d
        x = torch.flatten(x, 1)
        x = self.layers[9](x)  # Linear layer
        x = F.relu(x)
        x = self.dropout(x)
        concept4 = self.concept_layer4(x)
        concepts.append(concept4)

        x = self.intermediary_classifier(x)  # Out: N x 32
        x = F.relu(x)
        concept5 = self.concept_layer5(x)
        concepts.append(concept5)
        concepts = torch.concat(concepts, dim=1)

        if self.use_sigmoid:
            y = F.sigmoid(concepts)
        elif self.use_relu:
            y = F.relu(concepts)
        else:
            y = concepts
        if self.hard:
            y = torch.round(y)

        x = torch.concat((x, y), dim=1)
        x = self.final_classifier(x)  # Out: N x n_classes

        return x, concepts
