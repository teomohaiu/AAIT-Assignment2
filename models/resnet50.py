from torchvision.models import ResNet50_Weights, resnet50
from torch import nn

def resnet50_model():
    num_classes = 100
    weights=ResNet50_Weights.DEFAULT
    model= resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False 

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

