import torch.nn as nn
from torchvision import models

def config_model(model_name, num_classes):
    model = None
    available_models = ['fcn_resnet50', 'fcn_resnet101']

    if model_name in available_models:
        if model_name == 'fcn_resnet50':
            model = models.segmentation.fcn_resnet50(pretrained=True)
        elif model_name == 'fcn_resnet101':
            model = models.segmentation.fcn_resnet101(pretrained=True)

        try:
            model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        except Exception:
            raise ValueError(f"The model({model_name}) occurs error during revising classifier.")
    
    else:
        raise ValueError(f"The model({model_name}) is not available.\nYou can use only these models({available_models})")
    
    return model