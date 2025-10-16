import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class BaselineClassification(nn.Module):
    """
    VGG16-based classification model
    
    Outputs:
        pre-Softmax scores (logits) for no tumor/tumor classes
    """
    def __init__(self, dropout_prob=0.3, freeze_features=True):
        super().__init__()
        base = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = base.features
        self.avgpool  = base.avgpool

        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
   
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        logits = self.classifier(x)

        return logits