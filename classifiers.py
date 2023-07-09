import torch
import torchvision.models as models
import torch.nn as nn

class MultilabelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.model = nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.dent = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=1)
        )
        self.scratch = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=1)
        )
        self.crack = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=1)
        )
        self.glass_shatter = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=1)
        )
        self.broken_lamp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=1)
        )
        self.tire = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=1)
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)

        return {
            'dent': self.dent(x),
            'scratch': self.scratch(x),
            'crack': self.crack(x),
            'glass_shatter': self.glass_shatter(x),
            'lamp_broken': self.broken_lamp(x),
            'tire_flat': self.tire(x)
        }