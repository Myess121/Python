import torch.nn as nn
class CNN(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN,self).__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout(0.25),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout(0.25),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),nn.Dropout(0.25),
        )
        self.classifier=nn.Sequential(
            nn.Linear(2048,512),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x=self.conv_layers(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
def get_model(num_classes=10):
    return CNN(num_classes)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)