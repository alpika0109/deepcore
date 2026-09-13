import torch
import torch.nn as nn
import torchvision.models as models

class DualViewModel(nn.Module):
    def __init__(self, num_classes):
        super(DualViewModel, self).__init__()
        # Create two branches with pre-trained ResNet18 models
        self.branch1 = models.resnet18(pretrained=True)
        self.branch2 = models.resnet18(pretrained=True)
        
        # Remove the last fully connected layer to extract features (ResNet18 outputs a 512-dim vector)
        self.branch1.fc = nn.Identity()
        self.branch2.fc = nn.Identity()
        
        # Combine the two 512-dim feature vectors and pass them through a new fully connected layer for classification.
        self.fc = nn.Linear(512 * 2, num_classes)
        
    def forward(self, x1, x2):
        # Get features from each branch
        f1 = self.branch1(x1)  # from camera 1
        f2 = self.branch2(x2)  # from camera 2
        
        # Concatenate features along the feature dimension
        combined = torch.cat((f1, f2), dim=1)
        
        # Classify the combined features
        output = self.fc(combined)
        return output

# Example: if you have 5 classes (e.g., 5 different identities or 5 expression categories)
num_classes = 5
model = DualViewModel(num_classes)
