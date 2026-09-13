import torch
import torch.nn as nn
import torchvision.models as models
import h5py
import numpy as np

# -----------------------------------------------
# 1. Define the Dual-Stream Model
# -----------------------------------------------
class DualViewModel(nn.Module):
    def __init__(self, num_classes):
        super(DualViewModel, self).__init__()
        # Use pre-trained ResNet18 models for each camera branch
        self.branch1 = models.resnet18(pretrained=True)
        self.branch2 = models.resnet18(pretrained=True)
        
        # Remove the last fully connected layer (we want to extract features)
        self.branch1.fc = nn.Identity()
        self.branch2.fc = nn.Identity()
        
        # Combine the two feature vectors (each of size 512) into one classification layer
        self.fc = nn.Linear(512 * 2, num_classes)
        
    def forward(self, x1, x2):
        # Get features from both branches
        f1 = self.branch1(x1)  # Features from camera 1
        f2 = self.branch2(x2)  # Features from camera 2
        # Concatenate along the feature dimension
        combined = torch.cat((f1, f2), dim=1)
        # Pass the combined features through the final classifier
        output = self.fc(combined)
        return output

# Create an instance of the model
num_classes = 6  # Example: 6 different identities or expression categories
model = DualViewModel(num_classes)

# -----------------------------------------------
# 2. Save the Model in .pkl Format (Pickle)
# -----------------------------------------------
# torch.save() uses Python’s pickle internally.
torch.save(model, 'dual_view_model.pkl')
print("Model saved as 'dual_view_model.pkl'")

# -----------------------------------------------
# 3. Save the Model's State Dict in .h5 Format
# -----------------------------------------------
# We store the state_dict (a dictionary of parameter names and tensors)
state_dict = model.state_dict()
with h5py.File('dual_view_model.h5', 'w') as h5f:
    for key, tensor in state_dict.items():
        # Save each parameter as a dataset after converting it to a NumPy array.
        h5f.create_dataset(key, data=tensor.cpu().numpy())
print("Model state dict saved as 'dual_view_model.h5'")

# -----------------------------------------------
# 4. Download the Files (if in Google Colab)
# -----------------------------------------------
# In a Google Colab environment, you can use google.colab.files.download to trigger a download.
try:
    from google.colab import files
    print("Downloading 'dual_view_model.pkl' ...")
    files.download('dual_view_model.pkl')
    print("Downloading 'dual_view_model.h5' ...")
    files.download('dual_view_model.h5')
except ImportError:
    print("Not in a Google Colab environment. If you want to download the files, copy them manually.")
