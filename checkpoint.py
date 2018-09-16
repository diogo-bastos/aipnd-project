import torch
from functions import create_model

class Checkpoint:
     def __init__(self, model_state_dict, class_to_idx, arch, hidden_units):
         self.model_state_dict = model_state_dict
         self.class_to_idx = class_to_idx
         self.architecture = arch
         self.hidden_units = hidden_units
            
def save_checkpoint(model, class_to_idx, save_directory, arch, hidden_units):
    checkpoint = Checkpoint(model.state_dict(), class_to_idx, arch, hidden_units)
    torch.save(checkpoint, save_directory)

def load_checkpoint(checkpoint_directory):
    checkpoint = torch.load('checkpoint.pth')
    model = create_model(checkpoint.architecture, checkpoint.hidden_units)
    model.load_state_dict(checkpoint.model_state_dict)
    model.class_to_idx = checkpoint.class_to_idx
    return model


