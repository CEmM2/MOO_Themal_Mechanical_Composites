#%%
import torch
import torch.nn as nn
import json

class NeuralNetwork(nn.Module):
    def __init__(self, constants):
        super(NeuralNetwork, self).__init__()
        
        # Layer 1
        self.layer1 = nn.Linear(10, 50, dtype=torch.float64)
        self.layer1.weight.data = str_to_tensor(constants['IW1_1']).reshape(50, 10)
        self.layer1.bias.data = str_to_tensor(constants['b1'])
        
        # Layer 2
        self.layer2 = nn.Linear(50, 6, dtype=torch.float64)
        self.layer2.weight.data = str_to_tensor(constants['LW2_1']).reshape(6, 50)
        self.layer2.bias.data = str_to_tensor(constants['b2'])
        
        # Activation functions
        self.activation1 = nn.Tanh()  # tansig in MATLAB
        self.activation2 = nn.Identity()  # purelin in MATLAB
        
        # Store normalization parameters
        self.x1_step1 = {k: str_to_tensor(v) if isinstance(v, list) else torch.tensor([float(v)], dtype=torch.float64)
                         for k, v in constants['x1_step1'].items()}
        self.y1_step1 = {k: str_to_tensor(v) if isinstance(v, list) else torch.tensor([float(v)], dtype=torch.float64)
                         for k, v in constants['y1_step1'].items()}
        
    def forward(self, x):
        # Ensure input is double precision
        x = x.to(torch.float64)
        
        # Input normalization
        x = self.mapminmax_apply(x, self.x1_step1)
        
        # Layer 1
        x = self.layer1(x)
        x = self.activation1(x)
        
        # Layer 2
        x = self.layer2(x)
        x = self.activation2(x)
        
        # Output denormalization
        x = self.mapminmax_reverse(x, self.y1_step1)
        
        return x
    
    @staticmethod
    def mapminmax_apply(x, settings):
        return (x - settings['xoffset']) * settings['gain'] + settings['ymin']
    
    @staticmethod
    def mapminmax_reverse(y, settings):
        return (y - settings['ymin']) / settings['gain'] + settings['xoffset']
### End of NeuralNetwork class
#%%
### func for reading the json file and converting strings to tensors

def str_to_tensor(s):
    return torch.tensor([float(x) for x in s], dtype=torch.float64)
### function for creating the model

def create_model():
    with open('nn_constants_parsed.json', 'r') as f:
        constants = json.load(f)
    return NeuralNetwork(constants)
### function for running the model

def run_model(model, inputs):
    outputs = model(torch.tensor(inputs, dtype=torch.float64))
    return outputs.cpu().detach().numpy()
# %%
#Example of using the model:
# model = create_model()
# inputs = [2.,0.,0.,0.,0.,0.,0.,0.,0.,0.5]
# run_model(model,inputs)
# %%
