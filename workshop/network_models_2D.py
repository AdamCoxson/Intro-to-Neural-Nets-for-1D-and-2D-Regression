# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:44:39 2025

Author: Adam Coxson, PhD student, University of Liverpool
Department of Chemistry, Materials Innovation Factory, Levershulme Research Centre
Module: network_models_2D.py
Local dependencies: network_classes.py, stats_and_plot_functions.py
For the dML workshop Mar 2025

# https://medium.com/@shashankshankar10/introduction-to-neural-networks-build-a-single-layer-perceptron-in-pytorch-c22d9b412ccf
# Numpy only perceptron: https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html
# https://www.kaggle.com/code/pinocookie/pytorch-simple-mlp

Trains a neural network on 2D regression data and visualises the results.

"""

############## PACKAGE IMPORTS ###############
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

############### LOCAL IMPORTS ###############
from network_classes import SingleLayerPerceptron, MLP_2layer, MLP 
from stats_and_plot_functions import compute_z, sample_xy_values, normalize_data, mean_squared_error, plot_train_loss, heatmapplot, surfaceplot, calc_num_net_parameters

torch.manual_seed(42) # 42 for reproducibility
x_min_max = [-5, 10]
y_min_max = [-8, 8]
dxdy=[0.0001,0.0001]

# num_samples = 1000
# sampled_x, sampled_y = sample_xy_values(x_min_max, y_min_max, dxdy, num_samples)
# sampled_z = compute_z(sampled_x, sampled_y)
# heatmapplot(sampled_x, sampled_y, sampled_z, ["x","y","z","Randomly sampled xy data"])

# Generate uniform grid of x-y values for plotting and validation data
x_values = np.linspace(x_min_max[0], x_min_max[1], 100)
y_values = np.linspace(y_min_max[0], y_min_max[1], 100)
uniform_z = []
uniform_x = []
uniform_y = []
for x in x_values:
    for y in y_values:
        uniform_x.append(x)
        uniform_y.append(y)
        uniform_z.append(compute_z(x,y))
uniform_z = np.array(uniform_z)
#heatmapplot(uniform_x, uniform_y, uniform_z, ["x","y","z","Uniformly sampled xy data"])
#surface_plot(uniform_x, uniform_y, uniform_z, ["x", "y", "z", "True surface"])

# Hyperparameters
input_dim = 2           # We only have two input dimension (the scalar x and y value)
output_dim = 1          # We only have one output dimension (the scalar z value)
learning_rate = 0.001
num_epochs = 100
batch_size = 128
neurons=[300,200,100]
# See list of different activation functions https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
activation = [nn.SELU(), nn.ReLU(), nn.Sigmoid()][0]
dropout=0.0 # Careful with this as it may currently be broken

layers=[] # Format layers for variable MLP class
for n in neurons:
    layers.append((n, activation))

# Initialize model, loss function, and optimizer
model = MLP(input_dim,output_dim,layers,dropout_prob=dropout)
criterion = nn.MSELoss()  
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

if type(neurons) is list:
    nparams= calc_num_net_parameters(neurons.copy(), output_size=1)
else:
    nparams= calc_num_net_parameters(neurons, output_size=1)

# Training data (random) sampling
training_samples= 40000

# Create and preprocess training data into 32 bit floats for PyTorch torch.Tensor objects 
x_vals, y_vals = sample_xy_values(x_min_max, y_min_max, dxdy, training_samples)
z_vals = compute_z(x_vals, y_vals).astype(np.float32)
#heatmapplot(x_vals, y_vals, z_vals, ["x","y","z","Randomly sampled training data"])
x_vals_normalized, x_mean, x_std = normalize_data(x_vals)
y_vals_normalized, y_mean, y_std = normalize_data(y_vals)
z_vals_normalized, z_mean, z_std = normalize_data(z_vals)
train_input = torch.tensor(np.concatenate((x_vals_normalized.reshape(len(x_vals_normalized),1),
                            y_vals_normalized.reshape(len(y_vals_normalized),1)), axis=1), device='cpu')
train_target = torch.tensor(z_vals_normalized.reshape(len(z_vals),1), device='cpu')

# Create validation data, which in this case will be the 100x100 uniform grid.
# Note, I am DELIBERATELY using the mean and std from the TRAINING DATA to rescale the validation data
x_uniform_normalized = (((np.array(uniform_x) - x_mean) / x_std).astype(np.float32)).reshape(len(uniform_x),1)
y_uniform_normalized = (((np.array(uniform_y) - y_mean) / y_std).astype(np.float32)).reshape(len(uniform_y),1)
z_uniform_normalized = (((np.array(uniform_z) - z_mean) / z_std).astype(np.float32))
valid_input = torch.tensor(np.concatenate((x_uniform_normalized.reshape(len(x_uniform_normalized),1),
                            y_uniform_normalized.reshape(len(y_uniform_normalized),1)), axis=1), device='cpu')
valid_target_normalized = z_uniform_normalized

# Create DataLoader and define optimiser
train_dataset = TensorDataset(train_input, train_target)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop with validation tracking
train_losses=[]
valid_losses=[]
best_valid_loss, best_train_loss, best_epoch = 1e6, 1e6, 0
best_model_state = None  # To store the best model state dict
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train() # Tell model we are in training mode, weights can be modified, later we switch to evaluation mode
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        batch_X, batch_y = batch_X.float(), batch_y.float()
        outputs = model(batch_X).squeeze()  # squeeze to ensure output shape matches target
        loss = criterion(outputs, batch_y.squeeze())  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()
    train_loss = total_loss/len(train_loader)
    
    model.eval() # Important to tell the model we're in evaluation model, ensures no weights are changed by accident
    with torch.no_grad():
        valid_pred_norm = model(valid_input).squeeze().detach().numpy() # Predict validation data
    valid_loss = mean_squared_error(valid_pred_norm, valid_target_normalized) # Loss of true norm vs pred norm validation data
    train_losses.append(train_loss), valid_losses.append(valid_loss)
    
    # Save the best model based on validation loss
    if valid_loss < best_valid_loss: 
        best_epoch = epoch
        best_valid_loss, best_train_loss = valid_loss, train_loss
        best_model_state = model.state_dict().copy()  # Save model state
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.5f}, Valid Loss: {valid_loss:.5f}")  
print("\nTraining complete.\n")
print("Number of training samples:",training_samples)
print("Neurons:",neurons,"("+str(nparams)+" parameters)","\nNum Epochs:", num_epochs,
      "\nLearning Rate:",learning_rate,"\nBatch Size:",batch_size,"\nActivation:",activation)

plot_train_loss(train_losses, valid_losses)

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Best model from Epoch {best_epoch+1} restored with Training Loss: {best_train_loss:.5f} and Validation Loss: {best_valid_loss:.5f}")

valid_pred_normalised = np.squeeze(model(valid_input).detach().numpy())
valid_pred = (valid_pred_normalised * z_std) + z_mean # Unscale the prediction data
diff = np.squeeze(valid_pred)-np.array(uniform_z).astype(np.float32)  # Deviation between true and predicted data

"""
####################### VISUALISE AND COMPARE THE TRUE AND PREDICTED DATA #######################
This is important to consider.
Look at the 2D heatmaps, how easy is it to intuitively see where the model performs well?
Keep note of the of colourbars, is the scale range similar? (fun fact, viridis is known as a perceptually uniform colourmap)
Computing the deviation maps helps to clearly see variation in performance, what does the scale of the deviation colourbar describe? 
Now look at the 3D surface plots, does this convey the information better or worse?

"""

heatmapplot(uniform_x, uniform_y, np.squeeze(valid_pred), ["x","y","Prediction z", "Prediction Heatmap"])
heatmapplot(uniform_x, uniform_y, uniform_z, ["x","y","True z", "True Heatmap"])
heatmapplot(uniform_x, uniform_y, diff, ["x","y","Pred-True z deviation", "Deviation Heatmap"])

surfaceplot(uniform_x, uniform_y, uniform_z, ["x", "y", "z", "True surface"])
surfaceplot(uniform_x, uniform_y, np.squeeze(valid_pred), ["x", "y", "z", "Predicted surface"])
surfaceplot(uniform_x, uniform_y, diff, ["x", "y", "z", "Deviation surface"])
    

    
