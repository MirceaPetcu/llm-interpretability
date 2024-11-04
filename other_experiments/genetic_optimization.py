import torch
import torchga
import pygad
from scipy.stats import pearsonr
import numpy as np

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model

    # Convert the solution (a flat vector) into a model-compatible dictionary of weights.
    model_weights_dict = torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)

    # Make predictions using the current model weights.
    predictions = model(data_inputs).detach().numpy()

    # Calculate Pearson correlation between predictions and actual outputs.
    # Note: `pearsonr` returns both the correlation and the p-value, we only need the correlation value.
    correlation, _ = pearsonr(predictions.flatten(), data_outputs.numpy().flatten())

    # Since we're maximizing fitness, return the absolute value of the correlation as fitness.
    # Higher correlation implies better fitness.
    solution_fitness = abs(correlation)

    return solution_fitness

def callback_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

# Define the PyTorch model (MLP).
input_layer = torch.nn.Linear(3, 2)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(2, 1)

model = torch.nn.Sequential(input_layer, relu_layer, output_layer)

# Create an instance of the TorchGA class to generate the initial population.
torch_ga = torchga.TorchGA(model=model, num_solutions=10)

import pickle
with open('../processed_dataset_mixtral_8x7B_instruct_qlora_nf4_reflect_forward.pkl', 'rb') as f:
    data = pickle.load(f)

# Example input data.
data_inputs = torch.tensor([i['embeddgins_17'] for i in data]).squeeze()

# Example output data.
data_outputs = torch.tensor([i['target'] for i in data]).squeeze()

# PyGAD genetic algorithm parameters.
num_generations = 250  # Number of generations to evolve.
num_parents_mating = 5  # Number of parents to select for mating.
initial_population = torch_ga.population_weights  # Initial population of network weights.
parent_selection_type = "sss"  # Stochastic Universal Sampling selection.
crossover_type = "single_point"  # Single-point crossover.
mutation_type = "random"  # Random mutation.
mutation_percent_genes = 10  # Percentage of genes to mutate.
keep_parents = -1  # Keep all parents from the previous generation.

# Create a PyGAD GA instance.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

# Run the genetic algorithm.
ga_instance.run()

# Plot the evolution of fitness over generations.
ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Get the best solution after training.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution fitness = {solution_fitness}")
print(f"Best solution index   = {solution_idx}")

# Load the best model weights into the PyTorch model.
best_solution_weights = torchga.model_weights_as_dict(model=model, weights_vector=solution)
model.load_state_dict(best_solution_weights)

# Make predictions using the best model.
predictions = model(data_inputs)
print("Predictions: ", predictions.detach().numpy())

# Calculate and print Pearson correlation for the best model.
correlation, _ = pearsonr(predictions.detach().numpy().flatten(), data_outputs.numpy().flatten())
print("Pearson correlation for best solution: ", correlation)
