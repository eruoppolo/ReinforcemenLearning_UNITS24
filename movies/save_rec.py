from model import Actor, Critic, DRRAveStateRepresentation, PMF
from learn import DRRTrainer
from utils.general import csv_plot
import torch
import pickle
import numpy as np
import random
import os
import datetime

import matplotlib.pyplot as plt


class config():
    output_path = '../results/' + '290724-000000' + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plot_dir = output_path + 'rewards.pdf'

    train_actor_loss_data_dir = output_path + 'train_actor_loss_data.npy'
    train_critic_loss_data_dir = output_path + 'train_critic_loss_data.npy'
    train_mean_reward_data_dir = output_path + 'train_mean_reward_data.npy'

    train_actor_loss_plot_dir = output_path + 'train_actor_loss.png'
    train_critic_loss_plot_dir = output_path + 'train_critic_loss.png'
    train_mean_reward_plot_dir = output_path + 'train_mean_reward.png'

    trained_models_dir = '../../../movies/'

    actor_model_trained = trained_models_dir + 'actor_net.weights'
    critic_model_trained = trained_models_dir + 'critic_net.weights'
    state_rep_model_trained = trained_models_dir + 'state_rep_net.weights'

    actor_model_dir = output_path + 'actor_net.weights'
    critic_model_dir = output_path + 'critic_net.weights'
    state_rep_model_dir = output_path + 'state_rep_net.weights'

    csv_dir = output_path + 'log.csv'

    path_to_trained_pmf = trained_models_dir + 'ratio_0.800000_bs_100000_e_35_wd_0.100000_lr_0.000500_trained_pmf.pt'

    # hyperparams
    batch_size = 64
    gamma = 0.90
    replay_buffer_size = 200000
    history_buffer_size = 5
    learning_start = 5000
    learning_freq = 1
    lr_state_rep = 0.001
    lr_actor = 0.0005
    lr_critic = 0.001
    eps_start = 1
    eps = 0.1
    eps_steps = 5000
    eps_eval =  0.1
    tau = 0.001
    beta = 0.4
    prob_alpha = 0.5
    max_timesteps_train = 200000
    max_epochs_offline = 500
    max_timesteps_online = 6000# 20000
    embedding_feature_size = 100
    episode_length = 25
    train_ratio = 0.8
    weight_decay = 0.001
    clip_val = 1.0
    log_freq = 25
    saving_freq = 1000
    zero_reward = False

    no_cuda = True

    @classmethod
    def print_config(cls):
        print("\n--- Configuration Settings ---")
        for attr, value in cls.__dict__.items():
            if not attr.startswith("__") and not callable(value):
                print(f"{attr}: {value}")
        print("-----------------------------\n")

def seed_all(cuda, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed=seed)

print("Initializing DRR Framework ----------------------------------------------------------------------------")

# Get CUDA device if available
cuda = True if not config.no_cuda and torch.cuda.is_available() else False
print("Using CUDA") if cuda else print("Using CPU")

# Init seeds
# seed_all(mps, 0)
seed_all(cuda, 0)
print("Seeds initialized")

# Grab models
actor_function = Actor
critic_function = Critic
state_rep_function = DRRAveStateRepresentation

path = '../../../movies/'

# Import Data
users = pickle.load(open(path + 'user_id_to_num10.pkl', 'rb'))
items = pickle.load(open(path + 'prod_id_to_num10.pkl', 'rb'))
data = np.load(path + 'data10.npy', allow_pickle=True)

# Normalize rewards to [-1, 1]
data[:, 0] = 0.5 * (data[:, 0] - 3)

data = data.astype(np.float64)

# Shuffle data
np.random.shuffle(data)

# Split data
train_data = torch.from_numpy(data[:int(config.train_ratio * data.shape[0])])
test_data = torch.from_numpy(data[int(config.train_ratio * data.shape[0]):])

print("Data imported, shuffled, and split into Train/Test, ratio=", config.train_ratio)
print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)

# Create and load PMF function for rewards and embeddings
n_users = len(users)
n_items = len(items)

reward_function = PMF(n_users, n_items, config.embedding_feature_size, is_sparse=False, no_cuda=~cuda) #~mps
reward_function.load_state_dict(torch.load(config.path_to_trained_pmf, map_location=torch.device('cpu')))

# Freeze all the parameters in the network
for param in reward_function.parameters():
    param.requires_grad = False
print("Initialized PMF, imported weights, created reward_function")

# Extract embeddings
user_embeddings = reward_function.user_embeddings.weight.data
item_embeddings = reward_function.item_embeddings.weight.data
print("Extracted user and item embeddings from PMF")
print("User embeddings shape: ", user_embeddings.shape)
print("Item embeddings shape: ", item_embeddings.shape)

print('--------------------------------------- INITIAL SETTINGS ----------------------------------------------')
config.print_config()

# Init trainer
print("Initializing DRRTrainer -------------------------------------------------------------------------------")
trainer = DRRTrainer(config,
                      actor_function,
                      critic_function,
                      state_rep_function,
                      reward_function,
                      users,
                      items,
                      train_data,
                      test_data,
                      user_embeddings,
                      item_embeddings,
                      cuda #mps #cuda
                      )

# Change to newest trained data directories
config.trained_models_dir = config.output_path
output_path = config.output_path
#config.trained_models_dir = "../results/230724-120000/"
#output_path = "../results/230724-120000/"

train_actor_loss_data_dir = output_path + 'train_actor_loss_data.npy'
train_critic_loss_data_dir = output_path + 'train_critic_loss_data.npy'
train_mean_reward_data_dir = output_path + 'train_mean_reward_data.npy'

config.actor_model_trained = config.trained_models_dir + 'actor_net.weights'
config.critic_model_trained = config.trained_models_dir + 'critic_net.weights'
config.state_rep_model_trained = config.trained_models_dir + 'state_rep_net.weights'


sourceFile = open(output_path + "hyperparams.txt", 'w')
print(config.__dict__, file = sourceFile)
sourceFile.close()

Ts = [10]

# Extract data from simulation

for T in Ts:

  avgs = []
  print(f'----------- EVALUATING reward@{T}---------------')
  # Change T
  config.episode_length = T
  us = trainer.save_rec()

