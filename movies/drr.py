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
    output_path = '../results/' + '290724-000rd' + '/'
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
    max_timesteps_train = 0#200000
    max_epochs_offline = 500
    max_timesteps_online =15000# 20000
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

#data = data[:,:-1] # PMF only need id and stars
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
# Train
print("Starting DRRTrainer.learn() ---------------------------------------------------------------------------")
actor_losses, critic_losses, epi_avg_rewards = trainer.learn()

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


def noiseless_plot(y, title, ylabel, save_loc):
  # operate smoothing
  smoother = ConvolutionSmoother(window_len=1000, window_type='ones')
  smoother.smooth(y)

  # generate intervals
  low, up = smoother.get_intervals('sigma_interval', n_sigma=3)

  # plot the smoothed timeseries with intervals
  plt.close()
  plt.figure(figsize=(11,6))
  plt.xlabel("Epoch")
  plt.ylabel(ylabel)
  plt.title(title)
  plt.plot(smoother.data[0], color='orange')
  plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
  plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)
  plt.savefig(save_loc)
  plt.close()

import matplotlib.pyplot as plt
from tsmoothie.smoother import *

actor_losses = np.load(train_actor_loss_data_dir)
critic_losses = np.load(train_critic_loss_data_dir)
epi_avg_rewards = np.load(train_mean_reward_data_dir)

noiseless_plot(actor_losses,
               "Actor Loss (Train)",
               "Actor Loss (Train)",
               output_path + "train_actor_loss_smooth.png")

noiseless_plot(critic_losses,
               "Critic Loss (Train)",
               "Critic Loss (Train)",
               output_path + "train_critic_loss_smooth.png")

noiseless_plot(epi_avg_rewards,
               "Mean Reward (Train)",
               "Mean Reward (Train)",
               output_path + "train_mean_reward_smooth.png")

sourceFile = open(output_path + "hyperparams.txt", 'w')
print(config.__dict__, file = sourceFile)
sourceFile.close()

T_precisions = [5, 10, 15, 20]

# DRR
for T_precision in T_precisions:
    drr_Ts = []
    for i in range(5):
    #for i in range(20):
        # Evaluate
        avg_precision = trainer.offline_evaluate(T_precision)

        # Append to list
        drr_Ts.append(avg_precision)

    # Save data
    drr_Ts = np.array(drr_Ts)
    np.save(output_path + f'rd_avg_precision@{T_precision}_offline_eval.npy', drr_Ts)

    # Save
    sourceFile = open(output_path + f'rd_avg_precision@{T_precision}_offline_eval.txt', 'w')
    print(f'Random Action Average Precision@{T_precision} (Eval): {np.mean(drr_Ts)}', file=sourceFile)
    sourceFile.close()

Ts = [5, 10, 15, 20]

# Online evaluate
for T in Ts:

  avgs = []
  print(f'----------- EVALUATING reward@{T}---------------')
  # Change T
  config.episode_length = T
  for i in range(5):
    # Evaluate
    print(f'----------- cycle {i+1}---------------')
    avg_reward = trainer.online_evaluate()

    # Append data
    avgs.append(avg_reward)

  # Save data
  avgs = np.array(avgs)
  np.save(output_path + f'rd_avg_reward@{T}_online_eval.npy', avgs)

  # Save
  sourceFile = open(output_path + f'avg_reward@{T}_online_eval.txt', 'w')
  print(f'Random Action Average Reward@{T} (Eval): {np.mean(avgs)}', file=sourceFile)
  sourceFile.close()

# Evaluation @K Graphing

def createEvalPlot(title, ylabel, xlabel, filename, x, y, e, e_x_off, e_y_off):
    plt.figure()
    plt.errorbar(x, y, yerr=e, fmt='.-', ecolor="red", capsize=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x)
    for i, j in zip(x, y):
        plt.annotate(str(round(j, 4)), xy=(i+e_x_off, j+e_y_off))
    plt.savefig(filename)

online_fives = np.load(output_path + 'rd_avg_reward@5_online_eval.npy')
online_tens = np.load(output_path + 'rd_avg_reward@10_online_eval.npy')
online_fifteens = np.load(output_path + 'rd_avg_reward@15_online_eval.npy')
online_twenties = np.load(output_path + 'rd_avg_reward@20_online_eval.npy')

online_eval_data = [online_fives, online_tens, online_fifteens, online_twenties]
online_means, online_stds = [], []

for d in online_eval_data:
        online_means.append(np.mean(d))
        online_stds.append(np.std(d))

print(online_means)
print(online_stds)

# Create and save eval plots

createEvalPlot(
        "Random Action Average Reward @K for Online DRR Evaluation\n(15000 recommendations at each K, 5 times)",
        "Average Reward @K",
        "K",
        output_path + "rd_online_eval.png",
        Ts,
        online_means,
        online_stds,
        0.3,
        -0.004)


drr_fives = np.load(output_path + 'rd_avg_precision@5_offline_eval.npy')
drr_tens = np.load(output_path + 'rd_avg_precision@10_offline_eval.npy')
drr_fifteens = np.load(output_path + 'rd_avg_precision@15_offline_eval.npy')
drr_twenties = np.load(output_path + 'rd_avg_precision@20_offline_eval.npy')


# Combine data

offline_eval_data = [drr_fives, drr_tens, drr_fifteens, drr_twenties]

# Calculate means and stds for graphing
offline_means, offline_stds = [], []

for d in offline_eval_data:
    offline_means.append(np.mean(d))
    offline_stds.append(np.std(d))

print(offline_means)
print(offline_stds)

createEvalPlot("Random Average Precision @K for Offline DRR Evaluation\n(500 random users, K recommendations each, 5 times)",
                   "Average Precision @K",
                   "K",
                   output_path + "rd_offline_eval.png",
                   T_precisions,
                   offline_means,
                   offline_stds,
                   0.4,
                   0)

