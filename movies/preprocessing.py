import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# check whether CUDA (GPU acceleration) is available on the current system
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

# convert the values of the DataFrame to a PyTorch tensor
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).long().to(device)

# converts a pandas DataFrame to a PyTorch tensor
def df_to_tensor_cpu(df):
    return torch.from_numpy(df.values).long()

def load_data(train_percent, val_percent, test_percent):
    path = '../../../movies/'
    data = pd.read_csv(path + 'ratings.csv')

    user_counts = data['userId'].value_counts()
    movie_counts = data['movieId'].value_counts()

    new_df = pd.DataFrame({'userId': user_counts.index, 'count': user_counts.values})
    mew_df = pd.DataFrame({'movieId': movie_counts.index, 'movie_count': movie_counts.values})

        # Merge df with new_df on the user id column
    data = data.merge(new_df, on='userId', how='right')
    data = data.merge(mew_df, on='movieId', how='right')

    # Filter out users with less than 3 reviews
    data = data[data['count'] >= 50]
    data = data[data['movie_count'] >= 100]


    # Drop the 'count' column if you don't need it in the final result
    data = data.drop('count', axis=1)
    data = data.drop('movie_count', axis=1)

    data['userId'] = data['userId'].astype('category')
    data['user_id_num'] = data['userId'].cat.codes
    user_id_to_num = dict(zip(data['userId'], data['user_id_num']))

    data['movieId'] = data['movieId'].astype('category')
    data['product_id_num'] = data['movieId'].cat.codes
    prod_id_to_num = dict(zip(data['movieId'], data['product_id_num']))

    data = data.drop(columns='userId')
    data = data.drop(columns=['movieId'])
    print("DATA.HEAD() -------------------------------------------------------------------")
    print(data.head())


    pickle.dump(user_id_to_num, open(path + 'user_id_to_num10.pkl', 'wb'))
    pickle.dump(prod_id_to_num, open(path + 'prod_id_to_num10.pkl', 'wb'))
    np.save(path + 'data10.npy', data.values)

    training = data.sample(frac=train_percent)

    left = data.drop(training.index)
    validation = left.sample(frac=val_percent / (val_percent + test_percent))

    test = left.drop(validation.index)

    print("loaded")

    return df_to_tensor_cpu(training), df_to_tensor_cpu(validation), df_to_tensor_cpu(test), user_id_to_num, prod_id_to_num

if __name__ == "__main__":
    train, val, test, user, rest = load_data(0.6, 0.3, 0.1)
    print("TRAIN ----------------------------------------------")
    print(train.shape)
    print("VAL ----------------------------------------------")
    print(val.shape)
    print("TEST ----------------------------------------------")
    print(test.shape)
