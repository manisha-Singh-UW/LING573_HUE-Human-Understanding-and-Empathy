# Create and train the model

# import standard libraries
import pandas as pd
import numpy as np
import copy
import logging
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split

# import project modules
import conf
import utils


class ProjDataset(Dataset):
    def __init__(self, df):
        # load the training and target data
        embedding_vector_length = 1536
        emb_columns = [f'e{i}' for i in range(embedding_vector_length)]

        x = df[emb_columns].values
        y = df['empathy'].values

        # convert to torch tensors
        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    utils.set_seed()

    utils.setup_logging(log_filename='log_model_empathy')
    logging.info('**** starting empathy model logging ****')

    train_df = pd.read_csv(conf.train_data_with_embedding, sep='\t')
    logging.info('Training dataset')
    logging.info(f'\n{train_df.head()}')

    embedding_vector_length = 1536
    emb_columns = [f'e{i}' for i in range(embedding_vector_length)]

    modeling_df = pd.DataFrame(columns=emb_columns)

    for row in range(len(train_df)):
        emb_str = train_df.loc[row, 'essay_emb']
        str_without_brackets = emb_str.strip()[1:-1]
        emb_np = np.fromstring(str_without_brackets, dtype=np.float32, sep=',')
        modeling_df.loc[row] = emb_np.tolist()

    modeling_df = modeling_df.join(train_df['empathy'])
    modeling_df = modeling_df.join(train_df['distress'])

    logging.info('Training DataFrame')
    logging.info(f'\n{modeling_df}')

    train_df, val_df = train_test_split(modeling_df, test_size=0.2, random_state=573, shuffle=True)
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()

    x_val = torch.tensor(val_df[emb_columns].values, dtype=torch.float32)
    y_val = torch.tensor(val_df['empathy'], dtype=torch.float32)
    y_val.unsqueeze_(1)

    torch_dataset = ProjDataset(modeling_df)

    train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=30, shuffle=True)

    model_nn = NeuralNetwork()

    # loss function and optimizer
    loss_fn = nn.MSELoss()

    optimizer = optim.AdamW(model_nn.parameters(), lr=1e-4)  # default weight_decay=0.01

    epochs = 100
    train_history = []
    dev_history = []

    best_mse = np.inf
    best_mse_epoch = -1
    best_weights = None

    for e in range(epochs):
        model_nn.train()
        epoch_loss = 0.0
        epoch_count = 0
        for features, labels in train_loader:
            labels.unsqueeze_(1)
            # forward pass
            output = model_nn(features)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()  # clear previous gradients
            loss.backward()  # backward pass
            optimizer.step()

            # print stats
            epoch_loss += loss.item()
            epoch_count += 1

        # at the end of each epoch
        epoch_loss = epoch_loss / epoch_count
        model_nn.eval()
        y_dev_pred = model_nn(x_val)
        dev_loss_mse = loss_fn(y_dev_pred, y_val)

        logging.info(f'epoch: {e}; training_loss: {float(epoch_loss)}; dev_loss: {float(dev_loss_mse)}')
        train_history.append(float(epoch_loss))
        dev_history.append(float(dev_loss_mse))
        if dev_loss_mse < best_mse:
            best_mse = dev_loss_mse
            best_mse_epoch = e
            best_weights = copy.deepcopy(model_nn.state_dict())

    # restore the best weights
    model_nn.load_state_dict(best_weights)
    logging.info(f'Best MSE: {best_mse}; Best MSE Epoch: {best_mse_epoch}')

    logging.info(f'Train History: {train_history}')
    logging.info(f'Dev History: {dev_history}')

    torch.save(model_nn, conf.model_nn_empathy_save_path)
    if Path(conf.model_nn_empathy_save_path).exists():
        logging.info(f'Model file successfully written to: {conf.model_nn_empathy_save_path}')

    logging.info('**** finish empathy model logging ****')
