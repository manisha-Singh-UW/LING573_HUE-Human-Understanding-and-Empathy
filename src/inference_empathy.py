# Perform inference using the trained model and the Dev dataset

# import standard libraries
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.optim as optim

# import project modules
import conf
import utils


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    utils.set_seed()

    utils.setup_logging(log_filename='log_inference_empathy')
    logging.info('**** starting empathy inference logging ****')

    # load dev data for validation
    dev_df = pd.read_csv(conf.dev_data_with_embedding, sep='\t')

    embedding_vector_length = 1536
    emb_columns = [f'e{i}' for i in range(embedding_vector_length)]
    dev_modeling_df = pd.DataFrame(columns=emb_columns)

    for row in range(len(dev_df)):
        emb_str = dev_df.loc[row, 'essay_emb']
        str_without_brackets = emb_str.strip()[1:-1]
        emb_np = np.fromstring(str_without_brackets, dtype=np.float32, sep=',')
        dev_modeling_df.loc[row] = emb_np.tolist()

    dev_modeling_df = dev_modeling_df.join(dev_df['empathy'])
    dev_modeling_df = dev_modeling_df.join(dev_df['distress'])

    logging.info('Dev DataFrame')
    logging.info(f'\n{dev_modeling_df}')

    x_dev = torch.tensor(dev_modeling_df[emb_columns].values, dtype=torch.float32)
    y_dev = torch.tensor(dev_modeling_df['empathy'], dtype=torch.float32)

    model_nn = torch.load(conf.model_nn_empathy_save_path)
    model_nn.eval()

    y_dev_pred = model_nn(x_dev)
    y_dev_pred = torch.squeeze(y_dev_pred)
    y_dev_pred = y_dev_pred.tolist()

    y_dev_gold = y_dev.squeeze()
    y_dev_gold = y_dev_gold.tolist()

    y_dev_pred = model_nn(x_dev)
    y_dev_pred = torch.squeeze(y_dev_pred)
    y_dev_pred = y_dev_pred.tolist()

    y_dev_gold = y_dev.squeeze()
    y_dev_gold = y_dev_gold.tolist()

    outputs_dir = Path(conf.outputs_d2_empathy_path).parent
    if not outputs_dir.exists():
        outputs_dir.mkdir()

    dev_pred_df = pd.DataFrame(y_dev_pred)
    dev_pred_df.to_csv(conf.outputs_d2_empathy_path, sep='\t', index=False, header=['DevEmpathyPredictions'])

    logging.info('**** finish empathy inference logging ****')
