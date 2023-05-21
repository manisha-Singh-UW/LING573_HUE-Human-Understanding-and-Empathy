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

# for AdvancedDropout
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# import project modules
import conf
import utils


class AdvancedDropout(Module):
    # reference: Advanced Dropout: A Model-free Methodology for Bayesian Dropout Optimization (IEEE TPAMI 2021)
    # https://arxiv.org/abs/2010.05244
    def __init__(self, num, init_mu=0, init_sigma=1.2, reduction=16):
        '''
        params:
        num (int): node number
        init_mu (float): intial mu
        init_sigma (float): initial sigma
        reduction (int, power of two): reduction of dimention of hidden states h
        '''
        super(AdvancedDropout, self).__init__()
        if init_sigma <= 0:
            raise ValueError("Sigma has to be larger than 0, but got init_sigma=" + str(init_sigma))
        self.init_mu = init_mu
        self.init_sigma = init_sigma

        self.weight_h = Parameter(torch.rand([num // reduction, num]).mul(0.01))
        self.bias_h = Parameter(torch.rand([1]).mul(0.01))

        self.weight_mu = Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_mu = Parameter(torch.Tensor([self.init_mu]))
        self.weight_sigma = Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_sigma = Parameter(torch.Tensor([self.init_sigma]))

    def forward(self, input):
        if self.training:
            c, n = input.size()
            # parameterized prior
            h = F.linear(input, self.weight_h, self.bias_h)
            mu = F.linear(h, self.weight_mu, self.bias_mu).mean()
            sigma = F.softplus(F.linear(h, self.weight_sigma, self.bias_sigma)).mean()
            # mask
            if torch.cuda.is_available():
                epsilon = mu + sigma * torch.randn([c, n]).cuda()
            else:
                epsilon = mu + sigma * torch.randn([c, n])
            mask = torch.sigmoid(epsilon)

            out = input.mul(mask).div(torch.sigmoid(mu.data / torch.sqrt(1. + 3.14 / 8. * sigma.data ** 2.)))
        else:
            out = input
        return out



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.device_check_parameter = nn.Parameter(torch.empty(0))
        self.layers = nn.Sequential(
            AdvancedDropout(embedding_vector_length),
            nn.Linear(embedding_vector_length, 256),
            nn.GELU(),
            AdvancedDropout(256),
            nn.Linear(256, 128),
            nn.GELU(),
            AdvancedDropout(128),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layers(x)


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Dropout(p=0.4),
#             nn.Linear(1536, 256),
#             nn.ReLU(),
#             nn.Dropout(p=0.4),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(p=0.4),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         return self.layers(x)


if __name__ == '__main__':
    utils.set_seed()

    utils.setup_logging(log_filename='log_inference_empathy')
    logging.info('**** starting empathy inference logging ****')

    # load dev data for validation
    dev_df = pd.read_csv(conf.dev_data_with_embedding, sep='\t')

    embedding_vector_length = 1584 # includes 1536 original features and 48 additional lexical features

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

    outputs_dir = Path(conf.outputs_d3_empathy_path).parent
    if not outputs_dir.exists():
        outputs_dir.mkdir()

    dev_pred_df = pd.DataFrame(y_dev_pred)
    dev_pred_df.to_csv(conf.outputs_d3_empathy_path, sep='\t', index=False, header=['DevEmpathyPredictions'])

    logging.info('**** finish empathy inference logging ****')
