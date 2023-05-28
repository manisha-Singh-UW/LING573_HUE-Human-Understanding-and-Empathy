# Deliverable D4: Adaptation Task
# Perform inference using the trained model and the Dev/Test datasets

# import standard libraries
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.optim as optim

# for AdvancedDropout
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# import project modules
# import conf
import utils


class conf:
    adaptation_dev_data = './data/d4_dev_modeling_df.tsv.bz2'
    adaptation_test_data = './data/d4_test_modeling_df.tsv.bz2'
    adaptation_models = {
        'EmotionalPolarity': './models/d4_model_nn_EmotionalPolarity_text-embedding-ada-002.pth',
        'Emotion': './models/d4_model_nn_Emotion_text-embedding-ada-002.pth',
        'Empathy': './models/d4_model_nn_Empathy_text-embedding-ada-002.pth'
    }
    adaptation_devtest_outputs = {
        'EmotionalPolarity': './outputs/D4/adaptation/devtest/d4_devtest_outputs_EmotionalPolarity.tsv',
        'Emotion': './outputs/D4/adaptation/devtest/d4_devtest_outputs_Emotion.tsv',
        'Empathy': './outputs/D4/adaptation/devtest/d4_devtest_outputs_Empathy.tsv',
    }
    adaptation_evaltest_outputs = {
        'EmotionalPolarity': './outputs/D4/adaptation/evaltest/d4_evaltest_outputs_EmotionalPolarity.tsv',
        'Emotion': './outputs/D4/adaptation/evaltest/d4_evaltest_outputs_Emotion.tsv',
        'Empathy': './outputs/D4/adaptation/evaltest/d4_evaltest_outputs_Empathy.tsv',
    }
    d4_dev_parts_source_dir = './data/d4_dev_parts'
    d4_test_parts_source_dir = './data/d4_test_parts'


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


def join(source_dir, dest_file, read_size):
    # Reference: https://stonesoupprogramming.com/2017/09/16/python-split-and-join-file/
    output_file = open(dest_file, 'wb')
    parts = os.listdir(source_dir)
    parts.sort()
    for file in parts:
        path = os.path.join(source_dir, file)
        input_file = open(path, 'rb')
        while True:
            bytes = input_file.read(read_size)
            if not bytes:
                break
            output_file.write(bytes)
        input_file.close()
    output_file.close()


if __name__ == '__main__':
    utils.set_seed()

    utils.setup_logging(log_filename='log_d4_adaptation_inference')
    logging.info('**** starting d4 adaptation inference logging ****')

    join_parts = [(conf.adaptation_dev_data, conf.d4_dev_parts_source_dir), (conf.adaptation_test_data, conf.d4_test_parts_source_dir)]
    for dest_file, parts_dir in join_parts:
        if not Path(dest_file).exists():
            # join the parts of the dev_dataset
            logging.info(f'joining: {join_parts} -> {dest_file}')
            split_size = 20 * 1024 * 1024  # 20MB
            join(source_dir=parts_dir, dest_file=dest_file, read_size=split_size)

    embedding_vector_length = 1584  # includes 1536 original features and 48 additional lexical features

    emb_current_conv_headers = [f'e_curr_conv_{i}' for i in range(embedding_vector_length)]
    emb_prev_conv_headers = [f'e_prev_conv_{i}' for i in range(embedding_vector_length)]
    emb_essay_headers = [f'e_essay_{i}' for i in range(embedding_vector_length)]

    emb_columns = emb_current_conv_headers + emb_prev_conv_headers + emb_essay_headers
    embedding_vector_length = len(emb_columns)  # the final length is a concatenation of three embeddings

    # process dev data and associated outputs
    dev_modeling_df = pd.read_csv(conf.adaptation_dev_data, sep='\t')

    logging.info('Dev DataFrame')
    logging.info(f'\n{dev_modeling_df}')

    x_dev = torch.tensor(dev_modeling_df[emb_columns].values, dtype=torch.float32)

    for target_feature in ['EmotionalPolarity', 'Emotion', 'Empathy']:
        logging.info(f'Loading model: {conf.adaptation_models[target_feature]}')
        model_nn = torch.load(conf.adaptation_models[target_feature])
        model_nn.eval()

        y_dev_pred = model_nn(x_dev)
        y_dev_pred = torch.squeeze(y_dev_pred)
        y_dev_pred = y_dev_pred.tolist()

        output_file = conf.adaptation_devtest_outputs[target_feature]
        outputs_dir = Path(output_file).parent
        if not outputs_dir.exists():
            outputs_dir.mkdir(parents=True)

        dev_pred_df = pd.DataFrame(y_dev_pred)
        dev_pred_df.to_csv(output_file, sep='\t', index=False, header=[f'{target_feature}_Predictions'])
        logging.info(f'Saving output: {output_file}')

    # process dev data and associated outputs
    test_modeling_df = pd.read_csv(conf.adaptation_test_data, sep='\t')

    logging.info('Test DataFrame')
    logging.info(f'\n{test_modeling_df}')

    x_test = torch.tensor(test_modeling_df[emb_columns].values, dtype=torch.float32)

    for target_feature in ['EmotionalPolarity', 'Emotion', 'Empathy']:
        logging.info(f'Loading model: {conf.adaptation_models[target_feature]}')
        model_nn = torch.load(conf.adaptation_models[target_feature])
        model_nn.eval()

        y_test_pred = model_nn(x_test)
        y_test_pred = torch.squeeze(y_test_pred)
        y_test_pred = y_test_pred.tolist()

        output_file = conf.adaptation_evaltest_outputs[target_feature]
        outputs_dir = Path(output_file).parent
        if not outputs_dir.exists():
            outputs_dir.mkdir(parents=True, exist_ok=True)

        test_pred_df = pd.DataFrame(y_test_pred)
        test_pred_df.to_csv(output_file, sep='\t', index=False, header=[f'{target_feature}_Predictions'])
        logging.info(f'Saving output: {output_file}')

    logging.info('**** finish d4 adaptation inference logging ****')
