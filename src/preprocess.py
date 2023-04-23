# File to load data, perform preprocessing tasks, and geneate outputs that can be fed into the model

# import standard libraries
import logging
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# import project modules
import conf
import utils


def load_dataset():
    train_columns = ['essay', 'empathy', 'distress']

    # train_data = pd.read_csv(conf.train_data_path, sep='\t')
    train_df = pd.read_csv(conf.train_data_path, sep='\t', usecols=train_columns)

    dev1_df = pd.read_csv(conf.dev_data_path, sep='\t', usecols=['essay'])
    dev2_df = pd.read_csv(conf.dev_data_goldstandard_path, sep='\t', header=None)
    dev_df = dev1_df.assign(empathy=dev2_df[0], distress=dev2_df[1])

    # test_data = pd.read_csv(conf.test_data_path, sep='\t')

    # return train_data, dev_data, test_data
    return train_df, dev_df


def initialize_embedding_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    return model


def create_embeddings(model, str_list: list = ['']) -> list:
    # input: a list of strings
    # output: embeddings corresponding to each of the strings

    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

    embedding_list = model.encode(str_list)

    return embedding_list


if __name__ == '__main__':
    utils.set_seed()

    utils.setup_logging(log_filename='log_preprocess')
    logging.info('**** starting preprocessing logging ****')

    train_df, dev_df = load_dataset()

    model = None

    # create embeddings for train data    
    if Path(conf.train_data_with_embedding).exists():
        logging.info('Embedding file already exists for Training data')
    else:
        logging.info('Creating embeddings for train data')
        train_df = train_df.assign(essay_emb='')
        model = initialize_embedding_model()

        for row in range(len(train_df)):
            essay_str = train_df.loc[row, 'essay']
            emb = create_embeddings(model, [essay_str])[0]
            emb_list = np.array(emb).tolist()
            train_df.at[row, 'essay_emb'] = emb_list
            logging.info(f'row={row}; essay={essay_str[:20]}...; embedding={emb_list[:2]}...')

        train_df.to_csv(conf.train_data_with_embedding, sep='\t', index=False)
    # embedding for training data complete

    # create embeddings for dev data
    if Path(conf.dev_data_with_embedding).exists():
        logging.info('Embedding file already exists for Dev data')
    else:
        logging.info('Creating embeddings for dev data')
        dev_df = dev_df.assign(essay_emb = '')

        if model is None:
            model = initialize_embedding_model()

        for row in range(len(dev_df)):
            essay_str = dev_df.loc[row, 'essay']
            emb = create_embeddings(model, [essay_str])[0]
            emb_list = np.array(emb).tolist()
            dev_df.at[row, 'essay_emb'] = emb_list
            logging.info(f'row={row}; essay={essay_str[:20]}...; embedding={emb_list[:2]}...')

        dev_df.to_csv(conf.dev_data_with_embedding, sep='\t', index=False)
    # embedding for dev data complete

    logging.info('Preprocessing complete')
