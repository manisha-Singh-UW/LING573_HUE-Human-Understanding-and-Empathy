import ensemble_config as ec

import csv
import numpy as np

def load_embeddings(filepath): 
    """Return numpy array of essay embeddings from file at filepath."""
    with open(filepath, 'r', encoding='utf-8') as tsv_file:
        tsv_reader = csv.DictReader(tsv_file, delimiter='\t')
        embed = [row['essay_emb'] for row in tsv_reader]
        
    embed = [x[1:-2].split(', ') for x in embed]
    embed = [[float(y) for y in x] for x in embed]
    embed = np.array(embed)
    
    return embed

def load_gold_labels(filepath, empathy=True): 
    """Return numpy array of empathy or distress gold labels from file at 
    filepath."""
    emotion = 'empathy' if empathy == True else 'distress'
    
    with open(filepath, 'r', encoding='utf-8') as tsv_file:
        tsv_reader = csv.DictReader(tsv_file, delimiter='\t')
        gold_labels = [float(row[emotion]) for row in tsv_reader]
    
    return np.array(gold_labels)

def get_model_file(model_type, embedding, emotion, kernel=None):
    """Return filepath to stored model.
        model_type  'nn' or 'svr'
        embedding   'MiniLM', 'mpnet', 'roberta' or 'ada'
        emotion     'empathy' or 'distress'
        kernel      'linear', 'poly', 'rbf' or 'sigmoid' (specify for SVRs)"""
        
    model_str = f'{model_type}_{kernel}' if model_type == 'svr' else model_type
    embedding_str = ec.embedding2str[embedding]
    extension = 'pth' if model_type == 'nn' else 'pickle'
    
    return f'./models/model_{model_str}_{emotion}_{embedding_str}_advanced_lexicon.{extension}'

def get_train_data_file(embedding):
    """Return filepath for training data."""
    embedding_str = ec.embedding2str[embedding]
    return f'./data/train_data_with_embedding_{embedding_str}_lexicon.tsv'

def get_test_data_file(embedding, test_set='dev'):
    """Return filepath for test data."""
    embedding_str = ec.embedding2str[embedding]
    return f'./data/{test_set}_data_with_embedding_{embedding_str}_lexicon.tsv'