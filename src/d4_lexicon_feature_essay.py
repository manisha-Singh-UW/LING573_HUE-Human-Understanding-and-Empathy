# Build lexicon features for train and dev sets

# import standard libraries
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder
import ast
import pickle

# import project modules
import conf
import utils


#Load paths
nrc_emotion_path = conf.nrc_emotion_path
nrc_vad_path = conf.nrc_vad_path
mpqa_path = conf.mpqa_path
polarity_path = conf.polarity_path
dataset_path = conf.dataset_path
rescale_param_path = conf.rescale_param_path
train_data_path = conf.train_data_path_d4
dev_data_path = conf.dev_data_path_d4
test_data_path = conf.test_data_path_d4

def lemmatize_essay(dataset):
    lemmatized_essays = []
    lemmatizer = WordNetLemmatizer()

    for essay in dataset['essay']:
        lemma_sent = []
        tokenized = word_tokenize(essay)
        for token in tokenized:
            lemma_sent.append(lemmatizer.lemmatize(token))
        lemmatized_essays.append(lemma_sent)
        
    return lemmatized_essays

def load_lexicon(lexicon_path, lexicon_type):
    '''lexicon_type: nrc_emotion / nrc_vad / mpqa / polarity
        '''
    
    lexicon_dict = defaultdict(dict)
    
    #Load NRC emotion
    emotion_dict = defaultdict(dict)
    if lexicon_type == 'nrc_emotion':
        with open(lexicon_path, 'r') as f:
            for line in f:
                values = line.split()
                emotion_dict[values[0]][values[1]] = int(values[2])
                
        for k, v in emotion_dict.items():
            lexicon_dict[k] = np.array(list(emotion_dict[k].values()))
    
    #Load NRC VAD
    elif lexicon_type == 'nrc_vad':
        with open(lexicon_path, 'r') as f:
            for line in f:
                values_lst = line.strip().split()
                initial_index = len(values_lst) - 3
                lexicon_dict[values_lst[0]] = np.array([float(i) for i in values_lst[initial_index:]])
                
    #Load MPQA
    elif lexicon_type == 'mpqa':
        #Features to be extracted from lexicon
        extract_feat = ['type','pos1','priorpolarity']
        tmp_dict = defaultdict(dict)
        with open(lexicon_path, 'r') as f:
            for line in f:
                tmp_dict = {i.split('=')[0]:i.split('=')[1] for i in line.strip().split() if '=' in i}
                feat_lst = [tmp_dict[feat] for feat in extract_feat]
                lexicon_dict[tmp_dict['word1']] = feat_lst
                
            #One-hot encode
            prefixes = ['type','pos','valence']
            valence_df = pd.DataFrame.from_dict(lexicon_dict, orient = 'index', columns = prefixes)
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(valence_df)
            encoded_valence = encoder.transform(valence_df)
            
            lexicon_dict = {i[0]: i[1] for i in zip(valence_df.index, encoded_valence.toarray())}
            
    #Load polarity shifter
    elif lexicon_type == 'polarity':
        with open(polarity_path, 'r') as f:
            f = f.readlines()[1:]
            for line in f:
                values_lst = line.strip().split(',')
                if values_lst[1] == 'shifter':
                    lexicon_dict[values_lst[0]] = 1.0
                else:
                    lexicon_dict[values_lst[0]] = 0.0
                    
    return lexicon_dict

def count_feat(lexicon_dict, essay_list, feat_num):
    #Count number of emotions by essay
    feat_lst = []
    for e in essay_list:
        feat_cnt = np.zeros(feat_num)
        len_cnt = 0
        for t in e:
            if t in lexicon_dict:
                feat_cnt = feat_cnt + lexicon_dict[t]
                len_cnt += 1
        feat_lst.append([len_cnt, feat_cnt])
    
    return feat_lst

def build_feat(feat_lst, feat_type):
    '''feat_type: count / mean'''
    if feat_type == 'count':
        features = np.empty((0,len(feat_lst[0][1])))
        for i in feat_lst:
            features = np.vstack([features, i[1]])
            
    elif feat_type == 'mean':
        features = np.empty((0,len(feat_lst[0][1])))
        for i in feat_lst:
            if i[0] > 0:
                features = np.vstack([features, i[1]/i[0]])
            else:
                features = np.vstack([features, i[1]])
                
    return features

def standardize(features, m2, s2):
    m1 = np.mean(features, axis = 0)
    s1 = np.std(features, axis = 0)
    s_features = m2 + (features - m1) * s2/s1
    return np.nan_to_num(s_features)

def create_lex_embedding(dataset_path, emb_model, features, dataset_type, rescale_param_path):
    
    emb_df = pd.read_csv(dataset_path + 'd4_' + dataset_type + '-' + emb_model, sep = '\t')
    emb_len = len(np.array(ast.literal_eval(emb_df['string_embedding'][0]), dtype=float))
    emb_matrix = np.empty((0,emb_len))
    
    #Get mean and std from embeddings to rescale
    for i in emb_df['string_embedding']:
        arr = np.array(ast.literal_eval(i), dtype=float)
        emb_matrix = np.vstack([emb_matrix,arr])
     
    if dataset_type == 'train':
        parameter_dict = {}
        m2 = np.mean(np.mean(emb_matrix, axis = 0))
        s2 = np.mean(np.std(emb_matrix, axis = 0))
        parameter_dict['m2'] = m2
        parameter_dict['s2'] = s2
        
        #Save rescaling parameters 
        with open(rescale_param_path + '.pkl', 'wb') as fp:
            pickle.dump(parameter_dict, fp)
    
    else:
        
        #Load rescaling parameters
        with open(rescale_param_path + '.pkl', 'rb') as fp:
            parameter_dict = pickle.load(fp)
            
        m2 = parameter_dict['m2']
        s2 = parameter_dict['s2']

    
    ## Rescale features
    
    #Rescale emotion count
    rs_emotion_count = standardize(features[0], m2, s2)
    #Rescale emotion mean
    rs_emotion_mean = standardize(features[1], m2, s2)
    #Rescale mpqa count
    rs_mpqa_count = standardize(features[2], m2, s2)
    #Rescale mpqa mean
    rs_mpqa_mean = standardize(features[3], m2, s2)
    #Rescale vad mean
    rs_vad_mean = standardize(features[4], m2, s2)
    #Rescale polarity count
    rs_pol_count = standardize(features[5], m2, s2)
    
    ## Append rescaled features to matrix
    
    emb_matrix = np.append(emb_matrix, rs_emotion_count, axis = 1)
    emb_matrix = np.append(emb_matrix, rs_emotion_mean, axis = 1)
    emb_matrix = np.append(emb_matrix, rs_mpqa_count, axis = 1)
    emb_matrix = np.append(emb_matrix, rs_mpqa_mean, axis = 1)
    emb_matrix = np.append(emb_matrix, rs_vad_mean, axis = 1)
    emb_matrix = np.append(emb_matrix, rs_pol_count, axis = 1)
    
    #Create new embedding list
    emb_lst = emb_matrix.tolist()
    new_emb_lst = [str(i) for i in emb_lst]
    
    emb_df.drop(columns = 'string_embedding', inplace = True)
    emb_df['string_embedding'] = new_emb_lst
    emb_df.to_csv(dataset_path + dataset_type + '-' + emb_model.split('.')[0] + '_lexicon_adapt.tsv',
                  sep = '\t', index = False)


if __name__ == '__main__':

    emb_model = ['essays_with_embedding_text-embedding-ada-002.tsv']

    for e in emb_model:
        
        #############################################################
        #Train
        #############################################################
        
        train_df = pd.read_csv(train_data_path, delimiter = '\t')
        lemmatized_essays = lemmatize_essay(train_df)

        #Load lexicon dicts
        emotion_dict = load_lexicon(nrc_emotion_path, 'nrc_emotion')
        vad_dict = load_lexicon(nrc_vad_path, 'nrc_vad')
        mpqa_dict = load_lexicon(mpqa_path, 'mpqa')
        polarity_dict = load_lexicon(polarity_path, 'polarity')

        #Create lexicon count lists
        emotion_lst = count_feat(emotion_dict, lemmatized_essays, 10)
        vad_lst = count_feat(vad_dict, lemmatized_essays, 3)
        mpqa_lst = count_feat(mpqa_dict, lemmatized_essays, 12)
        polarity_lst = count_feat(polarity_dict, lemmatized_essays, 1)

        #Build features
        emotion_count = build_feat(emotion_lst, 'count')
        emotion_mean = build_feat(emotion_lst, 'mean')
        mpqa_count = build_feat(mpqa_lst, 'count')
        mpqa_mean = build_feat(mpqa_lst, 'mean')
        vad_mean = build_feat(vad_lst, 'mean')
        polarity_count = build_feat(polarity_lst, 'count')

        features = [emotion_count, emotion_mean, mpqa_count, mpqa_mean, vad_mean, polarity_count]
        dataset_type = 'train'
        create_lex_embedding(dataset_path, e, features, dataset_type, rescale_param_path)
        
        #############################################################
        #Dev
        #############################################################
        
        dev_df = pd.read_csv(dev_data_path, delimiter = '\t')
        lemmatized_essays = lemmatize_essay(dev_df)

        #Create lexicon count lists
        emotion_lst = count_feat(emotion_dict, lemmatized_essays, 10)
        vad_lst = count_feat(vad_dict, lemmatized_essays, 3)
        mpqa_lst = count_feat(mpqa_dict, lemmatized_essays, 12)
        polarity_lst = count_feat(polarity_dict, lemmatized_essays, 1)

        #Build features
        emotion_count = build_feat(emotion_lst, 'count')
        emotion_mean = build_feat(emotion_lst, 'mean')
        mpqa_count = build_feat(mpqa_lst, 'count')
        mpqa_mean = build_feat(mpqa_lst, 'mean')
        vad_mean = build_feat(vad_lst, 'mean')
        polarity_count = build_feat(polarity_lst, 'count')

        features = [emotion_count, emotion_mean, mpqa_count, mpqa_mean, vad_mean, polarity_count]
        dataset_type = 'dev'
        create_lex_embedding(dataset_path, e, features, dataset_type, rescale_param_path)


        #############################################################
        #Test
        #############################################################
        
        test_df = pd.read_csv(test_data_path, delimiter = '\t')
        lemmatized_essays = lemmatize_essay(test_df)

        #Create lexicon count lists
        emotion_lst = count_feat(emotion_dict, lemmatized_essays, 10)
        vad_lst = count_feat(vad_dict, lemmatized_essays, 3)
        mpqa_lst = count_feat(mpqa_dict, lemmatized_essays, 12)
        polarity_lst = count_feat(polarity_dict, lemmatized_essays, 1)

        #Build features
        emotion_count = build_feat(emotion_lst, 'count')
        emotion_mean = build_feat(emotion_lst, 'mean')
        mpqa_count = build_feat(mpqa_lst, 'count')
        mpqa_mean = build_feat(mpqa_lst, 'mean')
        vad_mean = build_feat(vad_lst, 'mean')
        polarity_count = build_feat(polarity_lst, 'count')

        features = [emotion_count, emotion_mean, mpqa_count, mpqa_mean, vad_mean, polarity_count]
        dataset_type = 'test'
        create_lex_embedding(dataset_path, e, features, dataset_type, rescale_param_path)