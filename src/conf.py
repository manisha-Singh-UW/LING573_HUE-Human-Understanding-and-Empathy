# This file stores the configuration data for the project
# Examples of conf data include:
# - path to files
# - path to directories
# - variables used across multiple files in the project

# Note: this file has been updated for D#3 to include lexical features, updated models


train_data_path = './data/messages_train_ready_for_WS.tsv'
dev_data_path = './data/messages_dev_features_ready_for_WS_2022.tsv'
dev_data_goldstandard_path = './data/goldstandard_dev_2022.tsv'
test_data_path = ''

# train_data_with_embedding = './data/train_data_with_embedding_text-embedding-ada-002.tsv'
# dev_data_with_embedding = './data/dev_data_with_embedding_text-embedding-ada-002.tsv'
train_data_with_embedding = './data/train_data_with_embedding_text-embedding-ada-002_lexicon.tsv'
dev_data_with_embedding = './data/dev_data_with_embedding_text-embedding-ada-002_lexicon.tsv'

# model_nn_empathy_save_path = './models/model_nn_empathy_all-MiniLM-L6-v2.pth'
# model_nn_distress_save_path = './models/model_nn_distress_all-MiniLM-L6-v2.pth'
# model_nn_empathy_save_path = './models/model_nn_empathy_text-embedding-ada-002_exp2.pth'
# model_nn_distress_save_path = './models/model_nn_distress_text-embedding-ada-002_exp2.pth'
model_nn_empathy_save_path = './models/model_nn_empathy_text-embedding-ada-002_advanced_lexicon.pth'
model_nn_distress_save_path = './models/model_nn_distress_text-embedding-ada-002_advanced_lexicon.pth'

#Lexicon paths 
nrc_emotion_path = "./data/lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
nrc_vad_path = "./data/lexicons/NRC-VAD-Lexicon.txt"
mpqa_path = "./data/lexicons/subjclueslen1-HLTEMNLP05.tff"
polarity_path = "./data/lexicons/polarity_shifters.csv"
dataset_path = './data/'
rescale_param_path = './data/rescale_param_dict'

# outputs_d2_empathy_path = './outputs/D2/d2_outputs_empathy.tsv'
# outputs_d2_distress_path = './outputs/D2/d2_outputs_distress.tsv'
# scores_d2_path = './results/D2_scores.out'
outputs_d3_empathy_path = './outputs/D3/d3_outputs_empathy.tsv'
outputs_d3_distress_path = './outputs/D3/d3_outputs_distress.tsv'
scores_d3_path = './results/D3_scores.out'
