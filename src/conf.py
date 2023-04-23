# This file stores the configuration data for the project
# Examples of conf data include:
# - path to files
# - path to directories
# - variables used across multiple files in the project


train_data_path = './data/messages_train_ready_for_WS.tsv'
dev_data_path = './data/messages_dev_features_ready_for_WS_2022.tsv'
dev_data_goldstandard_path = './data/goldstandard_dev_2022.tsv'
test_data_path = ''

# train_data_with_embedding = './data/train_data_with_embedding_all-MiniLM-L6-v2.tsv'
# dev_data_with_embedding = './data/dev_data_with_embedding_all-MiniLM-L6-v2.tsv'
train_data_with_embedding = './data/train_data_with_embedding_text-embedding-ada-002.tsv'
dev_data_with_embedding = './data/dev_data_with_embedding_text-embedding-ada-002.tsv'

# model_nn_empathy_save_path = './models/model_nn_empathy_all-MiniLM-L6-v2.pth'
# model_nn_distress_save_path = './models/model_nn_distress_all-MiniLM-L6-v2.pth'
model_nn_empathy_save_path = './models/model_nn_empathy_text-embedding-ada-002_exp2.pth'
model_nn_distress_save_path = './models/model_nn_distress_text-embedding-ada-002_exp2.pth'

outputs_d2_empathy_path = './outputs/D2/d2_outputs_empathy.tsv'
outputs_d2_distress_path = './outputs/D2/d2_outputs_distress.tsv'
scores_d2_path = './results/D2_scores.out'
