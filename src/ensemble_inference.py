from ensemble_regressor import EnsembleRegressor
from nnpredictor_wrapper import NNPredictor
from d3_inference_empathy import NeuralNetwork, AdvancedDropout
import ensemble_config as ec
import ensemble_utils as eu

from sklearn.svm import SVR
from torch import nn
import warnings
import sys
import pickle

def load_svr_model(embeddings, emotion, kernel):
    """Unpickle and return trained SVR model."""
    filepath = eu.get_model_file('svr', embeddings, emotion, kernel=kernel)
    with open(filepath, 'rb') as reader:
        model = pickle.load(reader)
    return model

def load_nn_model(embeddings, emotion):
    """Return trained neural model wrapped in NNPredictor."""
    filepath = eu.get_model_file('nn', embeddings, emotion)
    return NNPredictor(filepath)

def prepare_models(empathy, test_set):
    """Return list of trained models, list of model names, list of test data 
    embedding-sets, and dictionary with models as keys and index of 
    corresponding test data embeddings as values."""
        
    models = [] #models for ensemble
    names = [] #display names for models
    X_tests = [] #test features used by models
    model2test = dict() #key: model, value: index of corresponding test features in X_tests
    
    emotion = 'empathy' if empathy == True else 'distress'
        
    for embeddings in ec.use_embeddings:
        if ec.use_embeddings[embeddings] == True:
            added_models = []
            name_prefix = embeddings+' lexicon'
                    
            #load SVR models
            if ec.use_models['sklearn'] == True:
                for kernel in ec.use_kernels:
                    if ec.use_kernels[kernel] == True:
                        svr_model = load_svr_model(embeddings, emotion, kernel)
                        added_models.append(svr_model)
                        svr_name = name_prefix+' SVR '+kernel
                        names.append(svr_name)
                    
            #load neural network
            if ec.use_models['NN'] == True:
                nn_model = load_nn_model(embeddings, emotion)
                added_models.append(nn_model)
                nn_name = name_prefix+' NN'
                names.append(nn_name)
            
            #load test embeddings and update model2test                
            if len(added_models) > 0:
                models += added_models
                X_test = eu.load_embeddings(eu.get_test_data_file(embeddings, test_set))
                X_tests.append(X_test)
                for model in added_models:
                    model2test[model] = len(X_tests) - 1
                            
    return models, names, X_tests, model2test

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    emotion = sys.argv[1] #e.g. 'empathy'
    outfile_prefix = sys.argv[2] #e.g. './outputs/ensemble'
    test_set = sys.argv[3] #'dev' (devtest) or 'test' (evaltest)
    
    empathy = True if emotion == 'empathy' else False
    outfile_prefix = f'{outfile_prefix}_{"empathy" if empathy == True else "distress"}_' 
    
    #output file for each model's predictions on the test set
    model_prediction_writer = open(outfile_prefix+'predictions.txt', 'w', encoding='utf-8')
    
    #output file for the ensemble's predictions on the test set
    ensemble_prediction_writer = open(outfile_prefix+'prediction.txt', 'w', encoding='utf-8')
    
    #initialize ensemble
    models, names, test_X, model2X = prepare_models(empathy, test_set)
    ensemble = EnsembleRegressor(models, names=names)
    with open(outfile_prefix+'votes.txt', 'w') as writer:
        writer.write(f'{ensemble}') 
    
    #ensemble prediction on test set
    prediction = ensemble.predict(test_X, model2X)
    model_prediction_writer.write(f'ensemble\t{list(prediction)}\n')
    for x in prediction:
        ensemble_prediction_writer.write(f'{x}\n')
    
    #model predictions on test set
    for i in range(len(ensemble.models)):
        model = ensemble.models[i]
        name = ensemble.names[i]
        prediction = model.predict(test_X[model2X[model]])
        model_prediction_writer.write(f'{name}\t{list(prediction)}\n')
        
    model_prediction_writer.close()
    ensemble_prediction_writer.close()