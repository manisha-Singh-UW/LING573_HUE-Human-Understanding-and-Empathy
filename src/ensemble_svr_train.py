import ensemble_config as ec
import ensemble_utils as eu

from sklearn.svm import SVR
import pickle

def set_sample_weights(true_labels):
    """Return list of weights for a training set given the iterable true_labels
    of gold values. Higher weight is given to samples with true labels farther
    from the midpoint of the scale."""
    max_y = max(true_labels)
    min_y = min(true_labels)
    middle = (max_y + min_y) / 2
    weights = [abs(middle - x) + 1 for x in true_labels]
    
    return weights

def train_svr_model(X_train, y_train, kernel, weights):
    """Return trained SVR model."""
    model = SVR(kernel=kernel)
    model.fit(X_train, y_train, sample_weight=weights)
    return model

def store_svr_model(model, embeddings, emotion, kernel):
    """Pickle SVR model."""
    filepath = eu.get_model_file('svr', embeddings, emotion, kernel=kernel)
    with open(filepath, 'wb') as writer:
       pickle.dump(model, writer)

if __name__ == '__main__':
    for empathy in (True, False):
        emotion = 'empathy' if empathy == True else 'distress'
        y_train = eu.load_gold_labels(eu.get_train_data_file('ada'), empathy)
        weights = set_sample_weights(y_train)
        
        for embeddings in ec.use_embeddings:
            if ec.use_embeddings[embeddings] == True:
                #get training data
                X_train = eu.load_embeddings(eu.get_train_data_file(embeddings))
                
                for kernel in ec.use_kernels:
                    if ec.use_kernels[kernel] == True:
                        #train and store SVR
                        svr_model = train_svr_model(X_train, y_train, kernel, weights)
                        store_svr_model(svr_model, embeddings, emotion, kernel)