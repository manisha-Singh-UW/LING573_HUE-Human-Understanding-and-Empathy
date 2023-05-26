import ensemble_utils

import numpy as np
import heapq

class EnsembleRegressor:
    
    """Ensemble for regression tasks. Designed for ensembling models trained on
    different features."""
    
    def __init__(self, models, topk=None, X_devs=None, y_dev=None, model2X=None,
                 names=None):
        """
        Parameters:
            models  list of models that may participate in the ensemble
            names   list of names corresponding the models list (if names is 
                    None, the default string representation of each model will 
                    be used)
            topk    number of highest-performing models to be included in the 
                    ensemble (all will be included if topk is None)
            X_devs  validation datasets to calculate the top k highest-
                    performing models (only used if topk is not None)
            y_dev   validation gold labels to calculate the top k highest-
                    performing models (only used if topk is not None)
            model2X dictionary with models as keys and index of corresponding
                    validation dataset in X_devs as values (only used if topk 
                    is not None)
        """
        self.models = models
        self.names = names if names != None else models
        
        if topk == None:
            self.votes = self._initialize_votes() #ndarray shape: (len(self.models),)
        else:
            predictions = self._set_predictions(X_devs, model2X)
            self.votes = self._initialize_votes(topk, predictions, y_dev) #ndarray shape: (len(self.models),)
        
    def __str__(self):
        display = 'Ensemble:'
        for i in range(len(self.models)):
            display += f'\n{self.names[i]}\t {self.votes[i]}'
        return display
    
    def _set_predictions(self, Xs, model2X):
        """Return matrix of shape (len(self.models), len(Xs)) containing model 
        predictions for data in Xs. Each row is one model's predictions."""
        return np.vstack([x.predict(Xs[model2X[x]]) for x in self.models])
        
    def _score(self, gold, predicted):
        """Return Pearson correlation coefficient of gold and predicted vectors."""
        return ensemble_utils.score(gold, predicted)
    
    def _get_top_classifier_indices(self, k, predictions, true_values):
        """Return k highest-scoring models in self.models, based on
        their scores against the development gold labels (true_values)."""
        scores = [self._score(true_values, predictions[i]) for i in range(len(predictions))]
        return heapq.nlargest(k, range(len(scores)), key=lambda x: scores[x])
    
    def _initialize_votes(self, topk=None, predictions=None, true_values=None):
        """Return ndarray containing number of votes assigned to each model in 
        self.models. If topk is None, all models receive one vote (i.e. all 
        models are equally weighted). If topk is not None, only the topk 
        highest-scoring models will receive a vote in the ensemble, and other 
        models in self.models do not participate."""
        votes = np.zeros(shape=len(self.models), dtype=int)
        
        if topk == None:
            for i in range(len(self.models)):
                votes[i] = 1
                
        else:
            top_indices = self._get_top_classifier_indices(topk, predictions, 
                                                           true_values)
            for i in top_indices:
                votes[i] = 1
        
        return votes
            
    def predict(self, Xs, model2X):
        """Return (as a numpy array) ensemble prediction given datasets (Xs)
        and dictionary with this ensemble's models as keys and the index in Xs 
        of the dataset to use for each model as values."""
        model_predictions = self._set_predictions(Xs, model2X)
        
        denominator = np.full(len(Xs[0]), sum(self.votes))
        
        prediction = np.transpose(model_predictions) * self.votes
        prediction = np.sum(prediction, axis=1)
        prediction = prediction / denominator
        
        return prediction