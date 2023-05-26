import torch

class NNPredictor:
    
    """Wrapper class for neural network so that a predict function similar to 
    those in scikit-learn models can be used."""
    
    def __init__(self, model_filepath, name=None):
        self.model = torch.load(model_filepath)
        self.model.eval()
        
        self.name = str(self.model) if name == None else name
        
    def __str__(self):
        return self.name

    def predict(self, X):
        """Return prediction of model on matrix X."""
        X = torch.tensor(X, dtype=torch.float32)
        
        y_pred = self.model(X)
        y_pred = torch.squeeze(y_pred)
        y_pred = y_pred.detach().numpy()
        
        return y_pred