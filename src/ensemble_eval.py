from ensemble_utils import load_gold_labels
import ensemble_utils as eu

from scipy import stats
from pathlib import Path
import sys

def score(gold, predicted):
    """Return Pearson correlation of gold and predicted vectors."""
    return stats.pearsonr(gold, predicted)[0]

def read_prediction(filepath):
    """Return ensemble prediction read from file."""
    with open(filepath, 'r', encoding='utf-8') as reader:
        lines = reader.readlines() 
    return [float(x) for x in lines]

def score_ensemble(outputs_prefix, results_prefix, empathy_gold, distress_gold):
    """Calculate empathy, distress and mean Pearson Correlations with gold 
    labels. Write these scores to file."""
    empathy_prediction_file = f'{outputs_prefix}_empathy_prediction.txt'
    distress_prediction_file = f'{outputs_prefix}_distress_prediction.txt'
    output_file = f'{results_prefix}_scores.out'
    
    empathy_prediction = read_prediction(empathy_prediction_file)
    distress_prediction = read_prediction(distress_prediction_file)
    
    empathy_score = score(empathy_gold, empathy_prediction)
    distress_score = score(distress_gold, distress_prediction)
    mean_score = (empathy_score + distress_score) / 2
    
    #make folders for results file if they don't exist
    outputs_dir = Path(output_file).parent
    if not outputs_dir.exists():
        outputs_dir.mkdir(parents=True)
    
    with open(output_file, 'w', encoding='utf-8') as writer:
        writer.write(f'Pearson correlation for Empathy: {empathy_score}\n')
        writer.write(f'Pearson correlation for Distress: {distress_score}\n')
        writer.write(f'Mean Pearson correlation: {mean_score}')
        
def get_model_predictions(filepath):
    """Return dictionary with model names as keys and model predictions as values."""
    with open(filepath, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
    model2prediction = dict()
    
    for line in lines:
        pair = line.split('\t')
        model_name = pair[0]
        
        prediction = pair[1]
        prediction = prediction.strip()[1:-1]
        prediction = [float(x) for x in prediction.split(',')]
        
        model2prediction[model_name] = prediction
    
    return model2prediction
        
def score_models(outputs_prefix, empathy_gold, distress_gold):
    """Calculate Pearson Correlations of models' predictions with gold labels.
    Write these scores to file."""
    empathy_predictions_file = f'{outputs_prefix}_empathy_predictions.txt'
    emp_model2pred = get_model_predictions(empathy_predictions_file)
    emp_outfile = f'{outputs_prefix}_empathy_scores.txt'
    with open(emp_outfile, 'w', encoding='utf-8') as writer:
        for model in emp_model2pred:
            model_pred = emp_model2pred[model]
            model_score = score(empathy_gold, model_pred)
            writer.write(f'{model}\t{model_score}\n')
    
    distress_predictions_file = f'{outputs_prefix}_distress_predictions.txt'
    dis_model2pred = get_model_predictions(distress_predictions_file)
    dis_outfile = f'{outputs_prefix}_distress_scores.txt'
    with open(dis_outfile, 'w', encoding='utf-8') as writer:
        for model in dis_model2pred:
            model_pred = dis_model2pred[model]
            model_score = score(distress_gold, model_pred)
            writer.write(f'{model}\t{model_score}\n')

if __name__ == '__main__':
    outputs_prefix = sys.argv[1] #e.g. './outputs/ensemble'
    results_prefix = sys.argv[2] #e.g. './results/ensemble'
    
    #get gold labels
    test_gold_file = eu.get_test_data_file('ada')
    empathy_gold = load_gold_labels(test_gold_file, empathy=True)
    distress_gold = load_gold_labels(test_gold_file, empathy=False)
    
    #calculate and write to file Pearson Correlation scores
    score_ensemble(outputs_prefix, results_prefix, empathy_gold, distress_gold)
    score_models(outputs_prefix, empathy_gold, distress_gold)