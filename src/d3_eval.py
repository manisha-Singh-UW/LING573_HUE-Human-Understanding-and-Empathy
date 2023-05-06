# Run the evaluation for the model on dev dataset

# import standard libraries
import pandas as pd
import numpy as np
import scipy.stats
from pathlib import Path
import logging

# import project modules
import conf
import utils


def pearson_corr(v1, v2):
    r = scipy.stats.pearsonr(v1, v2)
    return r[0]


if __name__ == '__main__':
    utils.setup_logging(log_filename='log_eval')
    logging.info('**** starting evaluation logging ****')

    dev_df = pd.read_csv(conf.dev_data_with_embedding, sep='\t')
    y_dev_empathy_gold = dev_df['empathy'].tolist()
    y_dev_distress_gold = dev_df['distress'].tolist()

    dev_empathy_pred_df = pd.read_csv(conf.outputs_d3_empathy_path, sep='\t')
    y_dev_empathy_pred = dev_empathy_pred_df['DevEmpathyPredictions'].tolist()

    dev_distress_pred_df = pd.read_csv(conf.outputs_d3_distress_path, sep='\t')
    y_dev_distress_pred = dev_distress_pred_df['DevDistressPredictions'].tolist()


    logging.info('Dev Empathy predictions')
    logging.info(y_dev_empathy_pred)
    logging.info('Dev Distress predictions')
    logging.info(y_dev_distress_pred)

    pearson_empathy = pearson_corr(y_dev_empathy_pred, y_dev_empathy_gold)
    logging.info(f'Pearson correlation for Empathy: {pearson_empathy}')

    pearson_distress = pearson_corr(y_dev_distress_pred, y_dev_distress_gold)
    logging.info(f'Pearson correlation for Distress: {pearson_distress}')

    logging.info(f'Mean Pearson correlation: {(pearson_empathy+pearson_distress)/2}')

    scores_text = f'Pearson correlation for Empathy: {pearson_empathy}'
    scores_text += f'\nPearson correlation for Distress: {pearson_distress}'
    scores_text += f'\nMean Pearson correlation: {(pearson_empathy+pearson_distress)/2}'
    Path(conf.scores_d3_path).write_text(scores_text)

    logging.info('**** finish evaluation logging ****')
