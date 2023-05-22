# Run the evaluation for the D4 adaptation task

# import standard libraries
import pandas as pd
import numpy as np
import scipy.stats
from pathlib import Path
import logging

# import project modules
# import conf
import utils


class conf:
    adaptation_dev_data = './data/d4_dev_modeling_df.pkl.bz2'
    adaptation_devtest_outputs = {
        'EmotionalPolarity': './outputs/D4/adaptation/devtest/d4_devtest_outputs_EmotionalPolarity.tsv',
        'Emotion': './outputs/D4/adaptation/devtest/d4_devtest_outputs_Emotion.tsv',
        'Empathy': './outputs/D4/adaptation/devtest/d4_devtest_outputs_Empathy.tsv',
    }
    adaptation_evaltest_outputs = {
        'EmotionalPolarity': './outputs/D4/adaptation/evaltest/d4_evaltest_outputs_EmotionalPolarity.tsv',
        'Emotion': './outputs/D4/adaptation/evaltest/d4_evaltest_outputs_Emotion.tsv',
        'Empathy': './outputs/D4/adaptation/evaltest/d4_evaltest_outputs_Empathy.tsv',
    }
    results_d4_adaptation_path = './results/D4/adaptation/devtest/D4_scores.out'


def pearson_corr(v1, v2):
    r = scipy.stats.pearsonr(v1, v2)
    return r[0]


if __name__ == '__main__':
    utils.setup_logging(log_filename='log_d4_adaptation_eval')
    logging.info('**** starting evaluation logging ****')

    # process dev data and associated outputs
    dev_df = pd.read_pickle(conf.adaptation_dev_data)
    devtest_scores_sum = 0.0
    devtest_scores_text = ''
    scores_text_map = {
        'Empathy': 'Conv Empathy Pearson Correlation: ',
        'EmotionalPolarity': 'Conv Emotional Polarity Pearson Correlation: ',
        'Emotion': 'Conv Emotional Intensity Pearson Correlation: '
    }

    for target_feature in ['Empathy', 'EmotionalPolarity', 'Emotion']:
        logging.info(f'Processing Dev: {target_feature}')

        y_dev_gold = dev_df[target_feature].tolist()

        output_file = conf.adaptation_devtest_outputs[target_feature]
        dev_pred_df = pd.read_csv(output_file, sep='\t')
        y_dev_pred = dev_pred_df[f'{target_feature}_Predictions'].tolist()

        pearson_for_target = pearson_corr(y_dev_pred, y_dev_gold)
        logging.info(f'Pearson correlation for {target_feature}: {pearson_for_target}')
        devtest_scores_sum += pearson_for_target
        devtest_scores_text += f'{scores_text_map[target_feature]}{pearson_for_target}\n'

    devtest_mean_score = devtest_scores_sum / 3
    devtest_scores_text = f'Conv Pearson Correlations: {devtest_mean_score}\n' + devtest_scores_text

    adaptation_devtest_filepath = conf.results_d4_adaptation_path
    outputs_dir = Path(adaptation_devtest_filepath).parent
    if not outputs_dir.exists():
        outputs_dir.mkdir(parents=True)
    Path(adaptation_devtest_filepath).write_text(devtest_scores_text)
