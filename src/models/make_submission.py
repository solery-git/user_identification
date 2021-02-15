# -*- coding: utf-8 -*-
import click
from src.utils.click import PathlibPath
from pathlib import Path
import logging
import src.utils.joblib as ujob

import yaml
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR.joinpath('data/processed/')
DATASET_PROCESSED_DIR = PROCESSED_DATA_DIR.joinpath('dataset_processed')
MODELS_DIR = PROJECT_DIR.joinpath('models')
SUBMISSIONS_DIR = PROJECT_DIR.joinpath('submissions')


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


def main(output_path=SUBMISSIONS_DIR):
    output_path.mkdir(parents=True, exist_ok=True)
    X_submission = ujob.load(DATASET_PROCESSED_DIR, 'X_submission.joblib')
    model = ujob.load(MODELS_DIR, 'model.joblib')
    with open(MODELS_DIR.joinpath('metrics.yaml'), 'r') as fin:
        metrics = yaml.safe_load(fin)
    
    y_subm_pred = model.predict_proba(X_submission)[:, 1]
    
    submission_fname = f'subm_{metrics["score_mean"]:.5}_{metrics["score_std"]:.5}.csv'
    write_to_submission_file(y_subm_pred, output_path.joinpath(submission_fname))
    print(f'Saved as {submission_fname}')
    
    
@click.command()
@click.option('-o', 'output_path', 
              type=PathlibPath(exists=True, file_okay=False), default=SUBMISSIONS_DIR, 
              help='Output directory for the submission (default: <project_dir>/submissions)')
def cli(output_path):
    """ Evaluate the model on submission data and make a kaggle submission.
    """
    logger = logging.getLogger(__name__)
    logger.info('making a submission')
    
    main(output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()