# -*- coding: utf-8 -*-
import click
from src.utils.click import PathlibPath
from pathlib import Path
import logging
import src.utils.joblib as ujob
import src.utils.pickle as upkl

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR.joinpath('data/raw/')
PROCESSED_DATA_DIR = PROJECT_DIR.joinpath('data/processed/')
DATASET_DIR = PROCESSED_DATA_DIR.joinpath('dataset')
DATASET_PROCESSED_DIR = PROCESSED_DATA_DIR.joinpath('dataset_processed')


def main(input_path=DATASET_DIR, output_path=DATASET_PROCESSED_DIR):
    output_path.mkdir(parents=True, exist_ok=True)
    site_columns = ['site%d' % i for i in range(1, 10+1)]
    time_columns = ['time%d' % i for i in range(1, 10+1)]
    
    X, X_submission, y = ujob.load_multiple(input_path, ['X.joblib', 'X_submission.joblib', 'y.joblib'])
    site_dic = upkl.load(RAW_DATA_DIR, 'site_dic.pkl')
    
    X[site_columns] = X[site_columns].fillna(0).astype(int)
    X_submission[site_columns] = X_submission[site_columns].fillna(0).astype(int)
    
    # Decode site ids into names
    site_id_dic = {v:k for (k, v) in site_dic.items()}
    site_id_dic[0] = 'unknown'
    
    X[site_columns] = X[site_columns].applymap(site_id_dic.get)
    X_submission[site_columns] = X_submission[site_columns].applymap(site_id_dic.get)
    
    ujob.dump_multiple([X, X_submission, y], 
                       output_path, 
                       ['X.joblib', 'X_submission.joblib', 'y.joblib'])


@click.command()
@click.option('-i', 'input_path', 
              type=PathlibPath(exists=True, file_okay=False), default=DATASET_DIR, 
              help='Input directory for the dataset (default: <project_dir>/data/processed/dataset)')
@click.option('-o', 'output_path', 
              type=PathlibPath(exists=True, file_okay=False), default=DATASET_PROCESSED_DIR, 
              help='Output directory for the processed dataset (default: <project_dir>/data/processed/dataset_processed)')
def cli(input_path, output_path):
    """ Preprocesses the dataset into the format suitable for model pipeline input / inference.
    """
    logger = logging.getLogger(__name__)
    logger.info('preprocessing a dataset')
    
    main(input_path, output_path)
        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()