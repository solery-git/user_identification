# -*- coding: utf-8 -*-
import click
from src.utils.click import PathlibPath
from pathlib import Path
import logging
import src.utils.joblib as ujob

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR.joinpath('data/raw/')
PROCESSED_DATA_DIR = PROJECT_DIR.joinpath('data/processed/')
DATASET_DIR = PROCESSED_DATA_DIR.joinpath('dataset')


def fix_incorrect_date_formats(df, columns_to_fix):
    for time_col in columns_to_fix:
        d = df[time_col]
        index_mask = (d.dt.day <= 12)
        d_fix = d[index_mask]
        d_fix = pd.to_datetime(d_fix.apply(str), format='%Y-%d-%m %H:%M:%S')
        df.loc[index_mask, time_col] = d_fix
    return df

def main(input_path=RAW_DATA_DIR, output_path=DATASET_DIR):
    output_path.mkdir(parents=True, exist_ok=True)
    site_columns = ['site%d' % i for i in range(1, 10+1)]
    time_columns = ['time%d' % i for i in range(1, 10+1)]
    
    Xy = pd.read_csv(input_path.joinpath('train_sessions.csv'), index_col='session_id', parse_dates=time_columns)
    X_submission = pd.read_csv(input_path.joinpath('test_sessions.csv'), index_col='session_id', parse_dates=time_columns)
    
    Xy = fix_incorrect_date_formats(Xy, time_columns)
    X_submission = fix_incorrect_date_formats(X_submission, time_columns)
    
    Xy = Xy.sort_values(by='time1')
    
    y = Xy['target'].astype('int').values
    X = Xy.drop(columns=['target'])
    
    ujob.dump_multiple([X, X_submission, y], 
                       output_path, 
                       ['X.joblib', 'X_submission.joblib', 'y.joblib'])


@click.command()
@click.option('-i', 'input_path', 
              type=PathlibPath(exists=True, file_okay=False), default=RAW_DATA_DIR, 
              help='Input directory for raw data files (default: <project_dir>/data/raw)')
@click.option('-o', 'output_path', 
              type=PathlibPath(exists=True, file_okay=False), default=DATASET_DIR, 
              help='Output directory for the dataset (default: <project_dir>/data/processed/dataset)')
def cli(input_path, output_path):
    """ Processes raw data to construct a dataset.
    """
    logger = logging.getLogger(__name__)
    logger.info('making a dataset from raw data')
    
    main(input_path, output_path)
        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
