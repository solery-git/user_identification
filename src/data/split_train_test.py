# -*- coding: utf-8 -*-
import click
from src.utils.click import PathlibPath
from pathlib import Path
import logging
import src.utils.joblib as ujob

from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR.joinpath('data/processed/')
DATASET_PROCESSED_DIR = PROCESSED_DATA_DIR.joinpath('dataset_processed')
TRAIN_TEST_DIR = PROCESSED_DATA_DIR.joinpath('train_test')


def main(input_path=DATASET_PROCESSED_DIR, output_path=TRAIN_TEST_DIR):
    output_path.mkdir(parents=True, exist_ok=True)
    X, y = ujob.load_multiple(input_path, ['X.joblib', 'y.joblib'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.3, shuffle=False)
    
    ujob.dump_multiple([X_train, X_test, y_train, y_test], 
                       output_path, 
                       ['X_train.joblib', 'X_test.joblib', 'y_train.joblib', 'y_test.joblib'])


@click.command()
@click.option('-i', 'input_path', 
              type=PathlibPath(exists=True, file_okay=False), default=DATASET_PROCESSED_DIR, 
              help='Input directory for the processed dataset (default: <project_dir>/data/processed/dataset_processed)')
@click.option('-o', 'output_path', 
              type=PathlibPath(exists=True, file_okay=False), default=TRAIN_TEST_DIR, 
              help='Output directory for the train/test dataset parts (default: <project_dir>/data/processed/train_test)')
def cli(input_path, output_path):
    """ Splits dataset into train and test parts.
    """
    logger = logging.getLogger(__name__)
    logger.info('splitting data into train and test')
    
    main(input_path, output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()