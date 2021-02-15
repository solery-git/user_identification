# -*- coding: utf-8 -*-
import click
from src.utils.click import PathlibPath
from pathlib import Path
import logging
import src.utils.joblib as ujob

import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score
import eli5
from src.models.model import make_model


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR.joinpath('data/processed/')
DATASET_DIR = PROCESSED_DATA_DIR.joinpath('dataset')
TRAIN_TEST_DIR = PROCESSED_DATA_DIR.joinpath('train_test')
MODELS_DIR = PROJECT_DIR.joinpath('models')


def main(input_path=TRAIN_TEST_DIR, output_path=MODELS_DIR):
    output_path.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = ujob.load_multiple(input_path, 
        ['X_train.joblib', 'X_test.joblib', 'y_train.joblib', 'y_test.joblib'])
    
    model = make_model()
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)
    
    # Evaluate the model via cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # Store and print evaluation metrics
    metrics = {}
    metrics['cv_scores'] = {}
    for i, value in enumerate(cv_scores.tolist(), start=1):
        metrics['cv_scores'][f'fold{i}'] = float(value)
    metrics['score_mean'] = float(np.mean(cv_scores))
    metrics['score_std'] = float(np.std(cv_scores))
    
    print(f'CV scores: {list(metrics["cv_scores"].values())}')
    print(f'Mean CV score: {metrics["score_mean"]}')
    print(f'Std CV score: {metrics["score_std"]}')
    
    # Fit the model on the whole dataset
    model.fit(X_train, y_train)
    
    # Show feature importances
    formatter = eli5.formatters.text.format_as_text
    print(formatter(eli5.explain_weights(model, feature_filter=lambda x: x.split('__')[0] == 'site_vectorizer')))
    print(formatter(eli5.explain_weights(model, feature_filter=lambda x: x.split('__')[0] != 'site_vectorizer'), show=['targets']))
    
    # Calculate model errors
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)
    errors_mask = (y_test_pred != y_test)
    errors = np.array([np.arange(X_test.shape[0]), y_test_pred, y_test]).transpose()[errors_mask]
    errors_proba = y_test_pred_proba[errors_mask]
    
    # Export metrics and errors
    ujob.dump_multiple([model, errors, errors_proba], 
                       output_path, 
                       ['model.joblib', 'errors.joblib', 'errors_proba.joblib'])  
    with open(MODELS_DIR.joinpath('metrics.yaml'), 'w') as fout:
        yaml.dump(metrics, fout, sort_keys=False)


@click.command()
@click.option('-i', 'input_path', 
              type=PathlibPath(exists=True, file_okay=False), default=TRAIN_TEST_DIR, 
              help='Input directory for the train/test dataset parts (default: <project_dir>/data/processed/train_test)')
@click.option('-o', 'output_path', 
              type=PathlibPath(exists=True, file_okay=False), default=MODELS_DIR, 
              help='Output directory for the model (default: <project_dir>/models)')
def cli(input_path, output_path):
    """ Evaluate and export the trained model.
    """
    logger = logging.getLogger(__name__)
    logger.info('evaluating the model')
    
    main(input_path, output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()