import os
import click
from src.utils.click import PathlibPath
from pathlib import Path

from joblib import load
import numpy as np


MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)


@click.command()
@click.argument('input_file', 
              type=PathlibPath(exists=True, dir_okay=False))
@click.option('-o', 'output_file', 
              type=PathlibPath(dir_okay=False), 
              help='Path to the CSV file with generated predictions (default: <input directory>/predictions.csv)')
@click.option('-p', 'probabilities', is_flag=True, 
              help='Whether to generate probabilities of class 1 instead of predicted classes')
def main(input_file, output_file, probabilities):
    if output_file is None:
        output_file = input_file.resolve().parent.joinpath('predictions.csv')
    
    print("Loading input file...")
    X = load(input_file)
    
    print("Loading model...")
    model = load(MODEL_PATH)
    
    print("Generating predictions...")
    if probabilities:
        result = model.predict_proba(X)[:, 1]
        fmt = '%.5f'
    else:
        result = model.predict(X)
        fmt = '%d'
    
    np.savetxt(output_file, result, delimiter=',', fmt=fmt)
    print(f"Predictions saved to {output_file}.")


if __name__ == '__main__':
    main()
