# -*- coding: utf-8 -*-
import click
from pathlib import Path

from src.data import make_dataset, preprocess_dataset, split_train_test
from src.models import evaluate, make_submission
from src.utils.click import OrderedGroup


STAGES_PIPELINE = [
    ('make-dataset', make_dataset), 
    ('preprocess-dataset', preprocess_dataset), 
    ('split-train-test', split_train_test), 
    ('evaluate', evaluate)
]

STAGES_STANDALONE = [
    ('make-submission', make_submission)
]


@click.group(cls=OrderedGroup, 
             help='''Pipeline entrypoint. Use 'run-all' command to run the whole pipeline
             or use the respective commands to run specific steps.\n 
             For a detailed help on a specific command, use '<command> --help'.''')
def cli():
    pass


@cli.command(help='Run all the steps up to evaluate (with default parameters).')
def run_all():
    for stage, script in STAGES_PIPELINE:
        print(f'Running stage: {stage}')
        script.main()

for stage, script in STAGES_PIPELINE + STAGES_STANDALONE:
    cli.add_command(script.cli, name=stage)


if __name__ == '__main__':
    cli()