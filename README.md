
user_identification
==============================

User identification based on a session of visited websites. Final project of [Yandex&MIPT specialization](https://www.coursera.org/specializations/machine-learning-data-analysis) with a related kaggle competition ["Catch Me If You Can (Alice)"](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2).

Based on [ml_template](https://github.com/solery-git/ml_template).


How to reproduce
------------

The training process is organized as a pipeline of separate steps with command-line interface available in `main.py`. Use `python3 main.py` for help if needed.

Get the data from kaggle competition and place it into `/data/raw` directory.

Use `python3 main.py run-all` to run the pipeline. This produces a serialized trained model and evaluation metrics in `/models` directory. After that, run `python3 main.py make-submission` to generate a submission file.

Inference
------------

Additionally, you can use the trained model to create a Docker container suitable both for batch and online inference.

To build a Docker image: 
`docker build -t user-identification -f Dockerfile .`

Batch inference mode currently accepts only joblib-serialized preprocessed datasets, like those produced by `preprocess-dataset` step.
To run a container in batch inference mode: 
`docker run --rm -v [PATH_TO_DATASET_DIR]:/home/jovyan/work user-identification python3 batch_inference.py work/[DATASET_FILENAME]`

To run a container in online inference mode: 
`docker run --rm -it -p 5000:5000 user-identification python3 api.py`
An example of getting predictions in online inference mode is provided in `notebooks/api_test.ipynb`.

Course notebooks
-----------
Coursera final project notebooks can be found in `/notebooks/coursera` directory. 
For a final report, check `/notebooks/coursera/7.1-week7-final_report.ipynb`.
