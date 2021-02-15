FROM jupyter/scipy-notebook

COPY api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ENV MODEL_DIR=/home/jovyan
ENV MODEL_FILE=model.joblib
ENV METADATA_FILE=metadata.json

COPY models/model.joblib ./model.joblib
COPY src ./src
COPY api/batch_inference.py ./batch_inference.py
COPY api/api.py ./api.py