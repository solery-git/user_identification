# -*- coding: utf-8 -*-
from flask import Flask
from flask.views import MethodView
import marshmallow as ma
from flask_smorest import Api, Blueprint, abort

import os
from joblib import load

import numpy as np
import pandas as pd


MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

print("Loading model from: {}".format(MODEL_PATH))
model = load(MODEL_PATH)


app = Flask(__name__)
app.config['API_TITLE'] = 'My API'
app.config['API_VERSION'] = 'v1'
app.config['OPENAPI_VERSION'] = '3.0.2'
api = Api(app)


class SiteVisitSchema(ma.Schema):
    site = ma.fields.String(required=True)
    time = ma.fields.DateTime(required=True)


class SessionSchema(ma.Schema):
    session = ma.fields.List(ma.fields.Nested(SiteVisitSchema), validate=ma.validate.Length(min=1, max=10))


class PredictionSchema(ma.Schema):
    prediction = ma.fields.Int()
    probability_class_1 = ma.fields.Float()


blp = Blueprint(
    'predict', 'predict', url_prefix='/predict',
    description='Predictions on sessions'
)

@blp.route('/')
class Prediction(MethodView):
    
    @staticmethod
    def _to_dataframe(session_dict):
        session = session_dict['session']
        
        # pad to 10 site visits with default values
        session += [{'site': 'unknown', 'time': None}] * (10 - len(session))
        
        # construct a dict for DataFrame constructor
        session_data = {}
        for i, site_visit in enumerate(session, start=1):
            session_data[f'site{i}'] = [site_visit['site']]
            session_data[f'time{i}'] = [np.datetime64(site_visit['time'])]
        
        return pd.DataFrame.from_dict(session_data)
    
    @blp.arguments(SessionSchema)
    @blp.response(200, PredictionSchema)
    def post(self, session):
        """Make prediction"""
        X = Prediction._to_dataframe(session)
        y_pred_proba = model.predict_proba(X)[:, 1][0]
        y_pred = int(y_pred_proba >= 0.5)
        return {'prediction': y_pred, 'probability_class_1': y_pred_proba}


api.register_blueprint(blp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')