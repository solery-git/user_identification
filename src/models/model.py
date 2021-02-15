# -*- coding: utf-8 -*-
import numpy as np
from src.features.creation import FeatureCreator
from src.utils.sklearn_wrappers import SklearnTransformerWrapper, SklearnVectorizerWrapper
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from eli5 import transform_feature_names
import pickle
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR.joinpath('data/raw/')


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.features = X.columns
        return self
    
    def transform(self, X):
        return X.values
    
    def get_feature_names(self):
        return self.features


def make_site_vectorizer():
    site_columns = ['site%d' % i for i in range(1, 10+1)]
    time_columns = ['time%d' % i for i in range(1, 10+1)]
    
    # vectorize sites with tfidf
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\S+", 
                                 stop_words=['unknown'], 
                                 ngram_range=(1, 5), 
                                 max_features=50000, 
                                 sublinear_tf=True, 
                                 use_idf=True)
    
    site_vectorizer = SklearnVectorizerWrapper(vectorizer, 
                                               variables=site_columns, 
                                               sep=' ', 
                                               as_df=False, 
                                               drop_stop_words=True)
    
    return site_vectorizer

def make_feature_transformer():
    site_columns = ['site%d' % i for i in range(1, 10+1)]
    time_columns = ['time%d' % i for i in range(1, 10+1)]
    
    # transform some custom features (drop the rest) and add them to vectorized sites
    site_vectorizer = make_site_vectorizer()
    
    feature_transformer = ColumnTransformer([
        ('site_vectorizer', site_vectorizer, site_columns), 
        ('one_hot_encoder', OneHotEncoder(sparse=True, dtype=np.int16), ['time_of_day']), 
        ('standard_scaler', StandardScaler(), ['session_timespan']), 
        ('max_abs_scaler', MaxAbsScaler(), ['day_of_week', 
                                            'year_month', 
                                            'month'])
    ], remainder='drop')
    
    return feature_transformer

def make_model():
    feature_transformer = make_feature_transformer()
    predictor = LogisticRegression(C=1, random_state=17, solver='liblinear')
    
    # Final model
    model = Pipeline([
        ('feature_creator', FeatureCreator()), 
        ('feature_transformer', feature_transformer), 
        ('predictor', predictor)
    ])
    
    return model


# allow ColumnTransformer to show feature names with eli5
# from https://stackoverflow.com/questions/60949339/how-to-get-feature-names-from-eli5-when-transformer-includes-an-embedded-pipelin
@transform_feature_names.register(ColumnTransformer)
def _col_tfm_names(transformer, in_names=None):
    if in_names is None:
        from eli5.sklearn.utils import get_feature_names
        # generate default feature names
        in_names = get_feature_names(transformer, num_features=transformer._n_features)
    # return a list of strings derived from in_names
    feature_names = []
    for name, trans, column, _ in transformer._iter(fitted=True):
        if hasattr(transformer, '_df_columns'):
            if ((not isinstance(column, slice))
                    and all(isinstance(col, str) for col in column)):
                names = column
            else:
                names = transformer._df_columns[column]
        else:
            indices = np.arange(transformer._n_features)
            names = ['x%d' % i for i in indices[column]]
        # erm, want to be able to override with in_names maybe???

        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            continue
        if trans == 'passthrough':
            feature_names.extend(names)
            continue
        feature_names.extend([name + "__" + f for f in
                              transform_feature_names(trans, in_names=names)])
    return feature_names

@transform_feature_names.register(OneHotEncoder)
def _ohe_names(est, in_names=None):
    return est.get_feature_names(input_features=in_names)