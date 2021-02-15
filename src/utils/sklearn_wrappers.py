# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


# based on https://feature-engine.readthedocs.io/en/latest/wrappers/Wrapper.html
class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn pre-processing transformers like the
    SimpleImputer() or OrdinalEncoder(), to allow the use of the
    transformer on a selected group of variables.
    Parameters
    ----------
    transformer : sklearn transformer, default=None
        The desired Scikit-learn transformer.
    variables : list, default=None
        The list of variables to be imputed.
    """

    def __init__(self, transformer=None, variables=None):
        self.transformer = transformer
        self.variables = variables
        if isinstance(self.transformer, OneHotEncoder) and self.transformer.sparse:
            raise AttributeError('The SklearnTransformerWrapper can only wrap the OneHotEncoder if you '
                                 'set its sparse attribute to False')

    def fit(self, X, y=None):
        """
        The `fit` method allows Scikit-learn transformers to learn the required parameters
        from the training data set.
        """

        self.transformer.fit(X[self.variables])

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Apply the transformation to the dataframe. Only the selected features will be modified. 
        If transformer is OneHotEncoder, dummy features are concatenated to the source dataset.
        Note that the original categorical variables will not be removed from the dataset
        after encoding. If this is the desired effect, please use Feature-engine's 
        OneHotCategoricalEncoder instead.
        """

        if isinstance(self.transformer, OneHotEncoder):
            ohe_results_as_df = pd.DataFrame(
                data=self.transformer.transform(X[self.variables]),
                columns=self.transformer.get_feature_names(self.variables)
            )
            X_new = pd.concat([X, ohe_results_as_df], axis=1)
        else:
            X_new = X.copy()
            X_new[self.variables] = self.transformer.transform(X[self.variables])

        return X_new


class SklearnVectorizerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn text vectorizers like the
    CountVectorizer() or TfidfVectorizer(), to allow the use of the
    transformer on a selected group of variables with values 
    treated as strings.
    Parameters
    ----------
    vectorizer : sklearn text vectorizer, default=None
        The desired Scikit-learn vectorizer.
    variables : list, default=None
        The list of variables to be imputed.
    sep : str, default=''
        Separator to use.
    as_df : bool, default=True
        Whether to return transformed data as a dataframe or as a numpy array.
    """
    def __init__(self, vectorizer=None, variables=None, sep='', as_df=True, drop_stop_words=False):
        self.vectorizer = vectorizer
        self.variables = variables
        self.sep = sep
        self.as_df = as_df
        self.drop_stop_words = drop_stop_words
    
    def _to_text(self, X):
        X_text = X.apply(lambda row: self.sep.join(map(str, row)), axis=1, raw=True).values.flatten().tolist()
        return X_text
    
    def fit(self, X, y=None):
        """
        The `fit` method allows Scikit-learn transformers to learn the required parameters
        from the training data set.
        """
        self.input_shape_ = X.shape
        
        X_text = self._to_text(X[self.variables])
        self.vectorizer.fit(X_text)
        
        if self.drop_stop_words:
            self.vectorizer.stop_words_ = None
        
        return self

    def transform(self, X):
        """
        Apply the transformation to the dataframe. Only the selected features 
        will be modified, the rest will be removed.
        """
        X_text = self._to_text(X[self.variables])
        X_sparse = self.vectorizer.transform(X_text)
        if self.as_df:
            X_new = pd.DataFrame.sparse.from_spmatrix(X_sparse, index=X.index, columns=self.vectorizer.get_feature_names())
        else:
            X_new = X_sparse

        return X_new
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()