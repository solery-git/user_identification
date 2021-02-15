# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder


def _count_unique_sites(row):
    row_unique = np.unique(row)
    return len(row_unique[row_unique != 0]) #0 is not a valid site ID

def _count_popular_sites(sites_df, popular_sites_indicators):
    result = []
    for sites_row, popular_sites_mask in zip(sites_df.values, popular_sites_indicators):
        popular_sites = sites_row[popular_sites_mask]
        result.append( len(popular_sites) )
    return result

def _get_session_timespan(row):
    row_ne = row[~np.isnat(row)]
    return int((row_ne[-1]-row_ne[0]) / np.timedelta64(1, 's'))

def _get_times_std(row):
    row_ne = row[~np.isnan(row)]
    return np.std(row_ne) if len(row_ne) > 0 else 0

def _percent_long_visits(row, threshold=2):
    row_ne = row[~np.isnan(row)]
    return np.count_nonzero(row_ne >= threshold) / len(row_ne) if len(row_ne) > 0 else 0

def _between(left, x, right):
    return left <= x and x <= right

def _get_time_of_day(hour):
    if _between(0, hour, 6):
        return 0
    elif _between(7, hour, 11):
        return 1
    elif _between(12, hour, 18):
        return 2
    elif _between(19, hour, 23):
        return 3
    else:
        raise ValueError(f'hour should be within (0, 23) range; got {hour}')

def _encode_year_month(datetime):
    return (datetime.year * 12 + datetime.month) - (2013 * 12 + 11)


def pairwise_product(df1, df2, sparse=False):
    cols1 = df1.columns
    cols2 = df2.columns
    
    result = pd.DataFrame(index=df1.index)
    for col1 in cols1:
        for col2 in cols2:
            product = np.multiply(df1[col1].values, df2[col2].values)
            if sparse:
                result[f'[x]_{col1}_-_{col2}'] = pd.arrays.SparseArray(product)
            else:
                result[f'[x]_{col1}_-_{col2}'] = product
    
    return result

def make_indicators(df, sparse=True):
    ohe = OneHotEncoder(sparse=sparse, dtype=np.int8)
    data = ohe.fit_transform(df)
    if sparse:
        return pd.DataFrame.sparse.from_spmatrix(data, index=df.index, columns=ohe.get_feature_names(df.columns))
    else:
        return pd.DataFrame(data, index=df.index, columns=ohe.get_feature_names(df.columns))


class FeatureCreator(BaseEstimator, TransformerMixin):
    """ A transformer that constructs and appends new features to the dataframe.
    Parameters
    ----------
    None
    
    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    
    data_feature_names_ : list of strings
        Names of features from the fitted data.
    
    new_feature_names_ : list of strings
        Names of the constructed features.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        self.data_feature_names_ = list(X.columns)
        self.new_feature_names_ = ['hour', 
                                   'time_of_day', 
                                   'day_of_week', 
                                   'weekend', 
                                   '#popular_sites', 
                                   'year_month', 
                                   'month', 
                                   'session_timespan', 
                                   'times_std', 
                                   '%long_visits']
        
        # Derive various data from training set
        # Most popular sites
        site_columns = ['site%d' % i for i in range(1, 10+1)]
        X_sites = X[site_columns].copy()
        
        site_counter = Counter()
        for row in X_sites.values:
            site_counter.update(row)
        del site_counter['unknown']
        self._popular_sites = [site for site, count in site_counter.most_common(50)]
        
        # Return the transformer
        return self
    
    def transform(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        X_new = X.copy()
        
        site_columns = ['site%d' % i for i in range(1, 10+1)]
        time_columns = ['time%d' % i for i in range(1, 10+1)]
        X_sites = X[site_columns]
        X_times = X[time_columns]
        
        popular_sites_indicators = np.isin(X_sites.values, self._popular_sites)
        
        X_time_diffs = np.diff(X_times.values, 1, axis=1) / np.timedelta64(1, 's')
        
        # Add new features
        X_new['hour'] = X_new['time1'].dt.hour
        X_new['time_of_day'] = X_new['hour'].apply(_get_time_of_day)
        X_new['day_of_week'] = X_new['time1'].dt.dayofweek
        X_new['weekend'] = (np.isin(X_new['day_of_week'].values, [5, 6])).astype(int)
        X_new['#popular_sites'] = _count_popular_sites(X_sites, popular_sites_indicators)
        X_new['year_month'] = X_new['time1'].apply(_encode_year_month)
        X_new['month'] = X_new['time1'].dt.month
        X_new['session_timespan'] = X_times.apply(lambda row: _get_session_timespan(row), axis=1, raw=True)
        X_new['times_std'] = np.apply_along_axis(_get_times_std, 1, X_time_diffs)
        X_new['%long_visits'] = np.apply_along_axis(_percent_long_visits, 1, X_time_diffs)
        
        return X_new
    
    def get_feature_names(self):
        return self.data_feature_names_ + self.new_feature_names_