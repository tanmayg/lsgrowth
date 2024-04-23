import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def concat_non_na(series):
    """
    Concatenates non-NA (non-None) values from a Pandas Series into a single string.

    Parameters:
    - series (pandas.Series): The Pandas Series containing values to concatenate.

    Returns:
    str: A comma-separated string of non-NA values from the input Series.
    """
    try:
        # Use join to concatenate values, filtering out None or NA values
        return ', '.join(filter(lambda x: x is not None, series.astype(str)))
    
    except Exception as e:
        raise CustomException(e, sys)
    
def get_source(group):
    '''
    this function keeps the non-NA CTA_ID if there is a mix of NA/non-NA CTA_ID in the group else return the CTA_ID
    
    Args:
    group : grouped sub-dataframe

    Returns:
    updated sub-dataframe
    '''
    try:
        non_na_sources = group['CTA_ID'].dropna()
        if not non_na_sources.empty:
            return non_na_sources.values[0]
        return np.nan
    
    except Exception as e:
        raise CustomException(e, sys)
    
def update_row_flag(group):
    '''
    this function updates previous row flag as 0 if current row TimeDiff<30

    Args:
    group : grouped sub-dataframe

    Returns:
    group : updated sub-dataframe
    '''
    prev_idx = None
    for index, row in group.iterrows():
        if prev_idx is not None:
            if row['TimeDiff'] <= 30:
                # Update TD Flag
                group.at[prev_idx, 'TD_Flag'] = 0
                # Concatenate Follow up/Comments so that no information is lost
                if group.at[prev_idx, 'Follow_Up'] not in [None, 'Unknown']:
                    group.at[index, 'Follow_Up'] = group.at[prev_idx, 'Follow_Up'] + '\n' +group.at[index, 'Follow_Up']
                if group.at[prev_idx, 'Follow Up Comments'] not in [None, 'Unknown']:
                    group.at[index, 'Follow Up Comments'] = group.at[prev_idx, 'Follow Up Comments'] + '\n' + group.at[index, 'Follow Up Comments']
                # If lead id is null, update with the previous lead id
                if group.at[index, 'sfdc_lead_id'] is None:
                    group.at[index, 'sfdc_lead_id'] = group.at[prev_idx, 'sfdc_lead_id']
                
        prev_idx = index
    return group