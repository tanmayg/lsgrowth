#%%
import sys
sys.path.insert(0, 'c:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/leadscoring/src')

import src.components.data_ingestion
print("SRC:", src.components.data_ingestion.__file__)

import numpy as np 
import pandas as pd
import h2o

# Import your modules here
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_processing import DataProcessor, DataProcessingConfig
from src.components.data_train_prep import Word2VecConfig, Word2VecProcessor, DataPreparationConfig, DataPreparation
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

#%%
# Define input and reference files
input_files = {
    'cdo': {'file_name': ['C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/leadscoring/CDO_Reference.csv',
                          'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/Data/Eloqua/CDO Data/2023/MQL CDO Data for lead scoring 2023 Jan to June.csv',
                          'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/Data/Eloqua/CDO Data/2023/MQL CDO Data for lead scoring 2023 Jul to Dec.csv'
                          ]},
    'opportunity': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/leadscoring/E2E_MQL_to_Oppotunity.xlsx'}
}
reference_files = {
    'contacts': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/leadscoring/Contacts_Reference.csv'},
    'us_states': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/leadscoring/NAM_States_Provinces.csv'},
    'acct_ref': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/leadscoring/SFDC_NAM_Account_Reference_X360.csv'},
    'prch_hist': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/leadscoring/CDO_Purchase_History_Raw.csv'},
    'df_cdo_acct_final': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/leadscoring/CDO_Account_Reference_DS.csv'},
}

#%%
def run_data_ingestion(config):
    logging.info("Started: Data Ingestion Module")
    data_ingestion = DataIngestion(config)
    input_dfs, reference_dfs = data_ingestion.initiate_data_ingestion(is_training=True)
    logging.info("Completed: Data Ingestion Module")
    return input_dfs, reference_dfs

def run_data_transformation(config):
    logging.info("Started: Data Transformation Module")
    data_processor = DataProcessor(config)
    data_processor.process_data()
    df_features = data_processor.df_cdo_final
    logging.info("Completed: Data Transformation Module")
    return df_features

def run_model_training(config):
    logging.info("Started: Word2Vec Model Training")
    processor = Word2VecProcessor(config)
    processor.process()
    comments_vecs = processor.comments_vecs
    logging.info("Completed: Word2Vec Model Training")
    return comments_vecs

def run_data_preparation(df_features, comments_vecs, config):
    logging.info("Started: Data Prep Before Model Training")
    data_prep = DataPreparation(df_features, comments_vecs, config)
    train, valid, test = data_prep.prepare_data()
    predictors = data_prep.predictors
    save_file = f"{'artifacts'}\{'predictors'}"
    save_object(save_file, predictors)
    #logging.info(train.shape, valid.shape, test.shape)
    logging.info("Completed: Data Prep Before Model Training")
    return train, valid, test

def run_data_preparation_ex(df_features, comments_vecs, config):
    logging.info("Started: Data Prep Before Model Training")
    data_prep = DataPreparation(df_features, comments_vecs, config)
    train, valid, test = data_prep.prepare_data_ex()
    predictors = data_prep.predictors
    save_file = f"{'artifacts'}\{'predictors'}"
    save_object(save_file, predictors)
    #logging.info(train.shape, valid.shape, test.shape)
    logging.info("Completed: Data Prep Before Model Training")
    return train, valid, test

#%%
if __name__ == "__main__":
    logging.info("\n>>>>>> DATA PIPELINE BEGINS <<<<<<\n")
    
    # flag for new fuzzy match
    no_fuzz_flg = True

    # Add flags or conditions for each stage
    run_ingestion = True
    run_transformation = True
    run_training = True
    run_preparation = True
    run_preparation_ex = True

    # Data Ingestion
    if run_ingestion:
        ingestion_config = DataIngestionConfig(input_files=input_files, reference_files=reference_files)
        input_dfs, reference_dfs = run_data_ingestion(ingestion_config)

    # Data Transformation
    if run_transformation:
        # Assuming df_cdo_acct_final is available
        data_processing_config = DataProcessingConfig(
            df_cdo=input_dfs['cdo'],
            df_opty=input_dfs['opportunity'],
            df_st_prov_ref=reference_dfs['us_states'],
            df_sfdc_acct_nam=reference_dfs['acct_ref'],
            df_cdo_ph_raw=reference_dfs['prch_hist'], 
            df_cdo_acct_final=reference_dfs['df_cdo_acct_final'], 
            df_ct=reference_dfs['contacts']
            )
        df_features = run_data_transformation(data_processing_config)

    # Model Training
    h2o.init()
    if run_training:
        word2vec_config = Word2VecConfig(input_file='./artifacts/df_features_wo_personal.csv',
                                         output_file='./artifacts/word2vec_model',
                                         train_new=False)
        comments_vecs = run_model_training(word2vec_config)

    # Data Preparation
    if run_preparation:
        SEED = 12345
        split_ratio = 0.4
        data_prep_config = DataPreparationConfig(seed=SEED, split_ratio=split_ratio)
        train, valid, test = run_data_preparation(df_features, comments_vecs, data_prep_config)

        # Data Preparation
    if run_preparation_ex:
        df_features = pd.read_csv('./artifacts/df_features_wo_personal.csv')
        SEED = 555
        split_ratio = 0.4
        data_prep_config = DataPreparationConfig(seed=SEED, split_ratio=split_ratio)
        train, valid, test = run_data_preparation_ex(df_features, comments_vecs, data_prep_config)

    logging.info("\n>>>>>> DATA PIPELINE COMPLETED <<<<<<")

# %%
