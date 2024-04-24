import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    input_files: dict
    reference_files: dict

class DataIngestion:
    def __init__(self, ingestion_config, input_df=None):
        self.ingestion_config = ingestion_config
        self.input_df = input_df  # Direct DataFrame input

    def read_data_files(self, file_type):
        try:
            read_dfs = {}
            if file_type == 'input':
                file_items = self.ingestion_config.input_files.items()
            elif file_type == 'reference':
                file_items = self.ingestion_config.reference_files.items()
            else:
                raise ValueError(f"Invalid file type: {file_type}. Expected 'input' or 'reference'.")

            for ref_name, file_info in file_items:
                if isinstance(file_info['file_name'], list):
                    # Read all files and concatenate into a single DataFrame
                    dfs = [pd.read_csv(file) if os.path.splitext(file)[1] == '.csv' else pd.read_excel(file) for file in file_info['file_name']]
                    read_dfs[ref_name] = pd.concat(dfs, ignore_index=True)
                else:
                    file_ext = os.path.splitext(file_info['file_name'])[1]
                    if file_ext == '.csv':
                        df = pd.read_csv(file_info['file_name'], sep=file_info.get('sep', ','))
                        read_dfs[ref_name] = df
                    elif file_ext == '.xlsx':
                        df = pd.read_excel(file_info['file_name'])
                        read_dfs[ref_name] = df
                    elif file_ext == '.pkl':
                        with open(file_info['file_name'], 'rb') as file:
                            read_dfs = pickle.load(file)
                    else:
                        raise ValueError(f"Unsupported file extension: {file_ext}")

            return read_dfs
        
        except Exception as e:
            logging.exception(f"!!!!! Error occurred during execution {e}")
            raise CustomException(e, sys)

        
    def initiate_data_ingestion(self, is_training=True):
        try:
            # Reading common reference files
            if self.input_df is not None:
                in_df = {'main_input': self.input_df}
            else:
                in_df = self.read_data_files('input')
            ref_df = self.read_data_files('reference')
            '''
            if is_training:
                # Perform operations specific to training
                df = self.read_data_files('input')
                logging.info('Read input files')
                #os.makedirs(os.path.dirname("artifacts"), exist_ok=True)
                #df.to_csv(self.ingestion_config.data_path, index=False, header=True)

                logging.info("Ingestion of the training data is completed")
            else:
                # Perform operations specific to prediction
                logging.info("Ingestion of the prediction data is completed")
            '''
            return in_df, ref_df
        
        except Exception as e:
            logging.exception(f"!!!!! Error occurred during execution {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":

    # Define input and reference files
    input_files = {
        'cdo': {'file_name': ['C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/Data/Eloqua/CDO Data/CDO_Reference.csv',
                            'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/Data/Eloqua/CDO Data/2023/MQL CDO Data for lead scoring 2023 Jan to June.csv',
                            'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/Data/Eloqua/CDO Data/2023/MQL CDO Data for lead scoring 2023 Jul to Dec.csv'
                            ]},
        'opportunity': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/Data/E2E_MQL_to_Oppotunity_Growth.xlsx'}
    }
    reference_files = {
        'contacts': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/Data/Contacts_Reference.csv'},
        'mrkt_ref': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/Data/New_Market_Hierarchy.csv'},
        'acct_ref': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/Data/SFDC_Growth_Account_Reference_X360.csv'},
        'prch_hist': {'file_name': 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/Data/CDO_Purchase_History_Raw_Growth.csv'},
    }

    ingestion_config = DataIngestionConfig(input_files=input_files, reference_files=reference_files)
    data_ingestion = DataIngestion(ingestion_config)
    input_dfs, reference_dfs = data_ingestion.initiate_data_ingestion(is_training=True)