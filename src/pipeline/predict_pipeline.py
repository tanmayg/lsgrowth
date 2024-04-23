import numpy as np 
import pandas as pd
import h2o
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_processing import DataProcessor, DataProcessingConfig
from src.components.data_train_prep import Word2VecConfig, Word2VecProcessor, DataPreparationConfig, DataPreparation
from src.components.predict import LeadModelConfig, LeadModelPrediction
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object
import argparse
import pkg_resources

class PredictionPipeline:
    def __init__(self, input_files, reference_files):
        self.input_files = input_files
        self.reference_files = reference_files

    def run(self):
        logging.info(">>>>> Started: Data Ingestion Module <<<<<")
        ingestion_config = DataIngestionConfig(input_files=self.input_files, reference_files=self.reference_files)
        data_ingestion = DataIngestion(ingestion_config)
        input_dfs, reference_dfs = data_ingestion.initiate_data_ingestion(is_training=False)
        logging.info(">>>>> Completed: Data Ingestion Module <<<<<")

        logging.info(">>>>> Started: Data Processing Module <<<<<")
        # Assuming df_cdo_acct_final is available
        data_processing_config = DataProcessingConfig(
            df_cdo=input_dfs['cdo'],
            #df_opty=input_dfs['opportunity'],
            df_st_prov_ref=reference_dfs['df1'],
            df_sfdc_acct_nam=reference_dfs['df2'],
            df_cdo_ph_raw=reference_dfs['df3'], 
            df_cdo_acct_final=reference_dfs['df4'], 
            df_ct=reference_dfs['df5'],
            prediction_flag = 1
            )
        data_processor = DataProcessor(data_processing_config)
        data_processor.process_pred_data()
        df_features = data_processor.df_features
        logging.info(">>>>> Completed: Data Processing Module <<<<<")

        #Word2Vec processing
        logging.info(">>>>> Started: Word2Vec Module (for comments) <<<<<")
        h2o.init()
        word2vec_config = Word2VecConfig(input_df = df_features,
                                        word2vec_model = './artifacts/word2vec_model/Word2Vec_model_python_1705435940762_1',
                                        train_new=False,
                                        prediction_flag = 1)
        word2vec_processor = Word2VecProcessor(word2vec_config)
        word2vec_processor.process()
        comments_vecs = word2vec_processor.comments_vecs
        logging.info(">>>>> Completed: Word2Vec Module (for comments) <<<<<")

        logging.info(">>>>> Started: Data Ready for Prediction Module <<<<<")
        #ready frame for prediction
        data_prep_config = DataPreparationConfig(
            df_features = df_features,
            comments_vecs = comments_vecs
            )
        data_prep = DataPreparation(data_prep_config)
        df_h2o = data_prep.prepare_data()
        logging.info(">>>>> Completed: Data Ready for Prediction Module <<<<<")

        #make predictions
        logging.info(">>>>> Started: Prediction with Explanation Module <<<<<")
        lead_model_config = LeadModelConfig(input_df = df_h2o,
                                            lead_model = './artifacts/lead_model/final_grid_2024Jan17_0142_model_2',
                                            word_emb = word2vec_processor.word_embeddings,
                                            nrows = 10)
        lead_preds = LeadModelPrediction(lead_model_config)
        highlighted_comments = lead_preds.process_predict()
        highlighted_comments.to_csv('./artifacts/prediction_w_explain.csv')
        logging.info(">>>>> Completed: Prediction with Explanation Module <<<<<")
        
        # Clean up and prepare the final output
        final_columns = ['Prediction', 'Prob.of No Conversion', 'Prob.of Conversion', 'Significant Contributors', 'User Comment', 'Keywords Highlighted']
        highlighted_comments = highlighted_comments[['predict', 'p0', 'p1', 'Non-Text Contributors', 'Original text', 'Transformed text']]
        highlighted_comments.columns = final_columns
        highlighted_comments_dict = highlighted_comments.to_dict('records')

        logging.info(">>>>>> PREDICT PIPELINE COMPLETED <<<<<<")
        h2o.cluster().shutdown()
        return highlighted_comments_dict

# Example usage in the same script
if __name__ == "__main__":
    # Define input and reference files
    parser = argparse.ArgumentParser(description="Predict lead scores for new records.")
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing new records.')
    args = parser.parse_args()

    # Get the path to packaged reference files
    #reference_file_path_1 = pkg_resources.resource_filename('PythonLeadScoringModel', 'data/reference_file_1.pkl')
    reference_file_path_1 = 'C:/Users/310223340/OneDrive - Philips/MarketingAnalytics/LEAD_SCORING/SYSTEM LEAD SCORING/LS_Reference_Files.pkl'
    # Initialize your pipeline with the input file and reference file paths
    input_files = {'cdo': {'file_name':args.file_path}}  # Adjust according to your actual input structure
    reference_files = {'reference': {'file_name': reference_file_path_1}}  # Adjust according to how you use them

    # Initialize H2O and set up the configuration
    logging.info(">>>>>> PREDICT PIPELINE BEGINS <<<<<<")
    pipeline = PredictionPipeline(input_files, reference_files)
    results = pipeline.run()
    print(results)
