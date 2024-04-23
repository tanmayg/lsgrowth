import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
import re

from src.exception import CustomException
from src.logger import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import h2o
from h2o.estimators.word2vec import H2OWord2vecEstimator

from src.utils import save_object, evaluate_models

@dataclass
class Word2VecConfig:
    input_df: pd.DataFrame
    word2vec_model: str
    train_new: bool
    epochs: int = 20
    sent_sample_rate: float = 0.0
    save_path: str = "artifacts/"
    input_file: str = None
    prediction_flag: int = 0  # Added prediction_flag

class Word2VecProcessor:
    def __init__(self, config: Word2VecConfig):
        self.config = config
        self.df_comments = None
        self.model = None
        nltk.download('stopwords')
        nltk.download('wordnet')

    def read_data(self):
        if self.config.input_df is not None:
            # Use the DataFrame directly
            if self.config.prediction_flag == 0:
                self.df_comments = self.config.input_df[['Converted', 'Follow Up Comments']]
            else:
                self.df_comments = self.config.input_df[['Follow Up Comments']]
        elif self.config.input_file is not None:
            # Read from file
            df = pd.read_csv(self.config.input_file)
            if 'Converted' in df.columns and self.config.prediction_flag == 0:
                self.df_comments = df[['Converted', 'Follow Up Comments']]
            else:
                self.df_comments = df[['Follow Up Comments']]
        else:
            raise ValueError("No input source (file or DataFrame) specified in configuration")


    def preprocess_comments(self):
        try:
            stemmer = WordNetLemmatizer()
            documents = []
            for comment in self.df_comments['Follow Up Comments']:
                # Preprocess the comment
                document = re.sub(r'\W', ' ', str(comment))
                document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
                document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
                document = re.sub(r'[0-9]+', ' ', document)
                document = re.sub(r'\s+', ' ', document, flags=re.I)
                document = re.sub(r'^b\s+', '', document)
                document = re.sub(r"_", "", document)
                document = document.lower()
                document = " ".join([stemmer.lemmatize(word) for word in document.split() if word not in stopwords.words('english')])
                documents.append(document)

            self.df_comments['Follow_Up_Comments'] = documents
        except Exception as e:
            raise CustomException(e,sys)            

    def tokenize(self, sentences):
        try:
            stop_word = stopwords.words('english')
            tokenized = sentences.tokenize("\\W+")
            tokenized_lower = tokenized.tolower()
            tokenized_filtered = tokenized_lower[(tokenized_lower.nchar() >= 2) | (tokenized_lower.isna()),:]
            tokenized_words = tokenized_filtered[tokenized_filtered.grep("[0-9]",invert=True,output_logical=True),:]
            tokenized_words = tokenized_words[(tokenized_words.isna()) | (~ tokenized_words.isin(stop_word)),:]
            return tokenized_words
        except Exception as e:
            raise CustomException(e,sys)        

    def tokenize_comments(self):
        try:
            df_h2o_comments = h2o.H2OFrame(self.df_comments)
            df_h2o_comments['Follow_Up_Comments'] = df_h2o_comments['Follow_Up_Comments'].ascharacter()
            words = self.tokenize(df_h2o_comments['Follow_Up_Comments'])
            return words
        except Exception as e:
            raise CustomException(e,sys)        

    def train_upload_word2vec(self):
        try:
            self.words = self.tokenize_comments()
            if self.config.train_new:
                logging.info(">>>>> Training new Word2Vec Model <<<<<")
                self.model = H2OWord2vecEstimator(sent_sample_rate=self.config.sent_sample_rate, epochs=self.config.epochs)
                self.model.train(training_frame=self.words)
                h2o.download_model(self.model, path=self.config.word2vec_model)
            else:
                logging.info(">>>>> Uploading existing Word2Vec Model <<<<<")
                # upload the saved model to the H2O cluster
                self.model = h2o.upload_model(self.config.word2vec_model)
        except Exception as e:
            raise CustomException(e,sys)

    def gen_comments_vec_embeddings(self):
        try:
            # Calculate a vector for each comments:
            self.comments_vecs = self.model.transform(self.words, aggregate_method = "AVERAGE")
            # Get Word Embeddings per Word
            unique_words = self.words.asfactor().unique().ascharacter()
            unique_words.col_names = ["Word"]
            self.word_embeddings = self.model.transform(unique_words, aggregate_method="None")
            self.word_embeddings = unique_words.cbind(self.word_embeddings)
            self.word_embeddings = self.word_embeddings[~(self.word_embeddings["C1"].isna())]
            save_file = f"{self.config.save_path}{'word_embeddings'}"
            save_object(save_file, self.word_embeddings)
        except Exception as e:
            raise CustomException(e,sys) 

    def process(self):
        self.read_data()
        self.preprocess_comments()
        self.train_upload_word2vec()
        self.gen_comments_vec_embeddings()

@dataclass
class DataPreparationConfig:
    df_features: pd.DataFrame
    comments_vecs: pd.DataFrame
    seed: int = 555
    split_ratio: float = 0.4
    save_path: str = "artifacts/"


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.df_features = config.df_features
        self.comments_vecs = config.comments_vecs

    def prepare_data(self):
        try:
            columns_to_exclude = ['State', 'Product of Interest', 'Domain', 'Follow Up Comments']
            if 'Converted' in self.df_features.columns:
                columns_to_exclude.append('Converted')
            # Select columns that are not in the exclusion list
            columns_to_fill = [col for col in self.df_features.columns if col not in columns_to_exclude]
            # Replace NaN with 0 in these selected columns
            self.df_features[columns_to_fill] = self.df_features[columns_to_fill].fillna(0)

            # Convert to H2OFrame
            self.df_h2o = h2o.H2OFrame(self.df_features)

            # Convert certain columns to factors
            self.df_h2o["State"] = self.df_h2o["State"].asfactor()
            self.df_h2o['Product of Interest'] = self.df_h2o['Product of Interest'].asfactor()

            # If 'Converted' exists, treat it as a factor for classification
            if 'Converted' in self.df_features.columns:
                self.df_h2o["Converted"] = self.df_h2o["Converted"].asfactor()

            # Add word2vec from comments
            self.df_h2o = self.df_h2o.cbind(self.comments_vecs)

            # Identify predictors and remove unnecessary columns
            #self.predictors = self.identify_predictors(df, id_cols, tgt_col)
            return self.df_h2o
        except Exception as e:
            raise CustomException(e,sys)


    def identify_predictors(self, df, id_cols, tgt_col):
        try:
            predictors = df.columns
            CTA_cols = [col for col in df.columns if col.startswith('CTA')]
            to_remove = CTA_cols + id_cols + tgt_col + ['Follow Up Comments', 'Domain', 'Pref_Contct_0']
            return [e for e in predictors if e not in to_remove]
        except Exception as e:
            raise CustomException(e,sys)  

    def resample_and_split_data(self, df):
        try:
            df_1 = df[df['Converted'] == "1"]
            df_0 = df[df['Converted'] == "0"]
            df_00 = df_0.split_frame(ratios=[self.config.split_ratio], seed=self.config.seed - 1)
            df_upd = df_00[0].rbind(df_1)
            train, valid, test = df_upd.split_frame(
                ratios=[0.7, 0.15], 
                seed=self.config.seed, 
                destination_frames=['train.hex', 'valid.hex', 'test.hex']
            )
            return train, valid, test
        except Exception as e:
            raise CustomException(e,sys)
        
    def save_dataset(self, dataset, dataset_name):
        try:
            save_file = f"{self.config.save_path}{dataset_name}.csv"
            h2o.export_file(dataset, path=save_file, force=True)
        except Exception as e:
            raise Exception(f"Error saving {dataset_name} dataset: {e}")
        
    def prepare_data_ex(self):
        try:
            #the columns list
            id_cols = ['sfdc_lead_id', 'SFDC_Internal_ID', 'Email Address']
            tgt_col = ['Converted']
            reqd_cols = ['State', 'Business Group', 'Product of Interest', 'CTA_ID', 'sfdc_email_opt_out', 'Lead Source', 'Preferred_Contact_Method', 'Equipment_Count', 'Function', 'Main Specialty', 'First_Time_Contact', 'Healthcare_Market_Segment', 'Type', 'Philips_Classification_Level']
            # Convert string NaN to np.nan
            self.df_features["EquipCount"] = self.df_features["EquipCount"].replace(np.nan, 0)

            # Convert to H2OFrame
            df = h2o.H2OFrame(self.df_features)

            # Convert certain columns to factors
            for col in df.columns:
                # Check if the column is not of numeric type (integer or float)
                if not(df[col].isnumeric()[0]):
                    df[col] = df[col].asfactor()
            logging.info(f"df Data Types: {df.types}")
            #df["State"] = df["State"].asfactor()
            #df['Product of Interest'] = df['Product of Interest'].asfactor()
            df['Converted'] = df['Converted'].asfactor()

            # Add word2vec from comments
            df = df.cbind(self.comments_vecs)

            # Identify predictors and remove unnecessary columns
            # Identify predictors and response
            self.predictors = df.columns
            CTA_cols = [col for col in df.columns if col.startswith('CTA')]
            to_remove = tgt_col + CTA_cols + ['Follow Up Comments', 'Domain', 'Pref_Contct_0']
            self.predictors = [e for e in self.predictors if e not in to_remove]
            logging.info(f"Predictors: {self.predictors}")

            # Resample & split into train, validation, and test sets
            train, valid, test = self.resample_and_split_data(df)

            # Save the datasets
            self.save_dataset(train, "train")
            self.save_dataset(valid, "valid")
            self.save_dataset(test, "test")

            # Resample & split into train, validation, and test sets
            return train, valid, test
        except Exception as e:
            raise CustomException(e,sys)

