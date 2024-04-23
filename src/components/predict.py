import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from dataclasses import dataclass, field
import h2o
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import re
import sys
from h2o.utils.model_utils import reset_model_threshold

@dataclass
class LeadModelConfig:
    input_df: pd.DataFrame
    lead_model: str
    word_emb: pd.DataFrame
    nrows: int = 5000
    prediction_flag: int = 0

class LeadModelPrediction:
    def __init__(self, config: LeadModelConfig):
        self.config = config
        if self.config.input_df.nrows < self.config.nrows:
            self.config.nrows = self.config.input_df.nrows


    def predict(self):
        try:
            #upload the model
            lead_model = h2o.upload_model(self.config.lead_model)
            # Retrieve your default threshold:
            original_threshold = lead_model.default_threshold()
            new_threshold = 0.29
            old_returned = reset_model_threshold(lead_model, new_threshold)
            lead_updated_model = h2o.get_model(lead_model.model_id)
            reset_threshold = lead_updated_model.default_threshold()
            print("New.Threshold:", reset_threshold)

            # If in prediction mode, ensure 'Converted' column is not expected in the input data
            if self.config.prediction_flag == 1 and 'Converted' in self.config.input_df.columns:
                self.config.input_df = self.config.input_df.drop(['Converted'], axis=1)
            else:
                self.config.input_df = self.config.input_df

            #make predictions
            self.predictions = lead_updated_model.predict(self.config.input_df[:self.config.nrows, :])
            #compute contributions
            self.contributions = lead_updated_model.predict_contributions(self.config.input_df[:self.config.nrows, :])
        except Exception as e:
            raise CustomException(e,sys)

    #function to highlight the contributing words from a comment
    def transform_case(self, filtered_words, word):
        return word.upper() if word.lower() in filtered_words else word.lower()

    # Function to get top N words based on absolute score for a given column
    def top_n_words(self, comment_we, column, n=5):
        try:
            top = comment_we.nlargest(n, f'{column}', 'all')['Word'].tolist()
            print(column, top)
            return top
        except Exception as e:
            raise CustomException(e,sys)
    
    def tokenize_comment(self, X):
        try:
            documents = []
            stemmer = WordNetLemmatizer()
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(X))
            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
            # Remove numbers
            document = re.sub(r'[0-9]+', ' ', document) 
            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)
            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)
            # Remove underscores
            document = re.sub(r"_", "", document)
            # Converting to Lowercase
            document = document.lower()
            # Lemmatization
            document = document.split()
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            document = document.encode("ascii", "ignore").decode()
            stop = stopwords.words('english')
            tokens = word_tokenize(document)
            tokens = [w for w in tokens if w not in stop]
            return tokens
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_comments_highlight_multiple(self):
        try:
            # List to hold data for each row
            rows_data = []
            for row_id in list(range(self.config.nrows)):
                logging.info(f"============ ROW: {row_id} ============")
                # Conditional handling for 'Converted' column
                if 'Converted' in self.config.input_df.columns:
                    print("\n\nActual vs Predicted")
                    print(self.config.input_df[row_id, ['Converted']].cbind(self.predictions[row_id,:]))
                    logging.info(f"Actual: {self.config.input_df[row_id, ['Converted']]}")
                else:
                    print("\n\nPredicted")
                    logging.info("Actual column 'Converted' not available in prediction mode.")
                
                print(self.predictions[row_id,:])
                logging.info(f"Predicted: {self.predictions[row_id, 0]} | P0: {self.predictions[row_id, 1]} | P1: {self.predictions[row_id, 2]}")


                # Create an H2O frame for demonstration purposes
                h2o_frame = self.contributions[row_id,:]
                # Get the values from the H2O frame
                values = h2o.as_list(h2o_frame).iloc[0]
                # Create a list of tuples containing absolute values and column names
                abs_value_tuples = [(abs(value), col) for col, value in values.items()]
                # Sort the list of tuples based on absolute values in descending order
                abs_value_tuples.sort(reverse=True)
                #get the list of top 10 contributors
                top_10_contribs = [col for _, col in abs_value_tuples][:10]
                print("Top 10 Contributors:", top_10_contribs)
                top_10_non_text = [col for col in top_10_contribs if not (col.startswith('C') and col[1:].isdigit())]
                print("# of Non-Text Contributors:", len(top_10_non_text))
                print("Non-Text Contributors:", top_10_non_text)
                logging.info(f"Top Contributors (Non-Text): {top_10_non_text}")
                top_10_text = [col for col in top_10_contribs if (col.startswith('C') and col[1:].isdigit())]
                print("Count of Text Contributors:", len(top_10_text))
                # Filter columns starting with 'C' followed by an integer from the top 10 values
                top_10_columns_filtered = [col for _, col in abs_value_tuples if col.startswith('C') and col[1:].isdigit()]
                top_10_columns_filtered = top_10_columns_filtered[:10]
                # Display the result
                print("\nFiltered text vectors of top 10 absolute values:")
                print(top_10_columns_filtered)
                #tokenize the comment
                comment_token = self.tokenize_comment(self.config.input_df[row_id, 'Follow Up Comments'])
                word_emb_df = self.config.word_emb.as_data_frame()
                comment_we = word_emb_df.loc[word_emb_df["Word"].isin(comment_token), ["Word"] + top_10_columns_filtered]
                #print(comment_we)
                #get top N words based on absolute score for a given column
                top_all = []
                for col in top_10_columns_filtered:
                    comment_we[col] = comment_we[col].abs()
                    top_all+=self.top_n_words(comment_we, col)
                #print(top_all)
                # Count the occurrences of each word
                word_counts = Counter(top_all)
                # Sort words by frequency in descending order
                sorted_words_by_frequency = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                filtered_words = [word for word, frequency in word_counts.items() if frequency >= 2]
                print(filtered_words)
                # Convert the text to a list of words
                words = self.config.input_df[row_id, 'Follow Up Comments'].split()
                transformed_words = [self.transform_case(filtered_words, word) for word in words]
                transformed_text = " ".join(transformed_words)
                print("\nOriginal text:", self.config.input_df[row_id, 'Follow Up Comments'])
                logging.info(f"User Comment: {self.config.input_df[row_id, 'Follow Up Comments']}")
                logging.info(f"Contributing words (highlighted in upper case): {transformed_text}")
                print("\nTransformed text:", transformed_text)
                row_data = {
                    'predict': self.predictions[row_id, 0],
                    'p0': self.predictions[row_id, 1],
                    'p1': self.predictions[row_id, 2],
                    '# of Non-Text Contributors': len(top_10_non_text),
                    'Non-Text Contributors': top_10_non_text,
                    '# of Text Contributors': len(top_10_text),
                    'Original text': self.config.input_df[row_id, 'Follow Up Comments'],
                    'Transformed text': transformed_text
                }

                # Conditionally add 'Converted' to row_data if available
                if 'Converted' in self.config.input_df.columns:
                    row_data['Converted'] = self.config.input_df[row_id, 'Converted']

                rows_data.append(row_data)
            # Convert the list of dictionaries into a DataFrame
            result_df = pd.DataFrame(rows_data)
            logging.info(f"{result_df}")
            return result_df
        except Exception as e:
            raise CustomException(e,sys)
    
    def process_predict(self):
        self.predict()
        highlighted_df = self.get_comments_highlight_multiple()
        return highlighted_df