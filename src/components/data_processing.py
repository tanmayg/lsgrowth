import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
import numpy as np
import seaborn as sns
from src.utils import save_object, concat_non_na, get_source, update_row_flag
import time
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import re
import pickle

@dataclass
class DataProcessingConfig:
    df_cdo: pd.DataFrame = None
    #df_opty: pd.DataFrame = None
    df_st_prov_ref: pd.DataFrame = None
    df_sfdc_acct_nam: pd.DataFrame = None
    df_cdo_ph_raw: pd.DataFrame = None
    df_cdo_acct_final: pd.DataFrame = None
    df_ct: pd.DataFrame = None
    prediction_flag: int = None
    output_file_path: str = os.path.join("artifacts","feature_file.csv")

class DataProcessor:

    def __init__(self, config: DataProcessingConfig):
        self.config = config

    def combined_transform_and_convert(self, code):
        try:
            # First part: Transform the state code
            if '-' in code:
                parts = code.split('-')
                if parts[0] in ['US', 'CA']:
                    transformed_code = parts[1]
                else:
                    transformed_code = code
            else:
                transformed_code = code

            # Second part: Convert to abbreviation using the reference DataFrame
            match = self.config.df_st_prov_ref[self.config.df_st_prov_ref['State'].str.lower() == transformed_code.lower()]
            if not match.empty:
                return match['State_Code'].values[0]
            else:
                return transformed_code.upper()
    
        except Exception as e:
            raise CustomException(e, sys)
    
    def standardize_country_names(self):
        try:
            CA = ["CA", "CANADA", "Ca", "Canada"]
            US = ["U.S.", "U.S. Minor Outlying Islands", "U.S.A.", "UNITED STATES", 
                "UNITED STATES MINOR OUTLYING ISLANDS", "US", "USA", "United States", 
                "United States of America", "Us", "us"]
            self.config.df_cdo["CDO Country"] = self.config.df_cdo["CDO Country"].astype(str)
            #convert string nan to np.nan
            self.config.df_cdo["CDO Country"] = self.config.df_cdo["CDO Country"].replace('nan', np.nan)
            #remove trailing & leading white space
            self.config.df_cdo["CDO Country"] = self.config.df_cdo["CDO Country"].apply(lambda x: x.strip() if isinstance(x, str) else x)

            self.df_cdo_nam = self.config.df_cdo.loc[self.config.df_cdo["CDO Country"].isin(CA + US)]
            self.df_cdo_nam.columns = self.df_cdo_nam.columns.str.strip()
            self.df_cdo_nam["CDO Country"] = np.where(self.df_cdo_nam["CDO Country"].isin(CA), "CA", "US")
            self.df_cdo_nam['Country'] = np.where(self.df_cdo_nam['CDO Country']=='US', 'United States', np.where(self.df_cdo_nam['CDO Country']=='CA', 'Canada', np.nan))
            #convert string nan to np.nan
            self.df_cdo_nam["RecordDate"] = pd.to_datetime(self.df_cdo_nam["Record Date Created"])
            self.df_cdo_nam["sfdc_lead_id"] = self.df_cdo_nam["sfdc_lead_id"].replace('nan', np.nan)
            self.df_cdo_nam['sfdc_lead_id'] = self.df_cdo_nam['sfdc_lead_id'].fillna(self.df_cdo_nam['SFDC Task ID'])

        except Exception as e:
            raise CustomException(e,sys)

    def clean_and_transform_state_province(self):
        try:
            # Implement the cleaning and transforming logic here
            #create date column with 0 time
            self.df_cdo_nam["Record Date Created"] = pd.to_datetime(self.df_cdo_nam["Record Date Created"])
            self.df_cdo_nam["RecordDate"] = pd.to_datetime(self.df_cdo_nam["Record Date Created"].dt.strftime("%Y-%m-%d"))
            #replace NA in sfdc lead id with 'Not Available'
            self.df_cdo_nam['sfdc_lead_id'] = self.df_cdo_nam['sfdc_lead_id'].fillna('Not Available')
            #Clean State/Province data
            self.df_cdo_st_nam = self.df_cdo_nam[['CDO Country', 'CDO State or Province']].drop_duplicates()
            self.df_cdo_st_nam['CDO State or Province'] = self.df_cdo_st_nam['CDO State or Province'].str.replace('.', '', regex=False)
            #convert to abbreviation
            self.df_cdo_st_nam['CDO State or Province'] = self.df_cdo_st_nam['CDO State or Province'].astype(str)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['CDO State or Province'].apply(self.combined_transform_and_convert)
            # split further at space of '-' and check if its a valid code
            def simplified_state_code(row):
                parts = re.split(r'[. -]', row['State_Code'])
                first_part = parts[0]
                second_part_exists = len(parts) > 1 and len(parts[1]) != 2
                is_valid_code = first_part in self.config.df_st_prov_ref['State_Code'].values
                return first_part if is_valid_code and second_part_exists else row['State_Code']
            
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam.apply(simplified_state_code, axis=1)
            # Selective filtering, could be automated in future
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('ASHEVILLE', 'NC', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('CHICAGO', 'IL', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('COLUMBUS OHIO', 'OH', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('CORADO', 'CO', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('CALIFORNIA (CA)', 'CA', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('LOS ANGELES', 'CA', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('MANCHESTER', 'NH', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('MAYAGUEZ', 'PR', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('MONROE', 'LA', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('NELSONVILLE', 'OH', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('OHIOI', 'OH', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('OWOSSO', 'MI', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('PRINCETON', 'NJ', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('QUÉBEC', 'PQ', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('SOUT CAROLINA', 'SC', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('TRAVERSE CITY', 'MI', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('WESTERVILLE', 'OH', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('YORK', 'NY', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('FLA', 'FL', regex=False)
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].str.replace('MASS', 'MA', regex=False)

            # convert invalid State_code to NA
            self.df_cdo_st_nam['State_Code'] = self.df_cdo_st_nam['State_Code'].where(self.df_cdo_st_nam['State_Code'].isin(self.config.df_st_prov_ref['State_Code']), np.nan)
            # Convert to valid state
            #del(self.df_cdo_st_nam['State'])
            self.df_cdo_st_nam = pd.merge(self.df_cdo_st_nam, self.config.df_st_prov_ref[['Country_Code', 'State_Code', 'State']], left_on=['CDO Country', 'State_Code'], right_on=['Country_Code', 'State_Code'], how='left')
            self.df_cdo_st_nam['State'].isnull().sum()
            self.df_cdo_st_nam = self.df_cdo_st_nam.drop_duplicates()
            # update df_cdo_nam with the correct State name if available
            self.df_cdo_nam = pd.merge(self.df_cdo_nam, self.df_cdo_st_nam[['CDO Country', 'CDO State or Province', 'State']], on=['CDO Country', 'CDO State or Province'], how='left')

        except Exception as e:
            raise CustomException(e,sys)

    def concatenate_follow_up_comments(self):
        try:
            # Implement the concatenate follow-up comments logic here
            self.df_comments = self.df_cdo_nam[["Email Address", "RecordDate", "sfdc_lead_id", "Business Group", "Product of Interest", "Follow_Up", "Follow Up Comments", "CTA_ID", "Lead Source"]].drop_duplicates()
            # Group by and  concatenate
            self.df_comments_res = self.df_comments.groupby(["Email Address", "RecordDate", "sfdc_lead_id", "Business Group", "Product of Interest", "CTA_ID", "Lead Source"], dropna=False)[["Follow_Up", "Follow Up Comments"]].apply(lambda x: x.apply(concat_non_na)).reset_index()
            # replace older dataframe with the updated one
            self.df_comments = self.df_comments_res.copy()
            # Group by set of columns and keep non-NA for CTA_ID and remove trailing space from CTA_ID
            self.df_comments_res = self.df_comments.groupby(["Email Address", "RecordDate", "sfdc_lead_id", "Business Group", "Product of Interest", "Lead Source", "Follow_Up", "Follow Up Comments"], dropna=False).apply(get_source).reset_index(name="CTA_ID")
            #replace all string 'nan with np.nan
            self.df_comments_res.replace('nan', np.nan, inplace=True)
        
        except Exception as e:
            raise CustomException(e,sys)

    def compute_time_differences(self):
        try:
            # Implement the compute time differences logic here
            df_email_lead = self.df_comments_res.sort_values(by=['Email Address', 'RecordDate'], ascending=[True, False])
            # Identify duplicate rows based on a set of columns
            is_duplicate = df_email_lead.duplicated(subset=['Email Address', 'Business Group', 'Product of Interest'], keep=False)
            # Split the DataFrame into two based on duplicates
            df_el_dup = df_email_lead[is_duplicate]
            df_el_unq = df_email_lead[~is_duplicate]
            # Remove duplicate leads based on 'Email Address', 'RecordDate', 'Business Group', 'Product of Interest' but different lead_id
            df_el_dup['sfdc_lead_id'] = df_el_dup['sfdc_lead_id'].astype(str)
            df_el_dup.sort_values(by='sfdc_lead_id', inplace=True)
            df_el_dup.drop_duplicates(subset=['Email Address', 'RecordDate', 'Business Group', 'Product of Interest'], keep='first', inplace=True)
            df_el_dup = df_el_dup.sort_values(by=['Email Address', 'RecordDate'], ascending=[True, False])

            #merge the two dataframes
            df_email_lead = pd.concat([df_el_unq, df_el_dup])
            #Update sfdc_id back to Nan where Not Available
            df_email_lead['sfdc_lead_id'] = df_email_lead['sfdc_lead_id'].replace('Not Available', np.NaN)
            # Apply the function to each group
            df_email_lead = df_email_lead.sort_values(by=['Email Address', 'RecordDate'], ascending=[True, True])
            # Identify duplicate rows based on a set of columns
            is_duplicate = df_email_lead.duplicated(subset=['Email Address', 'Business Group', 'Product of Interest'], keep=False)
            # Split the DataFrame into two based on duplicates
            df_el_dup = df_email_lead[is_duplicate]
            df_el_unq = df_email_lead[~is_duplicate]

            # Compute TimeDiff within each group
            df_el_timediff = df_el_dup.copy()
            df_el_timediff['TimeDiff'] = df_el_timediff.groupby(['Email Address', 'Business Group', 'Product of Interest'], dropna=False)['RecordDate'].diff().dt.days
            df_el_timediff['TD_Flag'] = 1
            df_el_timediff['Follow_Up'] = df_el_timediff['Follow_Up'].astype(str) # <=Handle in the beginning
            df_el_timediff['Follow Up Comments'] = df_el_timediff['Follow Up Comments'].astype(str) # <=Handle in the beginning

            #update previous row flag as 0 if current row TimeDiff<30 for each group by using function: update_row_flag
            df_el_timediff = df_el_timediff.groupby(['Email Address', 'Business Group', 'Product of Interest'], dropna=False).apply(update_row_flag)

            #keep only the rows with flag as 1
            df_el_timediff = df_el_timediff.loc[df_el_timediff['TD_Flag']==1,]
            #delete extra columns
            df_el_timediff = df_el_timediff.drop(columns=['TimeDiff', 'TD_Flag'], axis=1)

            #merge with the unique dataframe
            df_email_lead_res = pd.concat([df_el_timediff, df_el_unq])

            # filter df_cdo_nam based on the result and keep updated columns from the df_email_lead_res and rest from df_cdo_nam
            self.df_cdo_nam_upd = self.df_cdo_nam.reset_index(drop=True)
            del(self.df_cdo_nam_upd['Follow_Up'])
            del(self.df_cdo_nam_upd['Follow Up Comments'])
            del(self.df_cdo_nam_upd['sfdc_lead_id'])
            self.df_cdo_nam_upd = self.df_cdo_nam_upd.drop_duplicates(subset=['Email Address', 'RecordDate', 'Business Group', 'Product of Interest', 'Lead Source', 'CTA_ID'])
            self.df_cdo_nam_upd = pd.merge(self.df_cdo_nam_upd, df_email_lead_res, on=['Email Address', 'RecordDate', 'Business Group', 'Product of Interest', 'Lead Source', 'CTA_ID'])
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def assign_opportunity(self):
        # Implement the opportunity assignment logic here
        try:
            #get the opportunity dataset for which lead id is available in cdo
            self.df_opty_nam = self.config.df_opty.loc[self.config.df_opty['Lead ID'].isin(list(self.df_cdo_nam_upd['sfdc_lead_id'].str.slice(0, 15))), ['Lead ID', 'Lead ID (18 digit)', 'Lead Creation Date', 'Opportunity ID']]
            #get the cdo dataset for which lead id/opportunity data is available in E2E dashboard
            df_cdo_nam_opty = self.df_cdo_nam_upd[self.df_cdo_nam_upd['sfdc_lead_id'].str.slice(0, 15).isin(list(self.config.df_opty['Lead ID']))]
            # create the dataset with email, lead_id and lead creation date
            df_cdo_nam_opty['sfdc_lead_id_15'] = df_cdo_nam_opty['sfdc_lead_id'].str.slice(0, 15)
            self.df_ctp_nam = pd.merge(df_cdo_nam_opty[['Email Address', 'RecordDate', 'sfdc_lead_id', 'sfdc_lead_id_15']], self.df_opty_nam, left_on='sfdc_lead_id_15', right_on='Lead ID')
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def conditional_fuzzy_parallel(self, row):
        try:
            '''
            The function iterates through the rows of the input DataFrame 'df' and performs conditional fuzzy matching based on
            the 'State', 'Country', and 'All' scenarios. It uses the 'fuzz.token_sort_ratio' function to calculate matching scores
            and populates the 'Company_fuzzy', 'Company_fuzzyScore', 'Company_fuzzyFlag', and 'Populated_ID' columns in 'df'
            with the matching results. Rows with matching scores below the specified threshold ('fuzzyTh') are not considered.
            '''
            
            # Extract the Series from the tuple
            row = row[1]
            # Initialize the dictionary to store updated values
            updated_values = {}
            # Initialize the best score to 0
            best_score = 0
            
            update_row = False
            if row['State']:
                df_dom_sfdc_st = self.config.df_sfdc_acct_nam[self.config.df_sfdc_acct_nam.Region_Name==row['State']]
                try:
                    best_match, best_score = process.extractOne(row['CDO Company'], list(df_dom_sfdc_st['Name']), scorer=fuzz.token_sort_ratio)
                    if best_score > 85:
                        acct_id_ica = df_dom_sfdc_st[df_dom_sfdc_st['Name']==best_match]['ICA_ID'].values[0]
                        acct_id_sii = df_dom_sfdc_st[df_dom_sfdc_st['Name']==best_match]['SFDC_Internal_ID'].values[0]
                        fuzzyFlag = 'State'
                        update_row = True
                except:
                    best_score = 0
            
            if not update_row and row['Country']:
                df_dom_sfdc_cy = self.config.df_sfdc_acct_nam[self.config.df_sfdc_acct_nam.Country==row['Country']]
                try:
                    best_match, best_score = process.extractOne(row['CDO Company'], list(df_dom_sfdc_cy['Name']), scorer=fuzz.token_sort_ratio)
                    if best_score > 90:
                        acct_id_ica = df_dom_sfdc_cy[df_dom_sfdc_cy['Name']==best_match]['ICA_ID'].values[0]
                        acct_id_sii = df_dom_sfdc_cy[df_dom_sfdc_cy['Name']==best_match]['SFDC_Internal_ID'].values[0]
                        fuzzyFlag = 'Country'
                        update_row = True
                except:
                    best_score = 0
            
            if not update_row:
                try:
                    best_match, best_score = process.extractOne(row['CDO Company'], list(self.config.df_sfdc_acct_nam['Name']), scorer=fuzz.token_sort_ratio)
                    if best_score > 90:
                        acct_id_ica = self.config.df_sfdc_acct_nam[self.config.df_sfdc_acct_nam['Name']==best_match]['ICA_ID'].values[0]
                        acct_id_sii = self.config.df_sfdc_acct_nam[self.config.df_sfdc_acct_nam['Name']==best_match]['SFDC_Internal_ID'].values[0]
                        fuzzyFlag = 'All'
                        update_row = True
                except:
                    best_score = 0
            
            if update_row:
                updated_values['Company_fuzzy'] = best_match
                updated_values['Company_fuzzyScore'] = best_score
                updated_values['Company_fuzzyFlag'] = fuzzyFlag
                updated_values['ICA_ID'] = acct_id_ica
                updated_values['SFDC_Internal_ID'] = acct_id_sii
            else:
                # Assign NaN to multiple columns in the dictionary
                updated_values['Company_fuzzy'] = np.nan
                updated_values['Company_fuzzyScore'] = np.nan
                updated_values['Company_fuzzyFlag'] = np.nan
                updated_values['ICA_ID'] = np.nan
                updated_values['SFDC_Internal_ID'] = np.nan

            # Update the row with the dictionary of updated values
            for key, value in updated_values.items():
                row[key] = value
            return row

        except Exception as e:
                raise CustomException(e,sys)
        
    def fuzzy_match_preptrain(self):
        try:
            logging.info("Entered the fuzzy_match_preptrain method or component")
            # Implement the parallel fuzzy matching logic here
            #Create master frame using email & company details
            df_email_cmp = self.df_cdo_nam[['Email Address', 'CDO Company', 'CDO Country', 'CDO State or Province', 'State', 'sfdc_lead_id']].drop_duplicates()
            # Updated CDO dataframe with all the transformations
            self.df_cdo_final = self.df_cdo_nam_upd[self.df_cdo_nam_upd.sfdc_lead_id.str.slice(0, 15).isin(list(self.df_opty_nam['Lead ID']))]
            #get the Opp ID column
            self.df_cdo_final = pd.merge(self.df_cdo_final, self.df_ctp_nam[['sfdc_lead_id', 'Opportunity ID']], on='sfdc_lead_id')
            # Subset of master data frame for lead_id having a valid opportunity id
            self.df_cmp_only_opp = df_email_cmp[df_email_cmp.sfdc_lead_id.isin(list(self.df_cdo_final[self.df_cdo_final['Opportunity ID']!='-']['sfdc_lead_id']))]
            # Subset of master data frame for lead_id having an invalid opportunity id
            self.df_cmp_non_opp = df_email_cmp[df_email_cmp.sfdc_lead_id.isin(list(self.df_cdo_final[self.df_cdo_final['Opportunity ID']=='-']['sfdc_lead_id']))]
        
        except Exception as e:
                raise CustomException(e,sys)
        
    def fuzzy_match_train(self):
        try:
            # === Fuzzy Matching for opportuity data === #
            # create pickle file to be used as input
            dataframes = {'df1': self.df_cmp_only_opp, 'df2':self.config.df_sfdc_acct_nam}
            # Save the dictionary containing DataFrames to a pickle file
            with open('./Data/company_fuzzy_input_only_opp.pkl', 'wb') as file:
                pickle.dump(dataframes, file)
            # the fuzzy operation is executed using another script with paralle processing
            # the Python script to execute
            script1_path = 'SGLS_CmpnyFuzzy_Pll_Only_Opp.py'
            # trigger the parallel script
            try:
                subprocess.run(['python', script1_path])
            except Exception as e:
                raise CustomException(e,sys)
            #Load the dictionary of DataFrames from the pickle file
            with open('./Data/company_fuzzy_output_only_opp.pkl', 'rb') as file:
                loaded_dataframes = pickle.load(file)
            # Access the DataFrames by their keys
            df_fuzz_opp = loaded_dataframes['df1']
            # === Repeat the last step for non-opportunity data as well === #
            # create pickle file to be used as input
            dataframes = {'df1': self.df_cmp_non_opp, 'df2':self.config.df_sfdc_acct_nam}
            # Save the dictionary containing DataFrames to a pickle file
            with open('./Data/company_fuzzy_input_non_opp.pkl', 'wb') as file:
                pickle.dump(dataframes, file)
            # the Python script to execute
            script2_path = 'SGLS_CmpnyFuzzy_Pll_Non_Opp.py'
            # trigger the parallel script
            try:
                subprocess.run(['python', script2_path])
            except Exception as e:
                raise CustomException(e,sys)
            #Load the dictionary of DataFrames from the pickle file
            with open('./Data/company_fuzzy_output_non_opp.pkl', 'rb') as file:
                loaded_dataframes = pickle.load(file)
            # Access the DataFrames by their keys
            df_fuzz_non_opp = loaded_dataframes['df1']
            #merge the two dataframes
            self.config.df_cdo_acct_final = pd.concat([df_fuzz_opp, df_fuzz_non_opp])
            #Add lead creation date
            self.config.df_cdo_acct_final = pd.merge(self.config.df_cdo_acct_final, self.df_cdo_final[['sfdc_lead_id', 'RecordDate']])

        except Exception as e:
            raise CustomException(e,sys)
        
    def purchase_history(self):
        try:
            # Updated CDO dataframe with all the transformations
            if self.config.prediction_flag == 0:
                self.df_cdo_final = self.df_cdo_nam_upd[self.df_cdo_nam_upd.sfdc_lead_id.str.slice(0, 15).isin(list(self.df_opty_nam['Lead ID']))]
                #get the Opp ID column
                self.df_cdo_final = pd.merge(self.df_cdo_final, self.df_ctp_nam[['sfdc_lead_id', 'Opportunity ID']], on='sfdc_lead_id')
            else:
                self.df_cdo_final = self.df_cdo_nam_upd.copy()
            #get the accounts where at least one of SFDC_Internal_ID & ICA_ID are not null
            df_cdo_acct_for_ph = self.config.df_cdo_acct_final[~((self.config.df_cdo_acct_final.ICA_ID.isnull()) & (self.config.df_cdo_acct_final.SFDC_Internal_ID.isnull()))]
            df_cdo_acct_for_ph = pd.merge(self.df_cdo_final, df_cdo_acct_for_ph[['CDO Company', 'Country', 'State', 'Company_fuzzy', 'Company_fuzzyScore', 'Company_fuzzyFlag', 'ICA_ID', 'SFDC_Internal_ID']], on=['CDO Company', 'Country', 'State'])
            # Merge the two dataframes on 'SFDC_Internal_ID'
            merged_df = df_cdo_acct_for_ph.merge(self.config.df_cdo_ph_raw, on='SFDC_Internal_ID', how='left')
            # Filter rows where 'Technical_start_Date' is less than 'RecordDate'
            filtered_df = merged_df[merged_df['Technical_start_Date'] < merged_df['RecordDate']]
            # Group by 'sfdc_lead_id' and 'RecordDate' and count unique 'Equipment_ID'
            self.df_cdo_prch_hist = filtered_df.groupby(['sfdc_lead_id', 'RecordDate'])['Equipment_ID'].nunique().reset_index()
            # Rename the column to 'EquipCount'
            self.df_cdo_prch_hist = self.df_cdo_prch_hist.rename(columns={'Equipment_ID': 'EquipCount'})
            # add account related details
            self.df_acct_demog = pd.merge(df_cdo_acct_for_ph[['sfdc_lead_id', 'SFDC_Internal_ID']], self.config.df_sfdc_acct_nam[['SFDC_Internal_ID', 'Type']])
            # Merge the purchase history dataset
            self.df_cdo_final = pd.merge(self.df_cdo_final, self.df_cdo_prch_hist[['sfdc_lead_id', 'EquipCount']], on='sfdc_lead_id', how='left')
            #Merge with contacts data
            self.df_cdo_final = pd.merge(self.df_cdo_final, self.config.df_ct[['Email Address', 'Function', 'Main Specialty', 'Main Specialty Code', 'Date Created', 'SFDC EmailOptOut']], how='left')
            #merge with demog data
            self.df_cdo_final = self.df_cdo_final.merge(self.df_acct_demog, how='left')
        
        except Exception as e:
            raise CustomException(e,sys)
        
    # Define a function to clean the column values
    def clean_column(self, column):
        try:
            # Step 0: Convert to lower case
            column = column.str.lower()
            # Step 1: Remove all special characters except for forward slash '/' and white space(' ')
            column = column.str.replace(r'[^\w\s/]', '', regex=True)
            # Step 2: Replace forward slash and space with underscore
            column = column.str.replace(r'[ /]', '_', regex=True)
            # Step 3: Replace consecutive underscores with a single underscore
            column = column.str.replace(r'_+', '_', regex=True)
            # Remove trailing underscores
            column = column.str.rstrip('_')
            return column
    
        except Exception as e:
            raise CustomException(e,sys)
                
    def custom_clean(self):
        try:
            # ============= Custom Clean: Function, BG & Main Speciality ===============#
            df_bg = self.df_cdo_final[['sfdc_lead_id', 'Business Group', 'Product of Interest']]
            df_fn = self.df_cdo_final[['sfdc_lead_id', 'Function', 'Main Specialty']]
            df_fn['Function'] = self.clean_column(df_fn['Function'])
            #============== Custom Clean: Function ===========#
            c_suite = ['ceo_coo_president', 'chief_executive_officer_president', 'chief_information_officer', 'chief_marketing_officer', 'chief_financial_officer', 'head_of_administration_cfo', 'civil_servant', 'chief_medical_officer', 'chief_nursing_officer', 'ceo_president', 'owner']
            df_fn['Function'] = df_fn['Function'].replace(c_suite, 'c_suite', regex=False, inplace=False)
            hod = ['clinical_director_chief_surgeon', 'director_head_of_department', 'head_of_department', 'medical_director', 'director_jefe_de_departamento', 'director_manager', 'manager', 'product_manager', 'director']
            df_fn['Function'] = df_fn['Function'].replace(hod, 'hod', regex=False, inplace=False)
            specialist = ['doctor_physician_specialist', 'dentist', 'surgeon', 'scientist_research', 'cardiologist', 'specialist', 'radiologist', 'general_practitioner', 'physicist', 'sonographer', 'physician', 'doctor_physician_specialist', 'médico_especialista', 'respiratory_therapist', 'scientist', 'medical_radiation_technologist']
            df_fn['Function'] = df_fn['Function'].replace(specialist, 'specialist', regex=False, inplace=False)
            it_tech = ['engineer', 'engineer_technician', 'it_manager', 'it_technologist', 'technician', 'technologist', 'biomedical_engineer']
            df_fn['Function'] = df_fn['Function'].replace(it_tech, 'engr_it_techn', regex=False, inplace=False)
            mkt = ['marketing_media_sales', 'field_sales_employee']
            df_fn['Function'] = df_fn['Function'].replace(mkt, 'market_media_sales', regex=False, inplace=False)
            nurse = ['nurse', 'nurse_mid_wife', 'assistant']
            df_fn['Function'] = df_fn['Function'].replace(nurse, 'nurse_midwife', regex=False, inplace=False)
            arch = ['architects_designers', 'architect']
            df_fn['Function'] = df_fn['Function'].replace(arch, 'architects', regex=False, inplace=False)
            others = ['student', 'legal', 'staff', 'lpm', 'lcom', 'commercial_consultant', 'supervisor', 'infirmière_sagefemme', 'enfermeira_parteira']
            df_fn['Function'] = df_fn['Function'].replace(others, 'others', regex=False, inplace=False)
            #============== Custom Clean: BG ===========#
            df_bg['Business Group'] = self.clean_column(df_bg['Business Group'])
            tbl_bg = pd.DataFrame(df_bg['Business Group'].value_counts()).reset_index()
            hpm = ['monitoring_and_analytics', 'hospital_patient_monitoring', 'patient_care_monitoring_solutions']
            df_bg['Business Group'] = df_bg['Business Group'].replace(hpm, 'hospital_patient_monitoring', regex=False, inplace=False)
            hsdp = ['bg_hsdp_hts_and_em', 'hsdp_hts_and_em']
            df_bg['Business Group'] = df_bg['Business Group'].replace(hsdp, 'hsdp_hts_and_em', regex=False, inplace=False)
            dim = ['diagnostic_imaging', 'mr_dxr_oem', 'mr_oem']
            df_bg['Business Group'] = df_bg['Business Group'].replace(dim, 'mr_dxr_oem', regex=False, inplace=False)
            igt = ['igt_systems', 'image_guided_therapy']
            df_bg['Business Group'] = df_bg['Business Group'].replace(igt, 'igt_systems', regex=False, inplace=False)
            phm = ['population_health_management', 'pers_emergency_resp_and_sen_living']
            df_bg['Business Group'] = df_bg['Business Group'].replace(phm, 'pers_emergency_resp_and_sen_living', regex=False, inplace=False)
            emr = ['therapeutic_care', 'theraputic_care', 'emergency_care']
            df_bg['Business Group'] = df_bg['Business Group'].replace(emr, 'emergency_care', regex=False, inplace=False)
            edi = ['enterprise_diagnostic_informatics', 'healthcare_informatics']
            df_bg['Business Group'] = df_bg['Business Group'].replace(edi, 'enterprise_diagnostic_informatics', regex=False, inplace=False)
            prd = ['precision_diagnosis_other', 'precision_diagnosis_solutions']
            df_bg['Business Group'] = df_bg['Business Group'].replace(prd, 'precision_diagnosis', regex=False, inplace=False)
            others = ['epd_solutions', 'hcother', 'ph_unallocated', 'cc_informatics', 'diagnostic_and_pathway_informatics']
            df_bg['Business Group'] = df_bg['Business Group'].replace(others, 'others', regex=False, inplace=False)
            #============= Custom Clean: Main Specialty ==============#
            df_fn['Main Specialty'] = self.clean_column(df_fn['Main Specialty'])
            tbl_ms = pd.DataFrame(df_fn['Main Specialty'].value_counts()).reset_index()
            emergency = ['emergency_care_amp_urgent_care', 'emergency_care_urgent_care', 'emergency_care', 'critical_care', 'emergency_carehealthcare_management']
            operations = ['hospital_operations', 'hospitaloperations', 'healthcare_consulting', 'healthcare_management']
            health_info = ['health_informatics']
            care = ['acute_care', 'ambulatory_care', 'dental_care', 'rural_care', 'postacute_care', 'home_care', 'mother_and_child_care']
            speciality = ['radiology', 'cardiology', 'oncology', 'pathology', 'healthcare_managementradiology', 'pediatrics', 'surgery', 'anesthesia', 'image_guided_therapy_radiography_ultrasound', 'computed_tomography_radiography_fluoroscopy', 'ultrasound']
            women = ['women_s_health_maternal_health', 'womens_health_amp_maternal_health', 'womens_health_maternal_health', 'womens_healthcare']
            sleep_resp = ['sleep_therapy', 'respiratory_care']
            neuro_vas = ['neurology_amp_neurovascular', 'neurology_neurovascular', 'peripheral_vascular', 'vascular']
            med = ['biomedicine', 'general_medicine']
            df_fn['Main Specialty'] = df_fn['Main Specialty'].replace(emergency, 'emergency_care', regex=False, inplace=False)
            df_fn['Main Specialty'] = df_fn['Main Specialty'].replace(operations, 'operations', regex=False, inplace=False)
            df_fn['Main Specialty'] = df_fn['Main Specialty'].replace(care, 'med_care', regex=False, inplace=False)
            df_fn['Main Specialty'] = df_fn['Main Specialty'].replace(speciality, 'med_speciality', regex=False, inplace=False)
            df_fn['Main Specialty'] = df_fn['Main Specialty'].replace(women, 'women_health', regex=False, inplace=False)
            df_fn['Main Specialty'] = df_fn['Main Specialty'].replace(sleep_resp, 'sleep_resp', regex=False, inplace=False)
            df_fn['Main Specialty'] = df_fn['Main Specialty'].replace(neuro_vas, 'neuro_vascular', regex=False, inplace=False)
            df_fn['Main Specialty'] = df_fn['Main Specialty'].replace(med, 'medicine', regex=False, inplace=False)
            df_fn['Main Specialty'] = df_fn['Main Specialty'].replace(health_info, 'health_informatics', regex=False, inplace=False)
            #============= Custom Clean: 'Product of Interest' ==============#
            df_bg['Product of Interest'] = self.clean_column(df_bg['Product of Interest'])
            #drop existing columns
            cols_del = ['Main Specialty', 'Function', 'Business Group', 'Product of Interest']
            self.df_cdo_final = self.df_cdo_final.drop([col for col in cols_del if col in self.df_cdo_final], axis=1)
            #add the cleaned data columns
            self.df_cdo_final = pd.merge(self.df_cdo_final, df_bg)
            self.df_cdo_final = pd.merge(self.df_cdo_final, df_fn, how='left', on='sfdc_lead_id')
            #create first time contact flag
            self.df_cdo_final['First_Time_Contact'] = np.where(self.df_cdo_final['Date Created'] < self.df_cdo_final['RecordDate'], 0, 1)
            #create personal domain flag column
            personal_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'comcast.net', 'outlook.com', 'msn.com', 'icloud.com', 'verizon.net', 'live.com', 'att.net', 'me.com']
            self.df_cdo_final['Domain'] = self.df_cdo_final['Email Address'].str.split('@').str[1]
            self.df_cdo_final['Personal_Domain'] = np.where(self.df_cdo_final['Domain'].isin(personal_domains), 1, 0)
            logging.info(f"Shape of df_cdo_final: {self.df_cdo_final.shape}")
            #============ create target column ==============#
            if self.config.prediction_flag == 0:
                self.df_cdo_final['Converted'] = self.df_cdo_final['Opportunity ID'].replace('-', 0)
                self.df_cdo_final['Converted'] = np.where(self.df_cdo_final['Converted']!=0, 1, self.df_cdo_final['Converted'])
        except Exception as e:
            raise CustomException(e,sys)
        
    def prep_final_data(self):
        try:
            #the columns list
            id_cols = ['sfdc_lead_id', 'Email Address', 'RecordDate', 'SFDC_Internal_ID']
            reqd_cols = ['State', 'Business Group', 'Product of Interest', 'CTA_ID', 'sfdc_email_opt_out', 'Lead Source', 'Preferred_Contact_Method', 'EquipCount', 'Function', 'Main Specialty', 'First_Time_Contact', 'Type', 'Domain', 'Personal_Domain', 'Follow Up Comments']
            extra_cols = ['CDO Company', 'CDO Country', 'CDO State or Province', 'CDO Zip or Postal Code', 'CDO City', 'Page URL', 'Event_Name', 'Event_Source', 'Form Session ID', 'Business Group Code', 'Form_Name', 'Referrer_Domain_Core', 'Referrer_Domain', 'Referrer', 'Best_Time_to_Call', 'Commercial Product Group Name', 'Contact Type', 'Preferred_Contact_Method']
            tgt_col = ['Converted'] if self.config.prediction_flag == 0 else []  # Include 'Converted' only if Prediction_flag is 0
            # Adjust the dataset based on Prediction_flag
            final_cols = id_cols + reqd_cols + tgt_col
            self.df_cdo_final = self.df_cdo_final[final_cols]

            # Creating a dummy variable for some of the categorical variables and dropping the NA one.
            #Type
            dummy_Type = pd.get_dummies(self.df_cdo_final['Type'], dtype=int, prefix='Type')
            #dummy_Type = dummy_Type.drop('Type_NotAvailable', axis=1)
            #CTA_ID
            dummy_cta = pd.get_dummies(self.df_cdo_final['CTA_ID'], dtype=int, prefix='CTA')
            #Business Group
            dummy_BG = pd.get_dummies(self.df_cdo_final['Business Group'], dtype=int, prefix='BG')
            #Lead Source
            dummy_lead_source = pd.get_dummies(self.df_cdo_final['Lead Source'], dtype=int, prefix='Lead')
            #Preferred_Contact_Method
            dummy_pref_contact = pd.get_dummies(self.df_cdo_final['Preferred_Contact_Method'], dtype=int, prefix='Pref_Contct')
            #dummy_pref_contact = dummy_pref_contact.drop('Pref_Contct_NotAvailable', axis=1)
            #Function
            dummy_function = pd.get_dummies(self.df_cdo_final['Function'], dtype=int, prefix='Function')

            #Main Specialty
            #df_lsm.rename(columns = {'Main Specialty':'MainSplty'}, inplace = True)
            dummy_mainsplty = pd.get_dummies(self.df_cdo_final['Main Specialty'], dtype=int, prefix='MainSp')

            #Features dataset
            feature_cols = ['EquipCount', 'First_Time_Contact', 'State', 'Product of Interest', 'Domain', 'Personal_Domain', 'Follow Up Comments', 'RecordDate'] + tgt_col
            self.df_features = self.df_cdo_final[feature_cols]
            # Concatenate dummy variables
            dummy_frames = [dummy_Type, dummy_cta, dummy_BG, dummy_lead_source, dummy_pref_contact, dummy_function, dummy_mainsplty]
            self.df_features = pd.concat([self.df_features] + dummy_frames, join='inner', axis=1)

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.output_file_path), exist_ok=True)
            # df_cdo_final is the final DataFrame to be saved
            self.df_features.to_csv(self.config.output_file_path, index=False)
            logging.info(f"DataFrame saved to: {self.config.output_file_path}")

        except Exception as e:
            raise CustomException(e,sys)

    def process_data(self):
        self.standardize_country_names()
        self.clean_and_transform_state_province()
        self.concatenate_follow_up_comments()
        self.compute_time_differences()
        #self.assign_opportunity()
        self.fuzzy_match_preptrain()
        self.purchase_history()
        self.custom_clean()
        self.prep_final_data()

    def process_pred_data(self):
        self.standardize_country_names()
        self.clean_and_transform_state_province()
        self.concatenate_follow_up_comments()
        self.compute_time_differences()
        #self.assign_opportunity()
        self.purchase_history()
        self.custom_clean()
        self.prep_final_data()

