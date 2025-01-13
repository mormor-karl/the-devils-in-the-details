import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
import sys

#######################################################################

DETAILED_2_GENERAL = {
    'firstname_male': 'personal_name', 'firstname_female': 'personal_name', 'firstname_unknown': 'personal_name', 'initials': 'personal_name', 'middlename': 'personal_name', 'surname': 'personal_name',  # personal name category
    'school': 'institution', 'work': 'institution', 'other_institution': 'institution',  # institution category
    'area': 'geographic', 'city': 'geographic', 'geo': 'geographic', 'country': 'geographic', 'place': 'geographic', 'region': 'geographic', 'street_nr': 'geographic', 'zip_code': 'geographic',  # geographic category
    'transport_name': 'transportation', 'transport_nr': 'transportation',  # transportation category
    'age_digits': 'age', 'age_string': 'age',  # age category
    'date_digits': 'date', 'day': 'date', 'month_digit': 'date', 'month_word': 'date', 'year': 'date',  # date category
    'phone_nr': 'other', 'email': 'other', 'url': 'other', 'personid_nr': 'other', 'account_nr': 'other', 'license_nr': 'other', 'other_nr_seq': 'other', 'extra': 'other', 'prof': 'other', 'edu': 'other', 'fam': 'other', 'sensitive': 'other',  # other category
    'pl': 'pl', 'def': 'def', 'gen': 'gen', 'foreign': 'foreign'  # functional
}  

PSEUDO_TAGS = [
    'firstname_male', 'firstname_female', 'firstname_unknown', 'initials', 'middlename', 'surname',  # personal name category
    'school', 'work', 'other_institution',  # institution category
    'area', 'city', 'geo', 'country', 'place', 'region', 'street_nr', 'zip_code',  # geographic category
    'transport_name', 'transport_nr',  # transportation category
    'age_digits', 'age_string',  # age category
    'date_digits', 'day', 'month_digit', 'month_word', 'year',  # date category
    'phone_nr', 'email', 'url', 'personid_nr', 'account_nr', 'license_nr', 'other_nr_seq', 'extra', 'prof', 'edu', 'fam', 'sensitive',  # other category
    'pl', 'def', 'gen', 'foreign'  # functional
]

#######################################################################

def import_files(path, ending):
    '''A function which imports all the desired files in a given directory (must be Excel format) as DataFrames.
    
    Args:
        path (str): The path to the desired directory.
        ending (str): The ending to tell the files apart from other files by.
    
    Returns:
        A list of tuples in the form DataFrame, filename.
    '''
    all_results = []
    # iterate through all the files in the directory
    for file in tqdm(os.listdir(path), desc='Reading in result files...'):
        # select matches
        if file.endswith(ending):
            all_results.append([pd.read_excel(os.path.join(path, file), index_col=0), file])
            
    return all_results

def merge_results(results, name_ending):
    '''A function which merges all the separate result DataFrames
    
    Args:
        results (list): A list of DataFrame, filename tuples.
        name_ending (str): The ending to tell the files apart from other files by.
    
    Returns:
        A DataFrame containing combined results.
    '''
    # define the initial dataframe
    merged = results[0][0]
    filename = results[0][1]
    # define extra columns
    merged['Is PII'] = (merged['Gold Standard'] != 'O')
    merged['Is Corr Annotated'] = (merged['Corr Annotation'] != np.NaN) & (merged['Corr Annotation'] != 'none')
    # get whether the prediction was correct
    merged[f'Correct_{filename.rstrip(name_ending)}'] = (merged['Gold Standard'] == merged['Prediction'])
    # drop unnecessary columns
    merged.drop(['Original Tag', 'Prediction', 'PII', 'Corr Annotation'], axis=1, inplace=True)
    
    # iterate over the rest
    for i in range(1, len(results)):
        new_addition = results[i][0]
        filename = results[i][1]
        # get whether the prediction was correct
        new_addition[f'Correct_{filename.rstrip(name_ending)}'] = (new_addition['Gold Standard'] == new_addition['Prediction'])
        new_addition.drop(['Gold Standard', 'Prediction', 'Original Tag', 'PII', 'Corr Annotation'], axis=1, inplace=True)
        # merge DataFrames
        merged = merged.merge(new_addition, how='inner', on=['Essay ID', 'Token', 'Context'], suffixes=[None, (' ' + filename.rstrip(name_ending))])
        
    return merged   

def generalize_2_general(val):
    if val != 'O':
        return DETAILED_2_GENERAL[val]
    else:
        return val

def generalize_2_basic(val):
    if val != 'O':
        return 'S'
    else:
        return 'O'

#######################################################################

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('results_path', help='The path to the folder containing the results.')
    parser.add_argument('name_ending', help='The name ending for the files to be used (e.g. _results.xlsx).')
    args = parser.parse_args()
    # import results
    results = import_files(args.results_path, args.name_ending)
    # merge results
    merged_results = merge_results(results, args.name_ending)
    # calculate total correct predictions
    merged_results['Total correct predictions'] = merged_results.drop(columns=["Essay ID", "Token", "Context", "Gold Standard", "Is PII", "Is Corr Annotated"]).sum(axis=1)
    # add additional info
    merged_results['General Tag'] = merged_results['Gold Standard'].apply(generalize_2_general) 
    merged_results['Basic Tag'] = merged_results['Gold Standard'].apply(generalize_2_basic) 
    # export results
    merged_results.to_excel(os.path.join(args.results_path, 'results_from_all_models.xlsx'))
