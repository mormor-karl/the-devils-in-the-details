import pandas as pd
import argparse
import os
from tqdm import tqdm
import sys

parent_dir = os.path.dirname('../src/')  # so that we can access utils.py
sys.path.append(parent_dir)
from utils import *
#######################################################################

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

def extract_corr_annotated(document):
    '''A function that extracts all the elements from SweLL-gold with their correction annotation.
    
    Args:
        document (list): A single SweLL-gold document.
    
    Returns:
        A list of lists where every element contains information about the token and its correction annotation.
    '''
    # retrieve the source and the edges from the Svala graph
    source = document['svala_graph']['source']
    edges = document['svala_graph']['edges']

    # reformat the edges to be accessible using the source token IDs
    reformatted_edges = {}
    
    for k, v in edges.items():
        for id in v['ids']:
            if id[0] == "s":
                reformatted_edges[id] = v['labels']
    
    assert len(source) == len(reformatted_edges)
    
    # get the tokenized text
    tokens = [source_dict['text'].strip() for source_dict in source]
    
    # extract the tokens which are tagged both as PII and as containing some error
    corr_annotated = []
    
    for i, source_dict in enumerate(source):
        id = source_dict['id'] 
        pii = 'none'
        corr_ann = 'none'
        
        for element in reformatted_edges[id]:
            if element in PSEUDO_TAGS:
                pii = element
            if element not in PSEUDO_TAGS and not element.isnumeric():
                corr_ann = element   
                 
        # get context
        if i >= 5:
            preceding = tokens[i-5:i]
        elif i != 0:
            preceding = tokens[:i]
        else:
            preceding = ''

        if i != len(tokens)-5:
            succeeding = tokens[i+1:i+6]
        elif i != len(tokens):
            succeeding = tokens[i+1:]
        else:
            succeeding = ''   
            
        context = ' '.join([' '.join(preceding), source_dict['text'].strip(), ' '.join(succeeding)])

        corr_annotated.append([document['id'], source_dict['text'].strip(), context, pii, corr_ann])
            
    return corr_annotated

def test_outputs(corr_annotated, results):
    '''A function that compares data with correction annotation with predictions and returns the ratio of the
    elements that are correction annotated and misclassified to all the elements that are correction annotated.
    
    Args:
        corr_annotated (DataFrame): A DataFrame containing information about tokens' correction annotation.
        results (DataFrame): A DataFrame containing results of prediction.
    
    Returns:
        A float representing the ratio of the elements that are correction annotated and misclassified to all the 
        elements that are correction annotated.
    '''
    # sorting results
    results.sort_values(by='Essay ID', inplace=True)
    
    # inner merging the two to get the overlap
    inter = pd.merge(results, corr_annotated, how='inner', on=['Essay ID', 'Token', 'Context'])
    results_plus = pd.merge(results, corr_annotated, how='left', on=['Essay ID', 'Token', 'Context'])
    only_pilot = results[~(results['Essay ID'].isin(corr_annotated['Essay ID']))]
    
    # getting the appropriate row counts
    total_row_nr_corr = inter[inter['Corr Annotation'] != 'none'].shape[0]
    total_row_nr_corr_and_misclass = inter[(inter['Corr Annotation'] != 'none') & (inter['Gold Standard'] != inter['Prediction'])].shape[0]
    total_row_nr_misclass = inter[inter['Gold Standard'] != inter['Prediction']].shape[0]
    
    return total_row_nr_corr, total_row_nr_corr_and_misclass, total_row_nr_misclass, inter, results_plus, only_pilot
    


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='The path to the folder containing the SweLL gold data.')
    parser.add_argument('results_path', help='The path to the folder containing the results.')
    parser.add_argument('output_name', help='The string to be used in output file names.')
    args = parser.parse_args()
    
    # getting the SweLL docs
    document_list, _ =  read_swell_directory(args.input_path)
    
    # getting all the elements and their correction annotation
    corr_annotated = []
    for document in tqdm(document_list, desc='Extracting corrected tokens...'):
        corr_annotated += extract_corr_annotated(document)
    
    # transforming that into a DataFrame
    error_df = pd.DataFrame(corr_annotated, columns=['Essay ID', 'Token', 'Context', 'PII', 'Corr Annotation'])
            
    # get all results
    all_results = []
    for file in os.listdir(args.results_path):
        if file.endswith('_model_all.xlsx'):
            all_results.append(pd.read_excel(os.path.join(args.results_path, file), index_col=0))
            
    results = all_results[0]
    for i in range(1, len(all_results)):
        results = pd.concat([results, all_results[i]])
    
    # printing the results
    corr, corr_mis, mis, overlap, results_plus, only_pilot = test_outputs(error_df, results)
    print(f"Total number of tokens with correction annotation: {corr}")
    print(f"Total number of tokens with correction annotation that have been misclassified: {corr_mis}")
    print(f"Total number of tokens that have been misclassified: {mis}")
    print()
    print(f"Ratio of misclassified token with correction annotation to all the tokens with correction annotation: {corr_mis/corr * 100:.3f}")
    print(f"Ratio of misclassified token with correction annotation to all the tokens with correction annotation: {corr_mis/mis * 100:.3f}")
    
    # saving the dfs
    overlap.to_excel(f'./results/nodalida/ratios/{args.output_name}_overlap_gold.xlsx')
    results_plus.to_excel(f'./results/nodalida/ratios/{args.output_name}_with_corr_ann.xlsx')
    only_pilot.to_excel(f'./results/nodalida/ratios/{args.output_name}_pilot.xlsx')
    