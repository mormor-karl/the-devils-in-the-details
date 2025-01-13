'''
This is based off of pre-existing code (see the anonymous submission for details). The original code was released under the CRAPL academic-strength open source license.
'''

#######################################################################

import sys
import os
import random
import math
import argparse
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from transformers import AutoTokenizer

parent_dir = os.path.dirname('../src/')  # so that we can access utils.py
sys.path.append(parent_dir)
from utils import *

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

def extract_pii_annotation(document_list: list):
    '''A function that returns a list of IOB-annotated graphs based on the SweLL pilot files extracted with the functions available in 
    utils.py.
    
    Args:
        document_list (list): A list of SweLL file contents as retrieved by functions from utils.py.

    Returns:
        A list of modified Svala-graph sources, now including IOB tags to represent sensitive data/passages.
        A list of document IDs for the reannotated documents.
    '''
    annotated_graphs = []
    
    # extract document ids
    doc_ids = [document['id'] for document in document_list]
    
    # reannotate graphs with the appropriate IOB tags and add document ID information
    for i, document in enumerate(document_list):
        graph = align_pii_annotation(document['svala_graph'])
        for entry in graph:
            entry['doc_id'] = doc_ids[i]
        annotated_graphs.append(graph)

    return annotated_graphs, doc_ids

def align_pii_annotation(graph: dict):
    '''A function that adds the IOB tags to Svala source annotation.
    
    Args:
        graph (dict): A single Svala-graph for a document.

    Returns:
        A list of dicts for every token in the source, including IOB annotation.
    '''
    source = graph['source']
    edges = graph['edges']

    annotated_source = []

    # perform IOB reannotation
    iob_annotations, original_labels = reannotate_2_iob(edges)
    
    # confirm that there are no errors in the annotation step
    assert len(iob_annotations) == len(source)

    # reformat the annotation element by element
    for source_dict in source:
        id = source_dict['id']
        tag = iob_annotations[id]
        original_tag = original_labels[id]
        
        source_dict['tag'] = tag
        source_dict['original_tag'] = original_tag

        annotated_source.append(source_dict)
        
    return annotated_source

def reannotate_2_iob(edges: dict):
    '''A function that takes the edges from a Svala-graph and creates IOB annotations based on them.
    
    Args:
        edges (dict): The edges part of a Svala-graph.

    Returns:
        Dictionaries of token-id and IOB annotation pairs and token-id and original label pairs.
    '''
    iob_annotations = {}
    original_labels = {}
    
    for edge in edges.values():
        source_ids = [x for x in edge['ids'] if x.startswith('s')]
        # select the appropriate annotation from among I, O, and B
        if len(source_ids) == 1 and len(edge['labels']) == 0:  # no annotation
            iob_annotations[source_ids[0]] = 'O'
            original_labels[source_ids[0]] = ''
        elif len(source_ids) == 1 and len(edge['labels']) > 0: # there is some annotation 
            if not set(edge['labels']).isdisjoint(set(PSEUDO_TAGS)):  # make sure the annotation is not just correction annotation 
                iob_annotations[source_ids[0]] = 'B'
                original_labels[source_ids[0]] = [label for label in edge['labels'] if not label.isnumeric() and label in PSEUDO_TAGS]  # only include pseudonymization annotation
            else:  # it was only error annotation, we do not care 
                iob_annotations[source_ids[0]] = 'O'
                original_labels[source_ids[0]] = ''
        else:  # len(source_ids) > 1:
            new_source_ids = [int(x.strip('s')) for x in source_ids]
            new_source_ids.sort()  # this is needed due to a quirk in the annotation (some ranges show up in the wrong order)
            if not set(edge['labels']).isdisjoint(set(PSEUDO_TAGS)):
                iob_annotations['s' + str(new_source_ids[0])] = 'B'
                original_labels['s' + str(new_source_ids[0])] = [label for label in edge['labels'] if not label.isnumeric() and label in PSEUDO_TAGS]
                for i in range(1, len(new_source_ids)):
                    iob_annotations['s' + str(new_source_ids[i])] = 'I'
                    original_labels['s' + str(new_source_ids[i])] = [label for label in edge['labels'] if not label.isnumeric() and label in PSEUDO_TAGS]
            else:
                for i in range(0, len(new_source_ids)):
                    iob_annotations['s' + str(new_source_ids[i])] = 'O'
                    original_labels['s' + str(new_source_ids[i])] = ''
                
    return iob_annotations, original_labels

def create_iob_subcategories(annotated_graphs: list, include_extras: bool = True, mapping: bool | dict = False, no_ib: bool = False):
    '''A function that takes a list of graphs annotated with basic IOB tags and turns those into more detailed ones, including the information about the original pseudo-tags.
    
    Args:
        annotated_graphs (list): A list of IOB-annotated Svala-like graphs.
        include_extras (bool): Determines whether additional functional tags (such as pl, def, gen, foreign) should be included.
        mapping (bool | dict): Determines whether detailed SweLL pseudo-categories (False) or more general pseudo-categories provided by a dictionary mapping should be used.
        no_ib (bool): Determines whether I and B distinctions should be used.

    Returns:
        A list of annotated graphs with more detailed IOB annotation.
    '''
    # determine what mapping to use
    if not mapping:
        for graph in annotated_graphs:
            for entry in graph:
                entry['tag'] = reformat_iob_tags(entry['tag'], entry['original_tag'], include_extras=include_extras, no_ib=no_ib)                   
        return annotated_graphs
    else:
        for graph in annotated_graphs:
            for entry in graph:
                entry['tag'] = reformat_iob_tags(entry['tag'], [mapping[tag] for tag in entry['original_tag']], include_extras=include_extras, no_ib=no_ib) 
        return annotated_graphs

def simplify_iob_tags(annotated_graphs: list):
    '''A function that takes a list of graphs annotated with basic IOB tags and turns those into an even simpler annotation with O standing for non-sensitive and S for sensitive.
    
    Args:
        annotated_graphs (list): A list of IOB-annotated Svala-like graphs.

    Returns:
        A list of annotated graphs with more simplified IOB annotation.
    '''
    for graph in annotated_graphs:
        for entry in graph:
            if entry['tag'] == 'O':
                continue
            else:  # if it is I or B
                entry['tag'] = 'S'
    return annotated_graphs

def reformat_iob_tags(iob_tag: str, original_tags: list, include_extras: bool = True, no_ib: bool = False):
    '''A function that takes the IOB tag and a list of the original tags of an entry in an annotated Svala-like graph and recombines them to create more detailed IOB tags.
    
    Args:
        iob_tag (str): The IOB tag of the element.
        original_tags (list): A list of original tags associated with the entry (one "core" tag and 0 or more functional tags).
        include_extras (bool): Determines whether additional functional tag foreign should be included.
        no_ib (bool): Determines whether I and B distinctions should be used.

    Returns:
        A new, detailed IOB tag.
    '''
    
    if iob_tag == 'O':  # don't do anything to non-entity elements
        pass
    else:
        if include_extras:
            if len(original_tags) > 1 and 'foreign' in original_tags:  # combine geo tag + foreign
                if no_ib:
                    iob_tag = original_tags[0] + '-foreign'
                else:
                    iob_tag = iob_tag + '-' + original_tags[0] + '-foreign'
            else:  # in case of other multi-tag situations, only pick the first one
                if no_ib:
                    iob_tag = original_tags[0]
                else:
                    iob_tag = iob_tag + '-' + original_tags[0]               
        else:  # only pick the first tag
            if no_ib:
                iob_tag = original_tags[0]
            else:
                iob_tag = iob_tag + '-' + original_tags[0]
    return iob_tag

def produce_bert_files(
    annotated_graphs: list,
    classes: list=['B', 'I', 'O'],
    seed: int=25, 
    test_size: float=0.1, 
    dev_size: float=0.1, 
    filenames: list=['train.txt.tmp', 'test.txt.tmp', 'dev.txt.tmp'], 
    meta_filenames: list=['meta_train.txt', 'meta_test.txt', 'meta_dev.txt'],
    path: str='./',
    model: str='KB/bert-base-swedish-cased',
    max_sample_len: int=512,
    k_fold: int=0
    ):
    '''A function that shuffles, splits, and writes out the data in the desired format.
    
    Args:
        annotated_graphs (list): A list of IOB-reannotated graphs.
        seed (int): The random seed to be used.
        test_size (float): The size of the test set, expressed using a decimal fraction.
        dev_size (float): The size of the dev set, expressed using a decimal fraction.
        filenames (list): A list of filenames for the train, test, and dev sets.
        path (str): The path to save the files at.
        model (str): The name of the HuggingFace model used.
        max_sample_len (int): The maximum length of a sample for a HuggingFace model.
        k_fold (int): The number of folds to divide the data into. If 0, regular only a regular train/test/dev split is produced.
        
    Returns:
        The train/test/dev data splits.
    '''
    # create the folder
    if not os.path.exists(path):
        os.mkdir(path)
        
    # save classes
    with open(path + 'labels.txt', 'w') as f:
            for cls in classes:
                f.write(cls + '\n')
    
    # save non-k-fold splits
    if not k_fold:
        data = balanced_shuffle_and_split(annotated_graphs, seed=seed, test_size=test_size, dev_size=dev_size, model=model, max_sample_len=max_sample_len, k_fold=k_fold)
        for i, name in enumerate(filenames):
            write_2_file(data[i], path + name, path + meta_filenames[i])

        print(f'\nData printed into {[path + filename for filename in filenames]} and {[path + filename for filename in meta_filenames]}') 
    
    # save k-fold splits    
    else:
        data, k_fold_tuples = balanced_shuffle_and_split(annotated_graphs, seed=seed, test_size=test_size, dev_size=dev_size, model=model, max_sample_len=max_sample_len, k_fold=k_fold)
        for j, fold in enumerate(k_fold_tuples):
            new_path = path + f'{j+1}_split/'
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            for i, name in enumerate(filenames[:2]):  # no dev 
                write_2_file(fold[i], new_path + name, new_path + meta_filenames[i])

        print(f'\n{k_fold} folds printed into {path}') 
    
    return data
    
def balanced_shuffle_and_split(
    annotated_graphs: list, 
    seed: int=25, 
    test_size: float=0.1, 
    dev_size: float=0.1,
    model: str='KB/bert-base-swedish-cased',
    max_sample_len: int=512,
    k_fold: int=0
    ):
    '''A function that shuffles, splits, and writes out the data in the desired format.
    
    Args:
        annotated_graphs (list): A list of IOB-reannotated graphs.
        seed (int): The random seed to be used.
        test_size (float): The size of the test set, expressed using a decimal fraction.
        dev_size (float): The size of the dev set, expressed using a decimal fraction.
        model (str): The name of the HuggingFace model used.
        max_sample_len (int): The maximum length of a sample for a HuggingFace model.
        k_fold (int): The number of folds to split the data into. Not performed if 0.
        
    Returns:
        Balanced test, train, and dev sets.
    '''
    random.seed(seed)
    random.shuffle(annotated_graphs)

    # splitting the essays by max length with inspiration from https://github.com/huggingface/transformers/tree/main/examples/legacy/token-classification
    tokenizer = AutoTokenizer.from_pretrained(model)
    subsplit_essays = []
    for essay in tqdm(annotated_graphs, desc=f'Creating subsplits of max length {max_sample_len}'):
        subsplit_essays.append(trim_essays_2_bert(essay, tokenizer, max_len=max_sample_len))
                 
    # split up the data into graphs with IB and graphs without it so that each set has an equal number of them
    with_iobs = []
    without_iobs = []
       
    # sorting the samples on the essay level according to the presence of I and B
    for essay in subsplit_essays:
        iobs = False
        for graph in essay:
            for entry in graph:
                if entry['tag'] != 'O':
                    iobs = True
        if iobs:
            with_iobs.append(essay)
        else:
            without_iobs.append(essay)
                
    # removing the non-ib subsamples from ib-essays
    with_iobs = [item for sublist in with_iobs for item in sublist]
    without_iobs = [item for sublist in without_iobs for item in sublist]
    
    for sample in with_iobs:
        iobs = False
        for entry in sample:
                if entry['tag'] != 'O':
                    iobs = True
        if not iobs:
            with_iobs.remove(sample)
         
    # selecting the maximum length for the sets so that our data is balanced
    max_len = min([len(with_iobs), len(without_iobs)])
    
    # trimming the shuffled sets
    with_iobs = with_iobs[:max_len]
    without_iobs = without_iobs[:max_len]
    
    # selecting cutoff points
    test_cutoff = math.floor(test_size * max_len)
    dev_cutoff = math.floor((dev_size+test_size) * max_len)
        
    # making sure we include all the subsamples of the same essay
    iob_test_cutoff = ensure_split_integrity(with_iobs, test_cutoff)
    no_iob_test_cutoff = ensure_split_integrity(without_iobs, test_cutoff)
    iob_dev_cutoff = ensure_split_integrity(with_iobs, dev_cutoff)
    no_iob_dev_cutoff = ensure_split_integrity(without_iobs, dev_cutoff)

    # recombining and reshuffling the data
    test_data = with_iobs[:iob_test_cutoff] + without_iobs[:no_iob_test_cutoff]
    dev_data = with_iobs[iob_test_cutoff:iob_dev_cutoff] + without_iobs[no_iob_test_cutoff:no_iob_dev_cutoff]
    train_data = with_iobs[iob_dev_cutoff:] + without_iobs[no_iob_dev_cutoff:]
        
    random.shuffle(test_data)
    random.shuffle(dev_data)
    random.shuffle(train_data)
        
    data = [train_data, test_data, dev_data]   
    
    if not k_fold:
        return data
    else:  # perform k splits
        k_splits = []
        one_k = max_len / k_fold
        prev_iob_cutoff = 0
        prev_no_iob_cutoff = 0
        
        # create cutoff points
        for i in range(1, k_fold):
            
            # create new cutoff points
            cutoff_point = math.floor(i*one_k)
            iob_cutoff = ensure_split_integrity(with_iobs, cutoff_point)
            no_iob_cutoff = ensure_split_integrity(without_iobs, cutoff_point)
            
            # create split
            k_split = with_iobs[prev_iob_cutoff:iob_cutoff] + without_iobs[prev_no_iob_cutoff:no_iob_cutoff]
            random.shuffle(k_split)
            k_splits.append(k_split)
            
            # reset previous cutoff points
            prev_iob_cutoff = iob_cutoff
            prev_no_iob_cutoff = no_iob_cutoff
            
        # last loop
        k_split = with_iobs[prev_iob_cutoff:] + without_iobs[prev_no_iob_cutoff:]
        random.shuffle(k_split)
        k_splits.append(k_split)
               
        # recombine
        k_fold_tuples = []
        for i in range(k_fold):
            test = k_splits[i]
            train = [sample for k_split in k_splits for sample in k_split if k_splits.index(k_split) != i]
            k_fold_tuples.append((train, test))
            
            # popped = test.pop(0)
            # if popped in test:
            #     print('oops')
            
        return data, k_fold_tuples


def ensure_split_integrity(data: list, split_point: int):
    '''A function that ensures that the selected split cutoff points do not place subsamples of the same essay in different splits.
    
    Args:
        data (list): A list subsamples.
        split_point (int): The desired point for the split.
        
    Returns:
        A verified cutoff point to make a split.
    '''
    last_sample = data[split_point-1]  # select last sample
    last_doc_id = last_sample[0]['doc_id']  # exctract the last sample's doc ID
    
    # iterate until all elements of the same essay are on one side of the split
    for i, sample in enumerate(data[split_point:]):
        if last_doc_id == sample[0]['doc_id']:
            continue
        else:
            new_split_point = split_point + i
            break         
    
    return new_split_point

def trim_essays_2_bert(essay: list, tokenizer, max_len: int = 512):
    '''A function that subsplits the essays into chunks compliant with the HuggingFace model's max input length. Removes NL elements, as they are UNK.
    Inspiration from https://github.com/huggingface/transformers/tree/main/examples/legacy/token-classification
    
    Args:
        essay (list): A list of elements making up the essay.
        tokenizer: A HuggingFace AutoTokenizer object.
        max_len (int): The maximum length of the input.
        
    Returns:
        A list of subsplits of the essay.
    '''
    max_len -= tokenizer.num_special_tokens_to_add()
    subsplits = []
    current_subsplit = []
    current_len = 0
    
    # sum lengths of tokenized tokens until the maximum, split
    for element in essay:
        tokenized = tokenizer.tokenize(element['text'])
        current_sublen = len(tokenized)
        if current_sublen == 0:
            continue
        elif '␤' in element['text']:
            continue
        elif (current_len + current_sublen) > max_len:
            subsplits.append(current_subsplit)
            current_subsplit = [element]
            current_len = current_sublen   
        else:  # it's within the length
            current_subsplit.append(element)
            current_len += current_sublen
    subsplits.append(current_subsplit)
       
    return subsplits
    
def write_2_file(data: list, path: str, path_meta: str):
    '''A function that writes the data in the desired format to a file.
    
    Args:
        data (list): A list of IOB-reannotated graphs representing a single subset (test, train, or dev).
        path (str): The path to save the files at, including filename.
        path_meta (str): The path to save the meta files at, including filename.
    '''
    with open(path, 'w') as f:
        for graph in data:
            for entry in graph:
                token = entry['text']
                tag = entry['tag']
                f.write(token.strip('\n ') + ' ' + tag + '\n')
            f.write('\n')
            
    with open(path_meta, 'w') as f:
        for graph in data:
            for entry in graph:
                token = entry['text']
                original_tag = entry['original_tag']
                doc_id = entry['doc_id']
                f.write(token.strip('\n ') + ' ' + doc_id + ' ' + str(original_tag) + '\n')
            f.write('\n')
            
def calculate_class_weights(annotated_graphs: list, classes: list=[]):
    '''A function that calculates class weights for the data.
    
    Args:
        annotated_graphs (list): A list of IOB-reannotated graphs.
        classes (list): A list of possible classes
        
    Returns:
        A vector of weights for the classes and a list of corresponding classes.
    '''
    y = [entry['tag'] for graph in annotated_graphs for entry in graph]
    
    if len(classes) == 0:
        classes = list(set(y))
    
    # smoothing    
    for cls in classes:
        if cls not in y:
            y.append(cls)

    return compute_class_weight(class_weight='balanced', classes=classes, y=y), classes

def count_classes(annotated_graphs: list, classes: list=[]):
    '''A function that counts instances of the classes in the data.
    
    Args:
        annotated_graphs (list): A list of IOB-reannotated graphs.
        classes (list): A list of possible classes
        
    Returns:
        A list of counts for the classes and a list of the corresponding classes.
    '''
    y = [entry['tag'] for graph in annotated_graphs for entry in graph]
    
    if len(classes) == 0:
        classes = list(set(y))
    
    counts = []
    for cls in classes:
        counts.append(y.count(cls))
    
    return counts, classes
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='The path to the folder containing the SweLL pilot data.')
    parser.add_argument('output_path', help='The path to the folder the output should be saved in.')
    
    parser.add_argument('--test_size', required=False, default=0.1, help='The required test set size as a decimal fraction. Default: 0.1')
    parser.add_argument('--dev_size', required=False, default=0.1, help='The required dev size as a decimal fraction. Default: 0.1')
    parser.add_argument('--seed', required=False, default=25, help='The desired random seed. Default: 25')
    parser.add_argument('--train_file', required=False, default='train.txt', help='The desired name for the train file. Default: train.txt')
    parser.add_argument('--test_file', required=False, default='test.txt', help='The desired name for the test file. Default: test.txt')
    parser.add_argument('--dev_file', required=False, default='dev.txt', help='The desired name for the dev file. Default: dev.txt')
    parser.add_argument('--k_fold', required=False, default=0, help='Split the data into k folds.')
    parser.add_argument('--max_len', required=False, default=512, help='The max length for graphs (longer ones get split up). Default: 512')
    parser.add_argument('--model', required=False, default='KB/bert-base-swedish-cased', help='The name of the HuggingFace model that will be used. Default: KB/bert-base-swedish-cased')
    parser.add_argument('--detailed_iob', required=False, action='store_true', help='Recombine IOB tags with pseudo-tags.')
    parser.add_argument('--general_pseudo', required=False, action='store_true', help='Utilize a mapping of SweLL pseudo-tags to more general pseudo-tags, only has effect if --detailed_iob is used.')
    parser.add_argument('--include_extras', required=False, action='store_true', help='Use functional tags like pl, def, gen, foreign in the more detailed IOB tags, only has effect if --detailed_iob is used.')
    parser.add_argument('--no_ib', required=False, action='store_true', help='Omit the I and B distinction (keep only O and class-specific tags).')
    
    
    args = parser.parse_args()
    
    filenames=[args.train_file, args.test_file, args.dev_file]
    document_list, _ =  read_swell_directory(args.input_path)
    print(f'Loaded in {len(document_list)} documents.')
    annotated_graphs, _ = extract_pii_annotation(document_list)
    if args.detailed_iob:
        if args.general_pseudo:
            mapping = DETAILED_2_GENERAL
            # create a list of classes
            classes = ['O']
            labels = list(set(DETAILED_2_GENERAL.values()))
            for element in ['pl', 'def', 'gen', 'foreign']:
                labels.remove(element)
            for label in labels:
                if args.no_ib:
                    classes.append(label)
                else:
                    classes.append('B-' + label)
                    classes.append('I-'+ label)
            if args.include_extras:
                if args.no_ib:
                    classes.append('geographic-foreign')
                else:
                    classes.append('B-geographic-foreign')
                    classes.append('I-geographic-foreign')
        else:
            mapping = False
            # create a list of classes
            classes = ['O']
            labels = list(DETAILED_2_GENERAL.keys())
            for element in ['pl', 'def', 'gen', 'foreign']:
                labels.remove(element)
            for label in labels:
                if args.no_ib:
                    classes.append(label)
                else:
                    classes.append('B-' + label)
                    classes.append('I-'+ label)
            if args.include_extras:
                foreign_classes = []
                for cls in classes:
                    if cls != 'O':
                        if args.no_ib:
                            if DETAILED_2_GENERAL[cls] == 'geographic':
                                foreign_classes.append(cls + '-foreign')  # This can be refined to only include classes that can be combined with 'foreign'
                        else:
                            if DETAILED_2_GENERAL[cls[2:]] == 'geographic':
                                foreign_classes.append(cls + '-foreign')
                classes += foreign_classes
                    
        annotated_graphs = create_iob_subcategories(annotated_graphs, include_extras=args.include_extras, mapping=mapping, no_ib=args.no_ib)

    else:
        if args.no_ib:
            annotated_graphs = simplify_iob_tags(annotated_graphs)
            classes = ['O', 'S']
        else:
            classes = ['B', 'I', 'O']
                
    data = produce_bert_files(
            annotated_graphs, 
            seed=int(args.seed), 
            test_size=float(args.test_size), 
            dev_size=float(args.dev_size), 
            filenames=filenames,
            path=args.output_path,
            model=args.model,
            max_sample_len=int(args.max_len),
            classes=classes,
            k_fold=int(args.k_fold)
            )
    
    print()
    counts, classes = count_classes(data[0] + data[1] + data[2], classes)
    weights, classes = calculate_class_weights(data[0] + data[1] + data[2], classes)
    print(f'The counts of instances of the {classes} classes are: {counts}')
    print()
    print(f'Weights for the {classes} classes are: {weights}')
    print()