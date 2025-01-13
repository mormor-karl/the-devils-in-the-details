import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from collections import Counter
from tqdm import tqdm

#######################################################################

def split_tags_and_tokens(input: list):
    '''A function that splits every entry in a list by whitespace and into two separate lists.
    
    Args:
        input (list): A list where every entry is a string containing whitespace.
        
    Returns:
        Two lists of lists, containing the first and the second element of every entry from the original list, split by samples.
    '''
    tokens = []
    tags = []

    temp_token = []
    temp_tag = []
    
    for line in input:
        if len(line.strip()) > 1:
            token = line.strip().split()[0]
            tag = ' '.join(line.strip().split()[1:])

            temp_token.append(token)
            temp_tag.append(tag)

        else:  # if it's a break
            tokens.append(temp_token)
            tags.append(temp_tag)
            # reset
            temp_token = []
            temp_tag = []

    return tokens, tags

def get_tags_and_tokens(filename: str):
    '''A function that extracts tokens and the corresponding tags from a .txt file.
    
    Args:
        filename (str): The name of the file.
    
    Returns:
        A list of tokens and a list of tags.
    '''
    with open(filename) as f:
        tags = f.readlines()
        tokens, tags = split_tags_and_tokens(tags)
        
    return tokens, tags

def get_metadata(filename: str):
    '''A function that extracts the metadata from a .txt file.
    
    Args:
        filename (str): The name of the file.
    
    Returns:
        A list of tokens, a list of essay ids, and a list of original tags.
    '''
    with open(filename) as f:
        tags = f.readlines()
        tokens, other = split_tags_and_tokens(tags)
        
        essay_ids = []
        original_tags = []
        for doc in other:
            temp_essay_ids = []
            temp_original_tags = []
            for entry in doc:
                if ' ' in entry:
                    temp_essay_ids.append(entry.split()[0])
                    temp_original_tags.append(entry.split()[1])
                else:
                    temp_essay_ids.append(entry)
                    temp_original_tags.append(' ')
            essay_ids.append(temp_essay_ids)
            original_tags.append(temp_original_tags)
    return tokens, essay_ids, original_tags

def get_measures(gold_standard: list, predictions: list, path: str, model_name: str, labels: list = [], matrix: bool = False, details: bool = False, ignore_classes: bool = False):
    '''A function intended for retrieving a selection of evaluation measures for comparing the gold standard and the tagger
    annotations. The measures are printed out and include accuracy, Matthew's Correlation Coefficient, per-class precision 
    and recall, as well as a confusion matrix, which, in addition, get saved locally. These measures are calculated using 
    functions from sklearn and pyplot.
    
    Args:
        gold_standard (list[str]): A list of gold standard labels.
        predictions (list[str]): A list of predicted labels.
        path (str): The path for saving the outputs.
        model_name (str): The name of the evaluated model.
        labels (list[str]): A list of labels (if it needs to be specified).
        matrix (bool): Whether or not to produce a confusion matrix.
        
    Returns:
        A list containing all of the non-per-class measures; prints out all of the measures in a separate file.
    '''
    
    if isinstance(gold_standard[0], list):
        gold_standard_list = [x for sentence in gold_standard for x in sentence]
    else:
        gold_standard_list = gold_standard
    if isinstance(predictions[0], list):
        predictions_list = [x for sentence in predictions for x in sentence]
    else:
        predictions_list = predictions

    if labels == []:  # setting up a list of labels based on the training data
        labels = sorted(list(set([x for sentence in gold_standard for x in sentence])))
    else:
        if isinstance(labels[0], list):
            labels = [x for sentence in labels for x in sentence]
            
    if ignore_classes:
        # determining if the classes include the B vs. I division
        ib = False
        for label in labels:
            if 'I' in label or 'B' in label:
                ib = True
        # updating labels
        if ib:
            labels = ['O', 'I', 'B']
        else:
            labels = ['O', 'S']
        # updating the gold standard and predictions
        for i, label in enumerate(gold_standard_list):
            if 'I' in label:
                gold_standard_list[i] = 'I'
            elif 'B' in label:
                gold_standard_list[i] = 'B'
            elif not 'O' in label:
                gold_standard_list[i] = 'S'
                
        for i, label in enumerate(predictions_list):
            if 'I' in label:
                predictions_list[i] = 'I'
            elif 'B' in label:
                predictions_list[i] = 'B'
            elif not 'O' in label:
                predictions_list[i] = 'S'
            
    # calculating the measures
    acc = sklearn.metrics.accuracy_score(gold_standard_list, predictions_list)
    prec = sklearn.metrics.precision_score(gold_standard_list, predictions_list, average="weighted", zero_division=0)
    rec = sklearn.metrics.recall_score(gold_standard_list, predictions_list, average="weighted", zero_division=0)
    f1 = sklearn.metrics.f1_score(gold_standard_list, predictions_list, average="weighted", zero_division=0)
    f2 = sklearn.metrics.fbeta_score(gold_standard_list, predictions_list, average="weighted", zero_division=0, beta=2)
    mcc = sklearn.metrics.matthews_corrcoef(gold_standard_list, predictions_list)

    # writng out the measures
    with open(path + model_name + '_results.txt', 'w') as f:
        f.write('MEASURES:\n')
        f.write(f'Accuracy: {"{:.2%}".format(acc)}\n')
        f.write(f'Precision (weighted): {"{:.2%}".format(prec)}\n')
        f.write(f'Recall (weighted): {"{:.2%}".format(rec)}\n')
        f.write(f'F1 (weighted): {"{:.2%}".format(f1)}\n')
        f.write(f'F2 (weighted): {"{:.2%}".format(f2)}\n')
        f.write(f'Matthew\'s Correlation Coefficient: {"{:.2%}".format(mcc)}\n')
        if details:
            # setting up the value counts for weighted measures of sensitive classes
            counter = Counter(gold_standard_list)
            summed_weights = 0
            for key in counter.keys():
                if key != 'O':
                    summed_weights += counter[key]
            
            f.write('\n')
            f.write('MEASURES PER CLASS:\n')
            
            pii_labels = list(labels)
            pii_labels.remove('O')
            o_idx = labels.index('O')
            
            precision = sklearn.metrics.precision_score(gold_standard_list, predictions_list, average=None, labels=labels, zero_division=0)
            summed_precisions = 0
            for i in range(len(pii_labels)):
                pii_precision = list(precision)
                del pii_precision[o_idx]
                summed_precisions += pii_precision[i] * counter[pii_labels[i]]
            weighted_precision = summed_precisions / summed_weights  
            f.write('Precision:\n')
            for i in range(0,len(labels)):
                f.write(f'\t{labels[i]}: {"{:.2%}".format(precision[i])}\n')
            f.write(f'\tWeighted precision for sensitive classes: {"{:.2%}".format(weighted_precision)}\n')
            
            recall = sklearn.metrics.recall_score(gold_standard_list, predictions_list, average=None, labels=labels, zero_division=0)
            summed_recalls = 0
            for i in range(len(pii_labels)):
                pii_recall = list(recall)
                del pii_recall[o_idx]
                summed_recalls += pii_recall[i] * counter[pii_labels[i]]
            weighted_recall = summed_recalls / summed_weights  
            f.write('Recall:\n')
            for i in range(0,len(labels)):
                f.write(f'\t{labels[i]}: {"{:.2%}".format(recall[i])}\n')
            f.write(f'\tWeighted recall for sensitive classes: {"{:.2%}".format(weighted_recall)}\n')
            
            f1s = sklearn.metrics.f1_score(gold_standard_list, predictions_list, average=None, labels=labels, zero_division=0)
            summed_f1s = 0
            for i in range(len(pii_labels)):
                pii_f1s = list(f1s)
                del pii_f1s[o_idx]
                summed_f1s += pii_f1s[i] * counter[pii_labels[i]]
            weighted_f1 = summed_f1s / summed_weights
            f.write('F1:\n')
            for i in range(0,len(labels)):
                f.write(f'\t{labels[i]}: {"{:.2%}".format(f1s[i])}\n')
            f.write(f'\tWeighted F1 for sensitive classes: {"{:.2%}".format(weighted_f1)}\n')
            
            f2s = sklearn.metrics.fbeta_score(gold_standard_list, predictions_list, average=None, labels=labels, zero_division=0, beta=2)
            summed_f2s = 0
            for i in range(len(pii_labels)):
                pii_f2s = list(f2s)
                del pii_f2s[o_idx]
                summed_f2s += pii_f2s[i] * counter[pii_labels[i]]
            weighted_f2 = summed_f2s / summed_weights
            f.write('f2:\n')
            for i in range(0,len(labels)):
                f.write(f'\t{labels[i]}: {"{:.2%}".format(f2s[i])}\n')
            f.write(f'\tWeighted f2 for sensitive classes: {"{:.2%}".format(weighted_f2)}\n')
    
    # printing out and saving the confusion matrix
    if matrix:
        # print('Confusion matrix:')
        sns.set_context('paper', font_scale=1.5)
        cm = sklearn.metrics.confusion_matrix(gold_standard_list, predictions_list, normalize='true', labels=labels)  # recall 
        ax = sns.heatmap(cm, cmap=sns.color_palette('cividis'), annot=True, xticklabels=labels, yticklabels=labels, cbar=False, linewidths=0.1, linecolor='white')
        ax.set(xlabel='Predicted tag', ylabel='True tag')
        
        # matrix = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labels)
        # fig, ax = plt.subplots(figsize=(12,12))
        # matrix.plot(ax=ax)
        
        plt.savefig(path + model_name + "_confusion_matrix.jpg", bbox_inches='tight')
    
    if details: 
        return [acc, prec, rec, f1, f2, mcc, weighted_precision, weighted_recall, weighted_f1, weighted_f2]
    else:
        return [acc, prec, rec, f1, f2, mcc]
    
def get_k_fold_averages(per_fold_measures: list, path:str):
    '''A function that calculates the averaged-out measures across k folds.
    
    Args:
        per_fold_measures (list): A list of lists of per-fold measures of various kinds.
        path (str): The path for saving the outputs.
    '''
    
    n_folds = len(per_fold_measures)

    # acc, prec, rec, f1, mcc, weighted_precision, weighted_recall, weighted_f1
    
    def get_average(n:int):
        '''An inner function that calculates the average of one measure across k folds.
    
        Args:
            n (int): The number of the measure in the sublist to be averaged out across.
        
        Returns:
            A single average.
        '''
        summed_measure = 0
        for j in range(n_folds):
            summed_measure += per_fold_measures[j][n]
        return summed_measure / n_folds
    
    all_measures = []
    for n in range(len(per_fold_measures[0])):
        all_measures.append(get_average(n))
    
    with open(path + 'all_results.txt', 'w') as f:
        f.write('MEASURES:\n')
        f.write(f'Accuracy: {"{:.2%}".format(all_measures[0])}\n')
        f.write(f'Precision (weighted): {"{:.2%}".format(all_measures[1])}\n')
        f.write(f'Recall (weighted): {"{:.2%}".format(all_measures[2])}\n')
        f.write(f'F1 (weighted): {"{:.2%}".format(all_measures[3])}\n')
        f.write(f'F2 (weighted): {"{:.2%}".format(all_measures[4])}\n')
        f.write(f'Matthew\'s Correlation Coefficient: {"{:.2%}".format(all_measures[5])}\n')
        if len(all_measures) > 5:
            f.write(f'Weighted precision for sensitive classes: {"{:.2%}".format(all_measures[6])}\n')
            f.write(f'Weighted recall for sensitive classes: {"{:.2%}".format(all_measures[7])}\n')
            f.write(f'Weighted F1 for sensitive classes: {"{:.2%}".format(all_measures[8])}\n')
            f.write(f'Weighted F2 for sensitive classes: {"{:.2%}".format(all_measures[9])}\n')
            
        
def get_comparison(standard: list, predictions: list, tokens: list, essay_ids: list, original_tags: list, errors_only: bool = True):
    '''A function that returns a comparison of where mistakes were made during annotation.
    
    Args:
        standard (list): A list of gold standard annotations.
        predictions (list): A list of predicted annotations.
        tokens (list): A list of original tokens corresponding to the tags.
        essay_ids (list): A list of corresponding essay IDs.
        original_tags (list): A list of original tags corresponding to the tokens.
    
    Returns:
        A Pandas dataframe containing the mismatched annotations, their context, tokens, and other data.
    '''

    if errors_only: 
        problematic = []
        for j, sample in enumerate(predictions):
            for i, ann in enumerate(sample):
                if standard[j][i] != ann:
                    if i >= 5:
                        preceding = tokens[j][i-5:i]
                    elif i != 0:
                        preceding = tokens[j][:i]
                    else:
                        preceding = ''
                        
                    if i != len(tokens[j])-5:
                        succeeding = tokens[j][i+1:i+6]
                    elif i != len(tokens[j]):
                        succeeding = tokens[j][i+1:]
                    else:
                        succeeding = ''
                    
                    problematic.append((essay_ids[j][i], tokens[j][i], ' '.join([' '.join(preceding), tokens[j][i], ' '.join(succeeding)]), standard[j][i], predictions[j][i], original_tags[j][i]))
           
        problematic_frame = pd.DataFrame(problematic, columns=['Essay ID', 'Token', 'Context', 'Gold Standard', 'Prediction', 'Original Tag'])
        
        return problematic_frame

    else:
        all_examples = []
        for j, sample in enumerate(predictions):
            for i, ann in enumerate(sample):
                if i >= 5:
                    preceding = tokens[j][i-5:i]
                elif i != 0:
                    preceding = tokens[j][:i]
                else:
                    preceding = ''
                        
                if i != len(tokens[j])-5:
                    succeeding = tokens[j][i+1:i+6]
                elif i != len(tokens[j]):
                    succeeding = tokens[j][i+1:]
                else:
                    succeeding = ''
                    
                all_examples.append((essay_ids[j][i], tokens[j][i], ' '.join([' '.join(preceding), tokens[j][i], ' '.join(succeeding)]), standard[j][i], predictions[j][i], original_tags[j][i]))
           
        frame = pd.DataFrame(all_examples, columns=['Essay ID', 'Token', 'Context', 'Gold Standard', 'Prediction', 'Original Tag'])
        
        return frame
    
def save_metrics(per_fold_measures:list, path:str):
    '''A function that saves the per-fold metrics in an Excel file for later statistical comparisons.
    
    Args:
        per_fold_measures (list): A list of lists featuring selected measures.
        path (str): The output path to save the file.
    '''
    if len(per_fold_measures[0]) > 5:
        dataframe = pd.DataFrame(per_fold_measures, columns=['Accuracy', 'Precision', 'Recall', 'F1', 'F2', 'MCC', 'Sensitive Precision', 'Sensitive Recall', 'Sensitive F1', 'Sensitive F2'])
    else:
        dataframe = pd.DataFrame(per_fold_measures, columns=['Accuracy', 'Precision', 'Recall', 'F1', 'F2', 'MCC'])
    
    mapping = {}
    for i in range(len(per_fold_measures)):
        mapping[i] = f'Model {i+1}'
    
    dataframe.rename(index=mapping, inplace=True)
    dataframe.loc['K-fold mean'] = dataframe.mean() 
    
    dataframe.loc['K-fold STD'] = dataframe.iloc[:-1, :].std()
    
    # dataframe_rounded = dataframe.mul(100).round(2)
        
    dataframe.to_excel(path + 'k_fold_metrics.xlsx')
    
    with open(path + 'latex_results.txt', 'w') as f:
        f.write(dataframe.to_latex())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='The path for saving the output.')
    parser.add_argument('model_name', help='The name of the model (in case of k_fold, just enter anything).')
    
    parser.add_argument('--bert_path', required=False, default='./bert/', help='The path to the folder with one or more folders with BERT runs.')
    parser.add_argument('--data_path', required=False, default='./data/', help='The path to the folder with test data.')
    
    parser.add_argument('--k_fold', help='Choose if the output to be analyzed is from k models.', action='store_true')
    parser.add_argument('--ignore_classes', help='Determines whether the measures should be calculated with detailed classes in mind or just as personal v.s non-personal', action='store_true')
  
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    
    if not args.k_fold:
        print('Loading in the data...')
        
        predictions = args.bert_path + args.model_name + '/test_predictions.txt'
        standard = args.data_path + 'test.txt'
        metadata = args.data_path + 'meta_test.txt'
        
        tokens, predicted_tags = get_tags_and_tokens(predictions)
        _, standard_tags = get_tags_and_tokens(standard)
        _, essay_ids, original_tags = get_metadata(metadata)
        
        print('Calculating measures...')
        
        measures = get_measures(standard_tags, predicted_tags, args.path, args.model_name, details=True, matrix=True, ignore_classes=args.ignore_classes)

        print('Generating comparisons...')
        
        comparison = get_comparison(standard_tags, predicted_tags, tokens, essay_ids, original_tags).sort_values('Gold Standard').reset_index(drop=True)
        comparison.to_excel(args.path + args.model_name + '_errors.xlsx')
        
        comparison = get_comparison(standard_tags, predicted_tags, tokens, essay_ids, original_tags, errors_only=False).sort_values('Gold Standard').reset_index(drop=True)
        comparison.to_excel(args.path + args.model_name + '_all.xlsx')
        
        print('Done!')
    
    else:
        # get the number of splits
        fold_foldernames = next(os.walk(args.bert_path))[1]
        data_foldernames = next(os.walk(args.data_path))[1]
        
        per_fold_measures = []
        
        for i in tqdm(range(len(fold_foldernames)), desc='Analyzing the folds...'):
            # print(f'Loading in the data for {fold}...')
            
            predictions = args.bert_path + fold_foldernames[i] + '/test_predictions.txt'
            standard = args.data_path + data_foldernames[i] + '/test.txt'
            metadata = args.data_path + data_foldernames[i] + '/meta_test.txt'
            
            tokens, predicted_tags = get_tags_and_tokens(predictions)
            _, standard_tags = get_tags_and_tokens(standard)
            _, essay_ids, original_tags = get_metadata(metadata)
            
            # print(f'Calculating measures for {fold}...')
        
            measures = get_measures(standard_tags, predicted_tags, args.path, fold_foldernames[i], details=True, matrix=True, ignore_classes=args.ignore_classes)
            per_fold_measures.append(measures)
            
            # print(f'Generating comparisons for {fold}...')
        
            comparison = get_comparison(standard_tags, predicted_tags, tokens, essay_ids, original_tags).sort_values('Gold Standard').reset_index(drop=True)
            comparison.to_excel(args.path + fold_foldernames[i] + '_errors.xlsx')
            
            comparison = get_comparison(standard_tags, predicted_tags, tokens, essay_ids, original_tags, errors_only=False).sort_values('Gold Standard').reset_index(drop=True)
            comparison.to_excel(args.path + fold_foldernames[i] + '_all.xlsx')
            
        # here we get the per-fold measures
        get_k_fold_averages(per_fold_measures, args.path)
        save_metrics(per_fold_measures, args.path)
        
        print('Done!')