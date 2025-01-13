import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#######################################################################

def get_counts_and_percentages(df, model_col, ann_type):
    '''A function gets the PII counts and accuracy percentages per class.
    
    Args:
        df (DataFrame): A DataFrame with all the results (output by compare_predictions.py).
        model_col (str): The name of the model column to look at.
        ann_type (str): The type of PII annotation to take into consideration.
    
    Returns:
        A DataFrame containing the results.
    '''
    # retrieve only PII tokens
    only_piis = df[df['Is PII'] == True]
    # get counts per PII tag
    tag_counts = only_piis[ann_type].value_counts() 
    # get counts of correct predictions per PII tag
    correct_counts = only_piis.groupby(ann_type)[model_col].sum()
    # combine the counts into one DataFrame
    combined_counts = pd.merge(tag_counts, correct_counts, on=ann_type)
    # calculate accuracy
    combined_counts['Accuracy'] = (round(combined_counts[model_col] / combined_counts['count'], 3))
    # reset the index
    combined_counts.reset_index(inplace=True)
    
    return combined_counts

def visualize(df, ann_type, model_cols, labels, filename, tick_rotation=45, legend_loc='upper right'):
    '''A function which generates a visualization comparing PII accuracies across two annotation types.
    
    Args:
        df (DataFrame): A DataFrame with all the results (output by compare_predictions.py).
        model_cols (list[str]): A list of the names of the model columns to look at.
        ann_type (str): The type of PII annotation to take into consideration.
        labels (list[str]): A list of labels to use to describe the models.
        filename (str): The name of the file to save the visualization to.
        tick_rotation (int): The degree of rotation for x-axis ticks. 45 by default.
    
    Returns:
        Saves a visualization.
    '''
    # set seaborn context
    sns.set_context("paper", font_scale=1.25)
    # colors: https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=5
    
    # get DataFrames for both annotation types
    dfs = []
    for col in model_cols:
        dfs.append(get_counts_and_percentages(df, col, ann_type))
    # initialize suffixes   
    suffixes = [model_cols[0].lstrip('Correct'), model_cols[1].lstrip('Correct')]
    # merge the two dataframes, drop certain columns  
    data = pd.merge(
        dfs[0], dfs[1], on=[ann_type, 'count'], suffixes=suffixes
        ).drop(
            columns=['count', model_cols[0], model_cols[1]]
            )
    # rename accuracy columns
    data = data.rename(
                columns={
                    f'Accuracy{suffixes[0]}': labels[0],
                    f'Accuracy{suffixes[1]}': labels[1]
                }
            )
    # melt the combined data for later use in a barplot
    tidy_data = data.melt(id_vars=ann_type).rename(columns={'variable': 'Annotation Type'})
    # initialize the figure
    fig, ax1 = plt.subplots() 
    # copy the axis for the lineplot
    ax2 = ax1.twinx() 
    # fill in the plots
    sns.barplot(data=tidy_data, x=ann_type, y='value', hue='Annotation Type', palette=sns.color_palette(['#8c510a', '#dfc27d']), ax = ax1) 
    sns.lineplot(data=dfs[0], x=ann_type, y='count', color='#35978f', marker='o', markersize=8, linestyle='', ax = ax2) 
    # rename the axes
    ax1.set_xlabel('PII tags')
    ax1.set_ylabel('Accuracy', color='#8c510a')
    ax2.set_ylabel('Class instances', color='#35978f')
    # flip the ticks
    ax1.tick_params(axis='x', rotation=tick_rotation)
    # move the legend
    sns.move_legend(ax1, loc=legend_loc)
    # save the figure
    plt.savefig(filename, bbox_inches='tight')
    
def visualize_pii_counts(df, ann_type, filename, tick_rotation=45):
    '''A function which generates a visualization comparing PII accuracies across two annotation types.
    
    Args:
        df (DataFrame): A DataFrame with all the results (output by compare_predictions.py).
        ann_type (str): The type of PII annotation to take into consideration.
        filename (str): The name of the file to save the visualization to.
        tick_rotation (int): The degree of rotation for x-axis ticks. 45 by default.
    
    Returns:
        Saves a visualization.
    '''
    # set seaborn context
    sns.set_context("paper")
    # get the DataFrame
    df = get_counts_and_percentages(df, 'Correct_k_detailed_no_ib', ann_type)
    # initialize the figure
    fig, ax1 = plt.subplots() 
    # fill in the plots
    sns.barplot(data=df, x=ann_type, y='count', color='#35978f', ax = ax1) 
    # rename the axes
    ax1.set_xlabel('PII tags')
    ax1.set_ylabel('Count', color='#35978f')
    # flip the ticks
    ax1.tick_params(axis='x', rotation=tick_rotation)
    # set numbers on display
    ax1.bar_label(ax1.containers[0])
    # save the figure
    plt.savefig(filename, bbox_inches='tight')     

#######################################################################

if __name__ == "__main__":
    # import data 
    df = pd.read_excel('./results/nodalida/ratios/results_from_all_models.xlsx')
    print('Data loaded!')
    # visualize detailed
    visualize(df, "Gold Standard", ['Correct_k_detailed_no_ib', 'Correct_k_detailed'], ['Detailed', 'Detailed IOB'], './results/nodalida/ratios/vis_detailed.png', tick_rotation=90)
    visualize_pii_counts(df, 'Gold Standard', './results/nodalida/ratios/counts_detailed.png', tick_rotation=90)
    print('Detailed visualization done!')
    # visualize general
    visualize(df, "General Tag", ['Correct_k_general_pseudo_no_ib', 'Correct_k_general_pseud'], ['General', 'General IOB'], './results/nodalida/ratios/vis_general.png', legend_loc='lower left')
    visualize_pii_counts(df, 'General Tag', './results/nodalida/ratios/counts_general.png')
    print('General visualization done!')
    # visualize basic
    visualize(df, "Basic Tag", ['Correct_k_only_iobs_no_ib', 'Correct_k_only_iob'], ['Basic', 'Basic IOB'], './results/nodalida/ratios/vis_basic.png', tick_rotation=0, legend_loc='lower left')
    visualize_pii_counts(df, 'Basic Tag', './results/nodalida/ratios/counts_basic.png')
    print('Basic visualization done!')