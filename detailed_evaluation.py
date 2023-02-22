# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load model output files (predictions)
def load_model_outputfile(file_name, C_or_N='C'):
    """
    :param (str) file_name: location
    :param (str) C_or_N: 'C' or 'N'
    :return: np arrays of prediction, true labels and the weight of the label C or N (int)
    """

    outputs, labels = [], []
    with open(file_name, "r", encoding='windows-1252') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i] != '----------\n':
                parts = lines[i].rstrip('\n').split('\t')

                # label N and C as 0 and 1, respectively
                if parts[1] == 'N': labels.append(0)
                else:               labels.append(1)

                if parts[2] == 'N': outputs.append(0)
                else:               outputs.append(1)

    outputs, labels = np.array(outputs), np.array(labels)
    # flip positive and negative class, if treating N as positive
    if C_or_N == 'N':
        outputs = 1 - outputs
        labels  = 1 - labels
    label_weight = np.sum(labels == 1)

    return outputs, labels, label_weight

# calculate precision
# inputs are predictions, true labels as numpy arrays, output is precision
def precision(outputs, labels):
    # get count of true positives and false positives
    tp = np.sum(np.logical_and(outputs == 1, labels == 1))
    fp = np.sum(np.logical_and(outputs == 1, labels == 0))

    # calculate precision if tp+fp is not 0
    if (tp+fp) != 0:
        return tp/(tp+fp)
    else:
        return np.nan

# calculate recall
# inputs are predictions, true labels as numpy arrays, output is recall
def recall(outputs, labels):
    # get count of true positives and false negatives
    tp = np.sum(np.logical_and(outputs == 1, labels == 1))
    fn = np.sum(np.logical_and(outputs == 0, labels == 1))

    # calculate recall if tp+fn is not 0
    if (tp+fn) != 0:
        return tp/(tp+fn)
    else:
        return np.nan

# calculate f1-score
# inputs are predictions, true labels as numpy arrays, output is f1-scorei am s
def f1(outputs, labels):
    rec = recall(outputs, labels)
    prec = precision(outputs, labels)

    # if either precision or recall is a NaN, or together add up to 0, then the metric itself is a NaN
    if prec == np.nan or rec == np.nan or (prec+rec) == 0:
        return np.nan
    else:
        # otherwise calculate f1-score
        return 2*((prec*rec)/(prec+rec))


if __name__ == '__main__':
    """
        This part is the evaluation of the additional metrics for the baselines and LSTM model
    """
    # initialize variables for getting predictions (i.e. indicatign where predictions are stored),
    # and storing f1-scores and label weights (for calculating weighted f1-score)
    filenames = {
        'random': './baseline predictions/test_rand_pred.tsv',
        'majority': './baseline predictions/test_maj_pred.tsv',
        'frequency': './baseline predictions/test_freq_pred.tsv',
        'length': './baseline predictions/test_len_pred.tsv',
        'lstm': 'experiments/base_model/model_output.tsv'
    }

    f1s = {
        'random': [],
        'majority': [],
        'frequency': [],
        'length': [],
        'lstm': []
    }

    label_weights = {
        'random': [],
        'majority': [],
        'frequency': [],
        'length': [],
        'lstm': []
    }

    # we are calculating metrics for both classes (where either Class N or C can be considered "positive"
    for category in ['N', 'C']:
        print(f'--- {category} ---')
        for file in filenames:

            out, lab, label_weight = load_model_outputfile(filenames[file], category)
            prec = precision(out, lab)
            rec  = recall(out, lab)
            f1_score = f1(out, lab)

            f1s[file].append(f1_score)
            label_weights[file].append(label_weight)

            # print outputs
            print(f'{file}:\n\nPrecision : {prec}\nRecall    : {rec}\nF1-score  : {f1_score}\n\n\n')

    # print weighted f1-scores
    print('Weighted F1-scores\n')
    for model in f1s:
        weighted_f1 = ((f1s[model][0] * label_weights[model][0]) + (f1s[model][1] * label_weights[model][1])) / (label_weights[model][0] + label_weights[model][1])
        print(f'{model}:  {weighted_f1}')

    """
       Run experiment by varying embedding size and learning rate
    """
    def init_vars():
        """
            initializes variables (empty lists) for experiment
        :return: two dictionaries with empty lists (for each run/value of the experiment)
        """
        f1s = {
            'a': [],
            'b': [],
            'c': [],
            'd': [],
            'e': [],
            'f': []
        }

        label_weights = {
            'a': [],
            'b': [],
            'c': [],
            'd': [],
            'e': [],
            'f': []
        }
        return f1s, label_weights


    # define experiment parameters
    hidden_d, lrs = [25, 50, 75, 100, 125, 150], [0.001, 0.003, 0.01, 0.017, 0.03, 0.1]
    experiments, weighted_f1s = [hidden_d, lrs], {'embedding_size': [], 'learning_rate': []}

    for j, experiment in enumerate(experiments):
        print(f'Running experiment: {list(weighted_f1s.keys())[j]}')
        f1s, label_weights = init_vars()
        for i, key in enumerate(f1s):

            # update params file
            with open("./experiments/base_model/params.json", "r+") as a_file:
                params = json.load(a_file)
                # this could be more elegant, but it gets the job done
                if j == 0:
                    params['learning_rate'] = 0.01
                    params["lstm_hidden_dim"] = experiment[i]
                else:
                    params['learning_rate'] = experiment[i]
                    params["lstm_hidden_dim"] = 100
                a_file.close()

            with open("./experiments/base_model/params.json", "w") as a_file:
                json.dump(params, a_file, indent=4)
                a_file.close()

            # run training by executing "train.py"
            print('Training Model...\n')
            trainer = open('train.py')
            train_file = trainer.read()
            exec(train_file)

            # run evaluation by executing "evaluate.py"
            print('Evaluating Model...\n')
            evaluator = open('evaluate.py')
            evaluate_file = evaluator.read()
            exec(evaluate_file)

            # get metrics and store them for calculating weighted f1-score
            print('Getting metrics...\n')
            for class_label in ['N', 'C']:
                out, lab, label_weight = load_model_outputfile('experiments/base_model/model_output.tsv', class_label)
                f1_score = f1(out, lab)
                f1s[key].append(f1_score)
                label_weights[key].append(label_weight)

        # calculate weighted f1-scores
        for key in f1s:
            weighted_f1s[list(weighted_f1s.keys())[j]].append(
                ((f1s[key][0] * label_weights[key][0]) + (f1s[key][1] * label_weights[key][1])) / (
                            label_weights[key][0] + label_weights[key][1]))

    # create Pandas dataframe of values
    weighted_f1s['embedding_size'] = np.array(weighted_f1s['embedding_size'])
    weighted_f1s['learning_rate'] = np.array(weighted_f1s['learning_rate'])
    f1_scores = pd.DataFrame(weighted_f1s)

    # plot
    sns.lineplot(data=f1_scores)
    plt.show()

