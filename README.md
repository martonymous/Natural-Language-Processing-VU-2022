# VU_NLPT assignment 1 Submission

Authors: Marton Fejer(2694002),Gergő Pandurcsek (2730356), Carol Rameder(2747982)
Instructions:
- Make sure to install the required libraries and packages with $ pip install -r requirements.txt
- For part A run analyses.py, the output of each exercise is printed in the same order as they appear. You can see the tags for the 5th exercise by opening the Assi1_A_NER.html in the browser.
- For part B run baselines.py. It outputs the values also found in the paper for exercises 8 (the plots) and 10.
- For part C, you need to build the vocabulary by running build_vocab.py and then train and evaluate the model by running train.py, then evaluate.py. 
Afterwards, run detailed_evaluation.py to calculate and output the values for exercise 12 AND to run the experiments testing different hyperparamaters and get the graphs for ex. 14 run experiments.py.
Other hyperparameters (such as Number of Epochs) of the model can be changed in params.json. The new model can be trained again - run train.py and evaluated and reevaluated - run detailed_evaluation.py.

- Alternatively, you can run main.py to go through all exercises