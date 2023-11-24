import pandas as pd


def is_consistent(hypothesis, instance):
    for i in range(len(hypothesis)):
        if hypothesis[i] != '?' and hypothesis[i] != instance[i]:
            return False
    return True


def generalize(hypothesis, instance):
    for i in range(len(hypothesis)):
        if hypothesis[i] == '0':
            hypothesis[i] = instance[i]
        elif hypothesis[i] != instance[i]:
            hypothesis[i] = '?'
    return hypothesis


def specialize(hypothesis, instance):
    new_hypothesis = list(hypothesis)
    for i in range(len(new_hypothesis)):
        if new_hypothesis[i] != '?' and new_hypothesis[i] != instance[i]:
            new_hypothesis[i] = '0'
    return new_hypothesis


def candidate_elimination(training_data):

    specific_hypothesis = ['0'] * (len(training_data.columns) - 1)
    general_hypothesis = ['?'] * (len(training_data.columns) - 1)
    
    
    for index, row in training_data.iterrows():
        instance = list(row[:-1])
        label = row[-1]

        
        if label == 'Yes':
            specific_hypothesis = generalize(specific_hypothesis, instance)
            
            for i in range(len(general_hypothesis)):
                if not is_consistent(general_hypothesis, instance):
                    general_hypothesis = specialize(general_hypothesis, instance)
                else:
                    general_hypothesis[i] = specific_hypothesis[i]

       
        elif label == 'No':
            general_hypothesis = specialize(general_hypothesis, instance)

    return specific_hypothesis, general_hypothesis
file_path = 'C:/Users/prana/OneDrive/Documents/training_data.csv'
training_data = pd.read_csv(file_path)
specific_hypothesis, general_hypothesis = candidate_elimination(training_data)
print("Specific Hypothesis:", specific_hypothesis)
print("General Hypothesis:", general_hypothesis)
