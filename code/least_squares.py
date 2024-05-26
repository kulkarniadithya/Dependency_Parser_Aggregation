##### This python file computes the least squares #####


## Packages import ##
import pickle
import numpy as np
from itertools import combinations
import pandas as pd


def class_balance_single(t_parameter):
    """Function to compute class balance for single parameter"""
    """Input: t_parameter
       Output: float32 """
    numerator = -np.exp(t_parameter)
    denominator = np.sqrt(4 + np.exp(2*t_parameter))
    return numerator/denominator


def compute_class_balance(t_parameters):
    """Function to compute class balance"""
    """Input: t_parameters
       Output: array(float32) """
    class_balance = []
    for i in range(0, len(t_parameters)):
        class_balance.append(class_balance_single(t_parameter=t_parameters[i]))
    return class_balance


def balanced_accuracy_single(t_parameter, class_balance_value):
    """Function to compute balanced accuracy for single parameter"""
    """Input: t_parameter, class_balance
       Output: float32 """
    numerator = np.exp(t_parameter)
    denominator = np.sqrt(1 - np.square(class_balance_value))
    fraction = numerator/denominator
    return (fraction+1)/2


def compute_balanced_accuracy(t_parameters, class_balance_value):
    """Function to compute balanced accuracy"""
    """Input: t_parameters, class_balance
       Output: array(float32)"""
    balanced_accuracy = []
    for i in range(1, len(t_parameters)):
        balanced_accuracy.append(balanced_accuracy_single(t_parameter=t_parameters[i], class_balance_value=class_balance_value))
    return balanced_accuracy


def compute_sensitivity_and_specificity(class_balance_array, balanced_accuracy):
    """Function to compute sensitivity and specificity"""
    """Input: class_balance_array, balanced_accuracy
       Output: sensitivity_array(float32), specificity_array(float32)"""
    sensitivity_array = []
    specificity_array = []
    for i in range(1, len(class_balance_array)):
        sensitivity = (2*balanced_accuracy[i-1]) + (class_balance_array[0]) - (2*balanced_accuracy[i-1]*class_balance_array[0]) + (class_balance_array[i])
        specificity = (2 * balanced_accuracy[i - 1]) - (class_balance_array[0]) + (
                    2 * balanced_accuracy[i - 1] * class_balance_array[0]) - (class_balance_array[i])
        sensitivity_array.append(sensitivity/2)
        specificity_array.append(specificity/2)

    return sensitivity_array, specificity_array


def compute_class_specific_accuracy(sensitivity_array, specificity_array, mu_00):
    """Function to compute class specific accuracy"""
    """Input: sensitivity_array(float32), specificity_array(float32), mu_00
       Output: class_specific_accuracy_array(float32)"""
    class_specific_accuracy_array = []
    for i in range(0, len(sensitivity_array)):
        first_term = (1 + mu_00)*((2*sensitivity_array[i]) - 1)
        second_term = (1 - mu_00)*((2*specificity_array[i]) - 1)
        value = (first_term/2) + (second_term/2)
        class_specific_accuracy_array.append(value)

    return class_specific_accuracy_array


if __name__ == "__main__":
    dataset_location = "../binary_labeled_dataset/"  ## location of binary labeled dataset
    save_path = "../ising_model_files/"  ## location to save the output

    datasets = ["en_ewt.conllu"]

    for i in range(0, len(datasets)):
        current_dataset = datasets[i]
        formatted_string = current_dataset.replace(".conllu", "")

        with open(save_path + str(formatted_string) + "_sigma_T_dictionary.pickle", "rb") as handle:
            sigma_T_dictionary = pickle.load(handle)

        data = pd.read_csv(dataset_location + str(formatted_string) + ".csv")
        columns = list(data.columns)
        base_parsers = columns[2:]
        print(base_parsers)
        variables = ['zero'] + base_parsers
        two_combinations = list(combinations(base_parsers, 2))
        three_combinations = list(combinations(base_parsers, 3))
        print(len(data))

        keys = list(sigma_T_dictionary.keys())

        print(three_combinations)

        position_dict = {}
        for i in range(0, len(variables)):
            position_dict[variables[i]] = i

        for i in range(0, len(keys)):
            if keys[i] in two_combinations:
                continue
            elif keys[i] in three_combinations:
                continue
            else:
                print(keys[i])

        left_matrix = []
        right_matrix = []

        for i in range(0, len(three_combinations)):
            triplet = three_combinations[i]
            doubles = list(combinations(triplet, 2))
            matrix = []
            matrix.append(1)
            for j in range(0, len(base_parsers)):
                if base_parsers[j] in triplet:
                    matrix.append(1)
                else:
                    matrix.append(0)
            if sigma_T_dictionary[triplet] > 0:
                left_matrix.append(matrix)
                right_matrix.append([np.log10(sigma_T_dictionary[triplet])])
            for k in range(0, len(doubles)):
                try:
                    if sigma_T_dictionary[doubles[k]] > 0:
                        matrix = []
                        matrix.append(0)
                        for j in range(0, len(base_parsers)):
                            if base_parsers[j] in doubles[k]:
                                matrix.append(1)
                            else:
                                matrix.append(0)
                        left_matrix.append(matrix)
                        right_matrix.append([np.log10(sigma_T_dictionary[doubles[k]])])

                except:
                    print("In Except")
                    tuple = (doubles[k][1], doubles[k][0])
                    if sigma_T_dictionary[tuple] > 0:
                        matrix = []
                        matrix.append(0)
                        for j in range(0, len(base_parsers)):
                            if base_parsers[j] in tuple:
                                matrix.append(1)
                            else:
                                matrix.append(0)
                        left_matrix.append(matrix)
                        right_matrix.append([np.log10(sigma_T_dictionary[tuple])])

        print(left_matrix)
        print(right_matrix)

        print(len(left_matrix))
        print(len(right_matrix))

        A = np.vstack(left_matrix)
        print(A)
        y = np.vstack(right_matrix)
        print(y)

        alpha = np.dot((np.dot(np.linalg.pinv(np.dot(A.T, A)), A.T)), y)
        print(alpha)

        t_parameters = []
        for i in range(0, len(alpha)):
            t_parameters.append(alpha[i][0])

        print(t_parameters)

        parameters_dictionary = {}
        parameters_dictionary['base_parsers'] = base_parsers
        parameters_dictionary['t_variables'] = variables
        parameters_dictionary['t_parameters'] = t_parameters
        class_balance_array = compute_class_balance(t_parameters)
        balanced_accuracy_array = compute_balanced_accuracy(t_parameters, class_balance_array[0])
        sensitivity_array, specificity_array = compute_sensitivity_and_specificity(class_balance_array,
                                                                                   balanced_accuracy_array)
        class_specific_accuracy_array = compute_class_specific_accuracy(sensitivity_array, specificity_array,
                                                                        class_balance_array[0])

        print("Class Balance Array")
        print(class_balance_array)
        print("balanced_accuracy_array")
        print(balanced_accuracy_array)
        print("sensitivity_array")
        print(sensitivity_array)
        print("specificity_array")
        print(specificity_array)
        print("class_specific_accuracy_array")
        print(class_specific_accuracy_array)

        parameters_dictionary['class_balance_array'] = class_balance_array
        parameters_dictionary['balanced_accuracy_array'] = balanced_accuracy_array
        parameters_dictionary['sensitivity_array'] = sensitivity_array
        parameters_dictionary['specificity_array'] = specificity_array
        parameters_dictionary['class_specific_accuracy_array'] = class_specific_accuracy_array

        with open(save_path + str(formatted_string) + "_parameters_dictionary.pickle", 'wb') as handle:
            pickle.dump(parameters_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

