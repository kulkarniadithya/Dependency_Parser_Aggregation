##### This python file computes exact solution for the problem considering the base parsers as independent #####


## Packages import ##
import pickle
import torch
import numpy as np


def compute_theta_00(sensitivity_array, specificity_array, mu_00):
    """Function to compute theta_00"""
    """Input: sensitivity_array(float32), specificity_array(float32), mu_00
       Output: theta_00"""

    first_term = np.log((1 + mu_00) / (1 - mu_00))
    second_term = 0
    for i in range(0, len(sensitivity_array)):
        temp = (sensitivity_array[i] * (1 - sensitivity_array[i])) / (specificity_array[i] * (1 - specificity_array[i]))
        print(temp)
        second_term = second_term + np.log(temp)

    value = (first_term / 2) + (second_term / 4)
    return value


def compute_theta_0j(sensitivity_array, specificity_array):
    """Function to compute theta_0j"""
    """Input: sensitivity_array(float32), specificity_array(float32)
       Output: theta_0j"""

    output_array = []
    for i in range(0, len(sensitivity_array)):
        first_term = np.log((sensitivity_array[i]) / (1 - sensitivity_array[i]))
        second_term = np.log((specificity_array[i]) / (1 - specificity_array[i]))

        value = (first_term + second_term) / 4
        output_array.append(value)
    return output_array


if __name__ == "__main__":
    dataset_location = "../binary_labeled_dataset/"  ## location of binary labeled dataset
    save_path = "../ising_model_files/"  ## location to save the output

    datasets = ["en_ewt.conllu"]

    for i in range(0, len(datasets)):
        current_dataset = datasets[i]
        formatted_string = current_dataset.replace(".conllu", "")

        with open(save_path + str(formatted_string) + "_parameters_dictionary.pickle", "rb") as handle:
            parameters_dictionary = pickle.load(handle)

        class_balance_array = parameters_dictionary['class_balance_array']
        sensitivity_array = parameters_dictionary['sensitivity_array']/(max(parameters_dictionary['sensitivity_array'])+0.01)
        specificity_array = parameters_dictionary['specificity_array']/(max(parameters_dictionary['specificity_array'])+0.01)

        print(class_balance_array)
        print(parameters_dictionary['balanced_accuracy_array'])
        print(sensitivity_array)
        print(specificity_array)

        theta_00 = compute_theta_00(sensitivity_array, specificity_array, class_balance_array[0])
        theta_0j = compute_theta_0j(sensitivity_array, specificity_array)

        print(theta_00)
        print(theta_0j)

        parameters_dictionary['theta_00'] = theta_00
        parameters_dictionary['theta_0j'] = theta_0j

        with open(save_path + str(formatted_string) + "_parameters_dictionary_independent_assumption.pickle", 'wb') as handle:
            pickle.dump(parameters_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
