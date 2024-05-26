##### This python file infers the aggregated value #####


## Packages import ##
import pickle
import pandas as pd
import math


def sigmoid(x):
    """Function to compute sigmoid value"""
    """Input: float32
       Output: float32"""
    return 1 / (1 + math.exp(-x))


def matmul(input_array, theta_0j):
    """Function to compute matrix multiplication"""
    """Input: input_array, theta_0j
       Output: float32"""
    sum_value = 0
    for i in range(0, len(input_array)):
        sum_value = sum_value + input_array[i]*theta_0j[i]
    return 2*sum_value


def inference(input_array, theta_0j, theta_00):
    """Function to compute prediction value"""
    """Input: input_array, theta_0j, theta_00
       Output: float32"""
    return sigmoid((2*theta_00) + matmul(input_array=input_array, theta_0j=theta_0j))


if __name__ == "__main__":
    dataset_location = "../binary_labeled_dataset/"  ## location of binary labeled dataset
    save_path = "../ising_model_files/"  ## location to save the output

    datasets = ["en_ewt.conllu"]

    for i in range(0, len(datasets)):
        current_dataset = datasets[i]
        formatted_string = current_dataset.replace(".conllu", "")
        data = pd.read_csv(dataset_location + str(formatted_string) + ".csv")
        columns = list(data.columns)
        base_parsers = columns[2:]
        print(base_parsers)

        with open(save_path + str(formatted_string) + "_parameters_dictionary_independent_assumption.pickle", "rb") as handle:
            parameters_dictionary = pickle.load(handle)

        theta_00 = parameters_dictionary['theta_00']
        theta_0j = parameters_dictionary['theta_0j']

        baseline_data_values = []
        for z in range(0, len(base_parsers)):
            data_value = data[base_parsers[z]]
            baseline_data_values.append(data_value)

        prediction = []
        for z in range(0, len(baseline_data_values[0])):
            input_array = []
            for y in range(0, len(base_parsers)):
                input_array.append(baseline_data_values[y][z])

            prediction.append(inference(input_array, theta_0j, theta_00))

        data['prediction'] = prediction

        data.to_csv(dataset_location + str(formatted_string) + "_pred.csv")
