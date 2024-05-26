##### This python file computes exact solution for the problem considering the base parsers as independent #####


## Packages import ##
import pickle
import torch


def compute_theta_00(sensitivity_array, specificity_array, mu_00):
    """Function to compute theta_00"""
    """Input: sensitivity_array(torch.Tensor(float32)), specificity_array(torch.Tensor(float32)), mu_00(torch.Tensor)
       Output: theta_00 (tensor.float)"""

    first_term = torch.log((1 + mu_00)/(1 - mu_00))
    temp_value = (sensitivity_array*(1-sensitivity_array))/(specificity_array*(1-specificity_array))
    second_term = temp_value.log()

    output_array = (first_term/2) + (second_term/4)
    return output_array


def compute_theta_0j(sensitivity_array, specificity_array):
    """Function to compute theta_0j"""
    """Input: sensitivity_array(torch.Tensor(float32)), specificity_array(torch.Tensor(float32))
       Output: theta_0j (tensor.float)"""

    first_term = sensitivity_array*(1-sensitivity_array).log()
    second_term = specificity_array*(1-specificity_array).log()

    output_array = (first_term + second_term)/4
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

        theta_00 = compute_theta_00(torch.tensor(sensitivity_array, dtype=torch.float32), torch.tensor(specificity_array, dtype=torch.float32), torch.tensor(class_balance_array[0]))
        theta_0j = compute_theta_0j(torch.tensor(sensitivity_array, dtype=torch.float32), torch.tensor(specificity_array, dtype=torch.float32))

        print(theta_00)
        print(theta_0j)

        parameters_dictionary['theta_00'] = theta_00
        parameters_dictionary['theta_0j'] = theta_0j

        with open(save_path + str(formatted_string) + "_parameters_dictionary_independent_assumption.pickle", 'wb') as handle:
            pickle.dump(parameters_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
