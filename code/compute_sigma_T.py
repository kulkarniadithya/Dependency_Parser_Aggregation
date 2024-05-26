##### This python file computes the values of sigma and T used for aggregation #####


## Packages import ##
import torch
from itertools import combinations
import pandas as pd
import pickle


def compute_mean(input_tensor):
    """Function to compute mean value of an input torch tensor"""
    """Input: input_tensor (torch.Tensor)
       Output: output_tensor (torch.float32)"""
    if not isinstance(input_tensor, torch.Tensor):
        try:
            input_tensor = torch.from_numpy(input_tensor)
        except:
            raise TypeError("Input is not of the type torch.Tensor")
    return input_tensor.mean()


def compute_sigma(*input_tensors, device="cpu"):
    """Function to compute sigma value of input torch tensors"""
    """Input: input_tensors (torch.Tensor)
       Output: output_tensor (torch.float32)"""
    if isinstance(input_tensors[0], torch.Tensor):
        cov_vec = torch.ones(input_tensors[0].shape, device=device)
    else:
        cov_vec = None
    for input_tensor in input_tensors:
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Input is not of the type torch.Tensor")
        elif input_tensor.shape!=cov_vec.shape:
            raise ValueError("Input tensors should all be of same shape")
        cov_vec *= input_tensor - input_tensor.mean()
    return cov_vec.sum()/(cov_vec.shape[0]-1)


if __name__ == "__main__":
    dataset_location = "../binary_labeled_dataset/"  ## location of binary labeled dataset
    save_path = "../ising_model_files/"  ## location to save the output
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = ["en_ewt.conllu"]

    for i in range(0, len(datasets)):
        current_dataset = datasets[i]
        formatted_string = current_dataset.replace(".conllu", "")
        data = pd.read_csv(dataset_location + str(formatted_string) + ".csv")
        columns = list(data.columns)
        base_parsers = columns[2:]
        print(base_parsers)
        two_combinations = list(combinations(base_parsers, 2))
        three_combinations = list(combinations(base_parsers, 3))
        print(len(data))

        sigma_T_dictionary = {}
        for z in range(0, len(two_combinations)):
            sigma_T_dictionary[two_combinations[z]] = compute_sigma(torch.from_numpy(data[two_combinations[z][0]].values).to(device=device, dtype=torch.float32), torch.from_numpy(data[two_combinations[z][1]].values).to(device=device, dtype=torch.float32), device=device).item()

        for z in range(0, len(three_combinations)):
            sigma_T_dictionary[three_combinations[z]] = compute_sigma(torch.from_numpy(data[three_combinations[z][0]].values).to(device=device, dtype=torch.float32), torch.from_numpy(data[three_combinations[z][1]].values).to(device=device, dtype=torch.float32), torch.from_numpy(data[three_combinations[z][2]].values).to(device=device, dtype=torch.float32), device=device).item()

        print(sigma_T_dictionary)

        with open(save_path + str(formatted_string) + "_sigma_T_dictionary.pickle", "wb") as handle:
            pickle.dump(sigma_T_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

