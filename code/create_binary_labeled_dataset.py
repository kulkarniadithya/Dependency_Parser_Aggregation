##### This python file creates binary labeled dataset for Ising model using the sentences with same token segmentation #####


## Packages import ##
import pickle
import csv


if __name__ == "__main__":
    data_location = "../selected_sentences/"  ## directory of selected sentences pickle file
    save_path = "../binary_labeled_dataset/"  ## location to save the output
    edge_id_save_path = "../edge_id_mapping/"  ## location to save the output of edge_id_mapping

    datasets = ["en_ewt.conllu"]

    for i in range(0, len(datasets)):
        current_dataset = datasets[i]
        formatted_string = current_dataset.replace(".conllu", "")

        with open(str(data_location) + str(formatted_string) + ".pickle", 'rb') as handle:
            selected_sentences = pickle.load(handle)

        list_selected_sentences = list(selected_sentences.keys())
        base_parser_list = list(selected_sentences[list_selected_sentences[0]].keys())
        print(base_parser_list)

        csv_file_data = []
        # print(selected_sentences[list_selected_sentences[0]])
        edge_id_dictionary = {}
        for j in range(0, len(list_selected_sentences)):
            base_parser_predictions = selected_sentences[list_selected_sentences[j]]
            unique_edges = []
            for k in range(0, len(base_parser_list)):
                edges = base_parser_predictions[base_parser_list[k]]['edges']
                for z in range(0, len(edges)):
                    if edges[z] not in unique_edges:
                        unique_edges.append(edges[z])
            edge_id = []
            for k in range(0, len(unique_edges)):
                edge_id.append(k)
                csv_dict_file = {}
                csv_dict_file['sentence_id'] = j
                csv_dict_file['edge_id'] = k
                for x in range(0, len(base_parser_list)):
                    edges = base_parser_predictions[base_parser_list[x]]['edges']
                    if unique_edges[k] in edges:
                        csv_dict_file[base_parser_list[x]] = 1
                    else:
                        csv_dict_file[base_parser_list[x]] = -1
                csv_file_data.append(csv_dict_file)

            edge_id_dictionary[list_selected_sentences[j]] = {}
            edge_id_dictionary[list_selected_sentences[j]]['unique_edges'] = unique_edges
            edge_id_dictionary[list_selected_sentences[j]]['edge_id'] = edge_id

        with open(str(edge_id_save_path) + str(formatted_string) + ".pickle", 'wb') as handle:
            pickle.dump(edge_id_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

        csv_columns = ['sentence_id', 'edge_id'] + base_parser_list
        with open(str(save_path) + str(formatted_string) + ".csv", "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_file_data:
                writer.writerow(data)
