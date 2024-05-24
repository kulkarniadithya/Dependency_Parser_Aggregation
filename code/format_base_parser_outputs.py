##### Python file to format the output of base parsers #####

## Packages import ##
import pickle


def get_graph(tuples):
    vertices = []
    edges = []
    indexes = []
    id_to_vertex = {}
    for i in range(0, len(tuples)):
        id_to_vertex[tuples[i][0]] = tuples[i][1]
    for i in range(0, len(tuples)):
        vertices.append(tuples[i][1])
        indexes.append(tuples[i][0])
        if tuples[i][2] != '0':
            try:
                temp = [tuples[i][2], tuples[i][0]]
                edges.append(temp)
            except:
                print(tuples)
    return indexes, edges, id_to_vertex, vertices


if __name__ == "__main__":
    data_location = "../processed_base_parsers_data/" ## location of processed base parser data
    save_path = "../formatted_base_parsers_data/"  ## location to save the formatted output

    datasets = ["en_ewt.conllu"]

    for i in range(0, len(datasets)):
        current_dataset = datasets[i]
        formatted_string = current_dataset.replace(".conllu", "")
        with open(str(data_location) + str(formatted_string) + ".pickle", 'rb') as handle:
            dictionary = pickle.load(handle)

        submission_sent_id = dictionary['submission_sent_id']
        submission_text = dictionary['submission_text']
        submission_sent_tuples = dictionary['submission_sent_tuples']
        submission = dictionary['submission']

        formatted_dictionary = {}
        for j in range(0, len(submission)):
            formatted_dictionary[submission[j]] = {}
            for k in range(0, len(submission_sent_tuples[j])):
                indexes, edges, id_to_vertex, vertices = get_graph(submission_sent_tuples[j][k])

                formatted_dictionary[submission[j]][k] = {}
                formatted_dictionary[submission[j]][k]['indexes'] = indexes
                formatted_dictionary[submission[j]][k]['edges'] = edges
                formatted_dictionary[submission[j]][k]['id_to_vertex'] = id_to_vertex
                formatted_dictionary[submission[j]][k]['vertices'] = vertices

        with open(str(save_path) + str(formatted_string) + ".pickle", 'wb') as handle:
            pickle.dump(formatted_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
