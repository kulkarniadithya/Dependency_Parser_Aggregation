##### The base parsers do not work on all the sentences. Moreover, for some sentences, the token segmentation does not match #####
##### This python file processes the output of the base parsers to find sentences with same token segmentation #####


## Packages import ##
import pickle


if __name__ == "__main__":
    data_location = "../formatted_base_parsers_data/" ## location of formatted base parser data
    save_path = "../selected_sentences/"  ## location to save the output

    datasets = ["en_ewt.conllu"]

    for i in range(0, len(datasets)):
        current_dataset = datasets[i]
        formatted_string = current_dataset.replace(".conllu", "")

        with open(str(data_location) + str(formatted_string) + ".pickle", 'rb') as handle:
            formatted_dictionary = pickle.load(handle)

        base_parser_list = list(formatted_dictionary.keys())
        print(base_parser_list)
        minimum_sentences = 0
        parsers_with_less_sentences = []
        for j in range(0, len(base_parser_list)):
            sentence_list = list(formatted_dictionary[base_parser_list[j]].keys())
            length_sentence_list = len(sentence_list)
            if minimum_sentences == 0:
                minimum_sentences = length_sentence_list
                parsers_with_less_sentences.append(base_parser_list[j])
            elif length_sentence_list < minimum_sentences:
                minimum_sentences = length_sentence_list
                parsers_with_less_sentences.append(base_parser_list[j])

        print(minimum_sentences)
        print(parsers_with_less_sentences)

        reference_parser = parsers_with_less_sentences[-1]

        sentence_list = list(formatted_dictionary[reference_parser].keys())
        mapping_dictionary = {}
        for j in range(0, len(sentence_list)):
            vertices = formatted_dictionary[reference_parser][sentence_list[j]]['vertices']
            sentence = "$$$".join(vertices)
            mapping_dictionary[sentence] = [0] * len(base_parser_list)

        for j in range(0, len(base_parser_list)):
            sentence_list = list(formatted_dictionary[base_parser_list[j]].keys())
            for k in range(0, len(sentence_list)):
                vertices = formatted_dictionary[base_parser_list[j]][sentence_list[k]]['vertices']
                sentence = "$$$".join(vertices)
                try:
                    mapping_dictionary[sentence][j] = 1
                except:
                    print("Sentence does not exist")

        dictionary_sentence_keys = list(mapping_dictionary.keys())

        selected_sentences = {}
        temp = ""
        for j in range(0, len(dictionary_sentence_keys)):
            mapping_array = mapping_dictionary[dictionary_sentence_keys[j]]
            if mapping_array == [1]*len(base_parser_list):
                selected_sentences[dictionary_sentence_keys[j]] = {}
                if temp == "":
                    temp = dictionary_sentence_keys[j]

        # print(selected_sentences)
        print(len(selected_sentences))

        for j in range(0, len(base_parser_list)):
            sentence_list = list(formatted_dictionary[base_parser_list[j]].keys())
            for k in range(0, len(sentence_list)):
                vertices = formatted_dictionary[base_parser_list[j]][sentence_list[k]]['vertices']
                sentence = "$$$".join(vertices)
                try:
                    selected_sentences[sentence][base_parser_list[j]] = {}
                    selected_sentences[sentence][base_parser_list[j]]['vertices'] = vertices
                    selected_sentences[sentence][base_parser_list[j]]['id_to_vertex'] = formatted_dictionary[base_parser_list[j]][sentence_list[k]]['id_to_vertex']
                    selected_sentences[sentence][base_parser_list[j]]['edges'] = formatted_dictionary[base_parser_list[j]][sentence_list[k]]['edges']
                except:
                    print("Sentence is not selected")

        print(selected_sentences[temp])
        with open(str(save_path) + str(formatted_string) + ".pickle", 'wb') as handle:
            pickle.dump(selected_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)









