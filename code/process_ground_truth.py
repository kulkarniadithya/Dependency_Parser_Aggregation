##### Python file to process ground truth #####
##### Official Submission Download Link: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2885 #####

## Packages import ##

import os
import pickle


if __name__ == "__main__":
    directory = "../official_submissions/" ## directory path for official submissions
    save_path = "../ground_truth_data/"  ## location to save the processed output

    sub_dir = os.listdir(directory)

    print(sub_dir)

    datasets = ["en_ewt.conllu"]
    for i in range(0, len(datasets)):
        current_dataset = datasets[i]
        submission_sent_id = []
        submission_text = []
        submission_sent_tuples = []
        submission = []

        for dir in sub_dir:
            if dir in ['00-gold-standard']:
                try:
                    path = directory + dir + "/"
                    files = os.listdir(path)
                    if current_dataset in files:
                        sent_id = []
                        text = []
                        sent_tuples = []
                        temp = []
                        file_open = open(path + current_dataset, "r")
                        for line in file_open:
                            string = line.rstrip()
                            if "# sent_id" in string:
                                sent_id.append(string)
                            elif "# text" in string:
                                text.append(string)
                            elif "\t" in string and string not in ["", "\n"]:
                                splits = string.split("\t")
                                inter_temp = []
                                inter_temp.append(splits[0])
                                inter_temp.append(splits[1])
                                inter_temp.append(splits[6])
                                if inter_temp[2] == '_':
                                    print("The following line is not part of the dependency tree since details about parent node is not provided.")
                                    print(string)
                                    print("The line is present in the directory: " + str(dir))
                                elif inter_temp[0] == '_':
                                    print(inter_temp)
                                    print(current_dataset)
                                else:
                                    temp.append(inter_temp)
                            else:
                                if len(temp) > 0:
                                    sent_tuples.append(temp)
                                    temp = []
                        submission_sent_id.append(sent_id)
                        submission_text.append(text)
                        submission_sent_tuples.append(sent_tuples)
                        submission.append(dir)
                        file_open.close()
                except:
                    print(dir)

        dictionary = {'submission_sent_id': submission_sent_id, 'submission_text': submission_text,
                      'submission_sent_tuples': submission_sent_tuples, 'submission': submission}
        formatted_string = current_dataset.replace(".conllu", "")
        with open(str(save_path) + str(formatted_string) + ".pickle", 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
