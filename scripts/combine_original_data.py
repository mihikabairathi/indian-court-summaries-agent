import json
import os

# return the raw contents of the file
def open_file(path):
    with open(path, 'r') as file: 
        return file.read()
    
# Append the JSON content to a file
def write_file(path, content):
    with open(path, 'a') as file:
        json.dump(content, file, indent=4)

if __name__ == "__main__":
    content = {}

    for path in ["Data/original_data/test_data", "Data/original_data/train_data"]:
        for content_type in ["judgement", "summary"]:
            for file in os.listdir(os.path.join(path, content_type)):
                # get the file name and raw contents
                case_number = int(file.split('.')[0])
                judgement = open_file(os.path.join(path, content_type, file))

                # add the contents to the content dictionary
                content[case_number] = content.get(case_number, {})
                content[case_number][content_type] = judgement

    write_file("Data/combined_original_data.json", dict(sorted(content.items())))
        