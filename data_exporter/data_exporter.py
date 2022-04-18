import json
import os
from os import path


# Check if dir exist else create it
def create_directory_if_not_exists(directory_path):
    print('Checking if directory : [ ' + directory_path + ' ] exists')
    if not path.exists(directory_path):
        print('Directory : [' + directory_path + ' ] doesnt exist , creating it')
        os.makedirs(directory_path)


# Export the data to json File
def export_data(is_fake_data, model, prediction, prev_close, signal, prev_date, end_date):
    directory_path = './data_exporter/fake_data' if is_fake_data == True else './data_exporter/real_data'
    create_directory_if_not_exists(directory_path)

    json_file_path = directory_path + '/data.json'

    print('Checking if file : [ ' + json_file_path + ' ] exists : ' + str(path.exists(json_file_path)))
    if not path.exists(json_file_path):
        print('Creating Json File')

        # Create the json file
        with open(directory_path + '/data.json', "a") as outfile:
            json.dump({}, outfile, indent=4)

    print('Opening Json File')

    # Read the json
    json_data = {}
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)

    # Check if tensorflow model is in json file if not append it
    model_in_dictionary = model in json_data
    if not model_in_dictionary:
        json_data[model] = {}

    json_data[model][end_date] = {'end_date': end_date, 'prediction': prediction, 'signal': signal,
                                  'prev_close': prev_close, 'prev_date': prev_date, }

    # Write the json file
    with open(json_file_path, 'r+') as outfile:
        # convert back to json.
        json.dump(json_data, outfile, indent=4)

    print('Finished writing to jon')
    # print(json_data)
