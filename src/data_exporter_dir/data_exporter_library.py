import json
import os
from os import path

from src import data_collector

print_info = False  # True


def winloss(previous_prediction, previous_signal):
    # take data
    data = data_collector.retrieve_data2()

    # take previous close
    previous_close = round(data.tail(1)['Close'].item(), 2)

    # προ-προ-χθεσινό close
    previous_previous_close = round(data[:-1].tail(1)['Close'].item(), 2)

    #  what signal really happened
    real_signal = 'UP' if previous_previous_close < previous_close else 'DOWN'

    # Compare the two signals WIN if signals are the same or else loss
    trade = 'WIN' if real_signal == previous_signal else 'LOSS'

    # If trade is win take the difference from real close and the predicted close
    # if real_signal == previous_signal else None
    out = int(abs(previous_close - previous_prediction))

    print('previous_close : [ ' + str(previous_close) + ' ] , previous_previous_close  : [ ' + str(
        previous_previous_close) + ' ] , previous_prediction : [ ' + str(previous_prediction) + ' ]')
    return trade, out


def print_message(message):
    if print_info:
        print(message)


# Check if dir exist else create it
def create_directory_if_not_exists(directory_path):
    print_message('Checking if directory : [ ' + directory_path + ' ] exists')
    if not path.exists(directory_path):
        print_message('Directory : [' + directory_path + ' ] doesnt exist , creating it')
        os.makedirs(directory_path)


# Export the data to json File
def export_data(is_fake_data, model, prediction, prev_close, signal, prev_date, end_date):
    print_message('Starting export data')

    directory_path = './fake_data' if is_fake_data == True else './real_data'
    create_directory_if_not_exists(directory_path)

    json_file_path = directory_path + '/data.json'

    print_message('Checking if file : [ ' + json_file_path + ' ] exists : ' + str(path.exists(json_file_path)))
    if not path.exists(json_file_path):
        print_message('Creating Json File')

        # Create the json file
        with open(directory_path + '/data.json', "a") as outfile:
            json.dump({}, outfile, indent=4)

    print_message('Opening Json File')

    # Read the json
    json_data = {}
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)

    # Check if tensorflow model is in json file if not append it
    model_in_dictionary = model in json_data
    if not model_in_dictionary:
        json_data[model] = {}

    # Calculate win loss if previous_prediction exists
    # print(datetime.utcnow().strftime("%Y-%m-%d"))

    # previous_prediction = json_data[model][prev_date]['prediction']
    # previous_signal = json_data[model][prev_date]['signal']
    # win = ''
    # out = ''
    #
    # if previous_prediction:
    #     print('Previous_prediction_2', previous_prediction)
    #     win, out = winloss(previous_prediction, previous_signal)
    #     data = json_data[model][prev_date]
    #     json_data[model][prev_date] = {
    #         'end_date': data['end_date'],
    #         'prediction': data['prediction'],
    #         'signal': data['signal'],
    #         'prev_close': data['prev_close'],
    #         'prev_date': data['prev_date'],
    #         'win': win,
    #         'out': out
    #     }

    json_data[model][end_date] = {
        'end_date': end_date,
        'prediction': prediction,
        'signal': signal,
        'prev_close': prev_close,
        'prev_date': prev_date
    }

    # Write the json file
    with open(json_file_path, 'r+') as outfile:
        # convert back to json.
        json.dump(json_data, outfile, indent=4)

    print_message('Finished writing to jon')
    # print(json_data)
