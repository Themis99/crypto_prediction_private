import time
import warnings
from datetime import datetime

from src.data_exporter_dir import data_exporter_library
from src.predictors.eth_predictor import eth_predictor_1
from src.predictors.predictor_1 import predictor_1
from src.predictors.predictor_2 import predictor_2
from src.predictors.predictor_3 import predictor_3
from src.predictors.predictor_4 import predictor_4

warnings.filterwarnings("ignore")


# For the project to run smoothly these need to be installed by hand
# pip install keras-tcn
# pip install keras-tuner --upgrade
# pip install pandas-ta

# Predict using the given model
def predict(modelPath, model, export_file_name, past, print_to_console):
    #  LAG = days to look back

    # model path
    path = modelPath + model

    # object predictor
    pred = ''

    if (model == 'btc_model_1'):
        pred = predictor_1(58, past)
    elif (model == 'btc_model_2'):
        pred = predictor_2(58, past)
    elif (model == 'btc_model_3'):
        pred = predictor_3(48, past)
    elif (model == 'btc_model_4'):
        pred = predictor_4(48, past)
    elif (model == 'eth_model_1'):
        pred = eth_predictor_1(86, past)
    elif (model == 'eth_model_2'):
        pred = eth_predictor_1(86, past)

    signal, prediction, end_date, prev_close, prev_date = pred.predict(PATH=path)  # predict

    # '⬆️' if signal == 'UP' else '⬇️'
    signal_string = signal
    if print_to_console:
        message = 'Using model : [ ' + model + ' ]' \
                                               ' \n\nBTC prediction price : [ ' + str(prediction) + ' $ ] ' \
                                                                                                    ' For Date : [ ' + end_date + ' ] 4 am Greek Time ' \
                                                                                                                                  ' \n\nDirection : ' + signal_string + \
                  ' \n\nBTC previous closing price : [ ' + str(prev_close) + ' $ ] ' \
                                                                             ' For Date : [ ' + prev_date + ' ] 4 am Greek Time'

        print(message)

        # ------------------------------ THE BELOW CODE IS FOR THE TELEGRAM BOT ------------------------------
        # Get Chat id for our bot with apikey = 5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To
        # https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/getUpdates
        # chat id = -1001720397362

        chat_id = '-1001720397362'
        base_url = 'https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/sendMessage?chat_id=' + chat_id + '&text=' + message

        # requests.get(base_url)

    # -------------------- THE BELOW CODE IS FOR EXPORTING THE DATA TO JSON FILE -------------------------

    # Export data to json file for each unique day we run this program
    data_exporter_library.export_data(export_file_name, False, model, prediction,
                                      prev_close, signal, prev_date,
                                      end_date, past)


# produce predictions for n last days
# if you put for example n = 31 and it is 31 May it will
# produce predictions from 1 May until 29 May
def predict_for_past_days(model_path, model, export_file_name, n_days, print_to_console):
    for PAST in range(0, n_days):
        print('Day = [ ' + str(n_days - PAST) + ' ]')
        print('Progress = [ ' + str(round(((PAST + 1) / n_days) * 100, 2)) + ' % ]')
        time.sleep(0.6)

        if PAST >= n_days - 1:
            print('End of predictions')
        else:
            predict(model_path, model, export_file_name, n_days - PAST, print_to_console)


def print_time_now():
    now = datetime.now()
    current_time = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Current Time =", current_time)


if __name__ == "__main__":
    btc_models_path = '../models/bitcoin/'
    btc_export_file_name = 'btc_data'

    ethereum_models_path = '../models/ethereum/'
    ethereum_export_file_name = 'eth_data'

    back_testing = False
    print_to_console = False

    if not back_testing:
        predict(btc_models_path, 'btc_model_1', btc_export_file_name, None, print_to_console)
        time.sleep(0.6)
        predict(btc_models_path, 'btc_model_2', btc_export_file_name, None, print_to_console)
        time.sleep(0.6)
        predict(btc_models_path, 'btc_model_3', btc_export_file_name, None, print_to_console)
        time.sleep(0.6)
        predict(btc_models_path, 'btc_model_4', btc_export_file_name, None, print_to_console)
        time.sleep(0.6)
        predict(ethereum_models_path, 'eth_model_1', ethereum_export_file_name, None, print_to_console)
        time.sleep(0.6)
        predict(ethereum_models_path, 'eth_model_2', ethereum_export_file_name, None, print_to_console)
        time.sleep(0.6)

    else:
        # How many days back to go on back testing
        n_days = 260

        btc_data_past_dates = 'btc_data_past_dates'
        eth_data_past_dates = 'eth_data_past_dates'

        print_time_now()
        # predict_for_past_days(btc_models_path, 'btc_model_2', btc_data_past_dates, n_days, print_to_console)
        print_time_now()
        time.sleep(0.6)
        predict_for_past_days(ethereum_models_path, 'eth_model_1', eth_data_past_dates, n_days, print_to_console)
        print_time_now()
