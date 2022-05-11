import time
import warnings

from src.data_exporter_dir import data_exporter_library
from src.predictors.eth_predictor import eth_predictor_1
from src.predictors.predictor_1 import predictor_1
from src.predictors.predictor_2 import predictor_2
from src.predictors.predictor_3 import predictor_3
from src.predictors.predictor_4 import predictor_4

warnings.filterwarnings("ignore")


# Predict using the given model
def predict(modelPath, model, export_file_name):
    #  LAG = days to look back

    # model path
    path = modelPath + model

    # object predictor
    pred = ''

    if (model == 'btc_model_1'):
        pred = predictor_1(LAG=58)
    elif (model == 'btc_model_2'):
        pred = predictor_2(LAG=58)
    elif (model == 'btc_model_3'):
        pred = predictor_3(LAG=48)
    elif (model == 'btc_model_4'):
        pred = predictor_4(LAG=48)
    elif (model == 'eth_model_1'):
        pred = eth_predictor_1(LAG=86)
    elif (model == 'eth_model_2'):
        pred = eth_predictor_1(LAG=86)

    signal, prediction, end_date, prev_close, prev_date = pred.predict(PATH=path)  # predict

    # '⬆️' if signal == 'UP' else '⬇️'
    signal_string = signal
    message = 'Using model : [ ' + model + ' ]' \
                                           ' \n\nBTC prediction price : [ ' + str(prediction) + ' $ ] ' \
                                                                                                ' For Date : [ ' + end_date + ' ] 4 am Greek Time ' \
                                                                                                                              ' \n\nDirection : ' + signal_string + \
              ' \n\nBTC closing price : [ ' + str(prev_close) + ' $ ] ' \
                                                                ' For Date : [ ' + prev_date + ' ] 4 am Greek Time'

    print(message)

    # "end_date": "2022-04-20",
    # "prediction": 40572.73,
    # "signal": "DOWN",
    # "prev_close": 41502.75,
    # "prev_date": "2022-04-19"

    # previous_prediction = 40572.73
    # previous_signal = 'DOWN'
    # win_loss = winloss(previous_prediction, previous_signal)
    # print(winloss)

    # ------------------------------ THE BELOW CODE IS FOR THE TELEGRAM BOT ------------------------------
    # Get Chat id for our bot with apikey = 5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To
    # https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/getUpdates
    # chat id = -1001720397362

    chat_id = '-1001720397362'
    base_url = 'https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/sendMessage?chat_id=' + chat_id + '&text=' + message

    # requests.get(base_url)

    # -------------------- THE BELOW CODE IS FOR EXPORTING THE DATA TO JSON FILE -------------------------

    # Export data to json file for each unique day we run this program
    data_exporter_library.export_data(export_file_name, False, model, prediction, prev_close, signal, prev_date,
                                      end_date)


# Predict using the given model
def predict_for_past(modelPath, model, export_file_name, PAST):
    #  LAG = days to look back

    # model path
    path = modelPath + model

    # object predictor
    pred = ''

    if (model == 'btc_model_1'):
        pred = predictor_1(LAG=58)
    elif (model == 'btc_model_2'):
        pred = predictor_2(past=PAST, LAG=58)
    elif (model == 'btc_model_3'):
        pred = predictor_3(LAG=48)
    elif (model == 'btc_model_4'):
        pred = predictor_4(LAG=48)
    elif (model == 'eth_model_1'):
        pred = eth_predictor_1(LAG=86)
    elif (model == 'eth_model_2'):
        pred = eth_predictor_1(LAG=86)

    signal, prediction, end_date, prev_close, prev_date = pred.predict(PATH=path)  # predict

    # '⬆️' if signal == 'UP' else '⬇️'
    signal_string = signal
    message = 'Using model : [ ' + model + ' ]' \
                                           ' \n\nBTC prediction price : [ ' + str(prediction) + ' $ ] ' \
                                                                                                ' For Date : [ ' + end_date + ' ] 4 am Greek Time ' \
                                                                                                                              ' \n\nDirection : ' + signal_string + \
              ' \n\nBTC previous closing price : [ ' + str(prev_close) + ' $ ] ' \
                                                                         ' For Date : [ ' + prev_date + ' ] 4 am Greek Time'

    print(message)

    # "end_date": "2022-04-20",
    # "prediction": 40572.73,
    # "signal": "DOWN",
    # "prev_close": 41502.75,
    # "prev_date": "2022-04-19"

    # previous_prediction = 40572.73
    # previous_signal = 'DOWN'
    # win_loss = winloss(previous_prediction, previous_signal)
    # print(winloss)

    # ------------------------------ THE BELOW CODE IS FOR THE TELEGRAM BOT ------------------------------
    # Get Chat id for our bot with apikey = 5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To
    # https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/getUpdates
    # chat id = -1001720397362

    chat_id = '-1001720397362'
    base_url = 'https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/sendMessage?chat_id=' + chat_id + '&text=' + message

    # requests.get(base_url)

    # -------------------- THE BELOW CODE IS FOR EXPORTING THE DATA TO JSON FILE -------------------------

    # Export data to json file for each unique day we run this program
    data_exporter_library.export_data(export_file_name, False, model, prediction, prev_close, signal, prev_date,
                                      end_date, PAST)


if __name__ == "__main__":
    btc_model_path = '../models/bitcoin/'
    btc_export_file_name = 'btc_data'

    ethereum_model_path = '../models/ethereum/'
    ethereum_export_file_name = 'eth_data'
    #
    # predict(btc_model_path, 'btc_model_1', btc_export_file_name)
    # time.sleep(0.6)
    # predict(btc_model_path, 'btc_model_2', btc_export_file_name)
    # time.sleep(0.6)
    # predict(btc_model_path, 'btc_model_3', btc_export_file_name)
    # time.sleep(0.6)
    # predict(btc_model_path, 'btc_model_4', btc_export_file_name)
    # time.sleep(0.6)
    # predict(ethereum_model_path, 'eth_model_1', ethereum_export_file_name)
    # time.sleep(0.6)
    # predict(ethereum_model_path, 'eth_model_2', ethereum_export_file_name)

    btc_export_file_name_past_dates = 'btc_data_past_dates'

    n = 61
    for PAST in range(0, n):
        if PAST >= n - 1:
            print('End of predictions')
        else:
            predict_for_past(btc_model_path, 'btc_model_2', btc_export_file_name_past_dates, n - PAST)

        time.sleep(0.6)
