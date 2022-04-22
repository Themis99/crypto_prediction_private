import time
import warnings

from src import data_collector
from src.data_exporter_dir import data_exporter_library
from src.predictors.predictor_1 import predictor_1
from src.predictors.predictor_2 import predictor_2
from src.predictors.predictor_3 import predictor_3
from src.predictors.predictor_4 import predictor_4
import warnings
warnings.filterwarnings("ignore")



def winloss(previous_prediction, previous_signal):

    # take data
    data = data_collector.retrieve_data()

    # take previous close
    previous_close = round(data.tail(1)['Close'].item(), 2)

    # προ-προ-χθεσινό close
    previous_previous_close = round(data[:-1].tail(1)['Close'].item(), 2)

    #  what signal really happened
    real_signal = 'UP' if previous_previous_close < previous_close else 'DOWN'

    # Compare the two signals WIN if signals are the same or else loss
    trade = 'WIN' if real_signal == previous_signal else 'LOSS'

    # If trade is win take the difference from real close and the predicted close
    out = abs(previous_close - previous_prediction) if real_signal == previous_signal else None

    print('previous_close : [ ' + str(previous_close) + ' ] , previous_previous_close  : [ ' + str(previous_previous_close) + ' ] , previous_prediction : [ ' + str(previous_prediction) + ' ]')
    return trade, out

# Predict using the given model
def predict(model):
    #  LAG = days to look back

    # model path
    path = '../models/' + model

    # object predictor
    pred = ''

    if (model == 'model_1'):
        pred = predictor_1(LAG=58)
    elif (model == 'model_2'):
        pred = predictor_2(LAG=58)
    elif (model == 'model_3'):
        pred = predictor_3(LAG=48)
    elif (model == 'model_4'):
        pred = predictor_4(LAG=48)

    signal, prediction, end_date, prev_close, prev_date = pred.predict(PATH=path)  # predict

    signal_string = '⬆️' if signal == 'UP' else '⬇️'
    message = 'Using model : [ ' + model + ' ]'\
             ' \n\nBTC closing price : [ ' + str(prediction) + ' $ ] '\
             ' For Date : [ ' + end_date + ' ] 4 am Greek Time '\
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
    data_exporter_library.export_data(False, model, prediction, prev_close, signal, prev_date, end_date, 5)

if __name__ == "__main__":
    predict('model_1')
    time.sleep(0.6)
    predict('model_2')
    time.sleep(0.6)
    predict('model_3')
    time.sleep(0.6)
    predict('model_4')
