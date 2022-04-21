from src import data_collector
from src.data_exporter_dir import data_exporter_library
from src.predictors.predictor_1 import predictor_1
from src.predictors.predictor_2 import predictor_2
from src.predictors.predictor_3 import predictor_3


def winloss(previous_prediction, previous_signal):
    # take data
    data = data_collector.retrieve_data()
    # take prev close
    close_r = round(data.tail(1)['Close'].item(), 2)
    # take comp day
    close_p = round(data[:-1].tail(1)['Close'].item(), 2)

    real_s = 'UP' if close_p < close_r else 'DOWN'

    trade = 'WIN' if real_s == previous_signal else 'LOSS'
    out = abs(close_r - previous_prediction) if real_s == previous_signal else None

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

    signal, prediction, end_date, prev_close, prev_date = pred.predict(PATH=path)  # predict

    signalString = '⬆️' if signal == 'UP' else '⬇️'
    message = 'Using model : [ ' + model + ' ]'\
             ' \n\nBTC closing price : [ ' + str(prediction) + ' $ ] '\
             ' For Date : [ ' + end_date + ' ] 4 am Greek Time '\
             ' \n\nDirection : ' + signalString + \
             ' \n\nBTC closing price : [ ' + str(prev_close) + ' $ ] ' \
             ' For Date : [ ' + prev_date + ' ] 4 am Greek Time'

    # print(message)

# print(winloss())
    # ------------------------------ THE BELOW CODE IS FOR THE TELEGRAM BOT ------------------------------
    # Get Chat id for our bot with apikey = 5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To
    # https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/getUpdates
    # chat id = -1001720397362

    chat_id = '-1001720397362'
    base_url = 'https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/sendMessage?chat_id=' + chat_id + '&text=' + message

    # requests.get(base_url)

    # -------------------- THE BELOW CODE IS FOR EXPORTING THE DATA TO JSON FILE -------------------------

    # Export data to json file for each unique day we run this program
    data_exporter_library.export_data(False, model, prediction, prev_close, signal, prev_date, end_date)


if __name__ == "__main__":
    predict('model_1')
    predict('model_2')
    predict('model_3')
