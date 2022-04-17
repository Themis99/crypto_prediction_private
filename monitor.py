from predictor import predictor
import requests

model = 'model_exp1_alt'

if __name__ == "__main__":

    LAG = 58  # days to look back
    PATH = './' + model  # model path
    pred = predictor(LAG=58)  # object predictor
    signal, prediction, end_date, prev_close, prev_date = pred.predict(PATH=PATH)  # predict

    signalString = '⬆️' if signal == 'UP' else '⬇️'
    message = 'Using model : [ ' + model + ' ]'\
             ' \n\nBTC price : [ ' + str(prediction) + ' $ ] '\
             ' For Date : [ ' + end_date + ' ]'\
             ' \n\nDirection : ' + signalString + \
             ' \n\nPREVIOUS CLOSE AT: [ ' + str(prev_close) + ' $ ] ' \
             ' For Date : [ ' + prev_date + ' ]'

    print(message)

    # Get Chat id for our bot with apikey = 5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To
    # https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/getUpdates
    # chat id = -1001720397362

    chatId = '-1001720397362'
    baseUrl = 'https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/sendMessage?chat_id=' + chatId + '&text=' + message

    requests.get(baseUrl)
