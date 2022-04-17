from predictor import predictor
import requests

model = 'model_exp1_alt'

if __name__ == "__main__":

    LAG = 58  # days to look back
    PATH = './' + model  # model path
    pred = predictor(LAG=58)  # object predictor
    signal, prediction, end_date, prev_close, prev_date = pred.predict(PATH=PATH)  # predict

    message = 'Model : ' + model + \
             ' \n\nBTC Price : ' + str(prediction) + \
             ' FOR ' + end_date +\
             ' \n\nUP/DOWN: ' + signal + \
             ' \n\nPREVIOUS CLOSE AT: ' + str(prev_close) + \
             ' FOR ' + prev_date

    print(message)

    baseUrl = 'https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/sendMessage?chat_id=-606025109&text=' + message

    requests.get(baseUrl)