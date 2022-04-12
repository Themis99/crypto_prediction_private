import requests
import yfinance as yf
import pandas as pd
from predictor import predictor
import matplotlib.pyplot as plt


if __name__ == "__main__":
    PATH = 'C:\\Users\\Themis\\Desktop\\bitcoin_pred\\Best models\\new_model'
    look_back = 24
    pred = predictor(days=look_back)
    signal,predicted_price,prediction_date,previous_close = pred.predict(PATH)
    print('BTC: '+str(predicted_price)+' SIGNAL: '+signal+' PREDICTION FOR DATE: '+prediction_date+' PREVIOUS CLOSE: '+str(previous_close))
    print('Running the bot')

    messege = 'BTC: '+str(predicted_price)+' SIGNAL: '+signal+' PREDICTION FOR DATE: '+prediction_date+' PREVIOUS CLOSE: '+str(previous_close)

    baseUrl = 'https://api.telegram.org/bot5145257581:AAFFag1OAu9fR5KE0YTHsY2303z8CF-o6To/sendMessage?chat_id=-606025109&text=' + messege

    requests.get(baseUrl)



