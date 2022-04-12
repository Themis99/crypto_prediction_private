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



