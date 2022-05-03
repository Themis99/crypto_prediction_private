import pandas as pd
import pandas_ta as ta
import yfdata


def prepropcess(IND_LAG = 14):
    data = yfdata.yahoo_retriever()
    close = data['Close']
    data['RSI'] = ta.rsi(close,IND_LAG)
    data['EMA'] = ta.ema(close,IND_LAG)
    macd_df = ta.macd(close)
    data['MACDh'] = macd_df['MACDh_12_26_9']
    data['MACDs'] = macd_df['MACDs_12_26_9']
    data['MACD'] = macd_df['MACD_12_26_9']

    data = data.fillna(0)

    return data