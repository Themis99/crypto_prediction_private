import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.stats as st
import matplotlib.patches as mpl_patches
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import describe
import matplotlib.cm as cm
import matplotlib.lines as mlines
from pandas.core.window.rolling import Rolling
import pandas_ta as ta


def feature_smoothening(df, feature_name, smoothening_type, smoothening_range=[7,12,24], show_plot=False,
                        show_original_Feature_in_plot=False):
    if smoothening_type == 'sma':
        for j in smoothening_range:
            df[f'{smoothening_type}{j} {feature_name}'] = ta.sma(df[feature_name], j)

    elif smoothening_type == 'var':
        for j in smoothening_range:
            df[f'{smoothening_type}{j} {feature_name}'] = ta.variance(df[feature_name], j)

    elif smoothening_type == 'stdev':
        for j in smoothening_range:
            df[f'{smoothening_type}{j} {feature_name}'] = ta.stdev(df[feature_name], j)

    elif smoothening_type == 'ema':
        for j in smoothening_range:
            df[f'{smoothening_type}{j} {feature_name}'] = ta.ema(df[feature_name], j)

    elif smoothening_type == 'wma':
        for j in smoothening_range:
            df[f'{smoothening_type}{j} {feature_name}'] = ta.wma(df[feature_name], j)

    elif smoothening_type == 'rsi':
        for j in smoothening_range:
            df[f'{smoothening_type}{j} {feature_name}'] = ta.rsi(df[feature_name], j)

    elif smoothening_type == 'roc':
        for j in smoothening_range:
            df[f'{smoothening_type}{j} {feature_name}'] = ta.roc(df[feature_name], j)

    elif smoothening_type == 'dema':
        for j in smoothening_range:
            df[f'{smoothening_type}{j} {feature_name}'] = ta.dema(df[feature_name], j)

    elif smoothening_type == 'tema':
        for j in smoothening_range:
            df[f'{smoothening_type}{j} {feature_name}'] = ta.tema(df[feature_name], j)

    elif smoothening_type == 'bband_lower':
        for j in smoothening_range:
            bband_df = ta.bbands(df[feature_name], j)
            df[f'{smoothening_type}{j} {feature_name}'] = bband_df[f'BBL_{j}_2.0']

    elif smoothening_type == 'bband_upper':
        for j in smoothening_range:
            bband_df = ta.bbands(df[feature_name], j)
            df[f'{smoothening_type}{j} {feature_name}'] = bband_df[f'BBU_{j}_2.0']

    elif smoothening_type == 'macd':
        macd_df = ta.macd(df[feature_name])
        df[f'{smoothening_type} hist {feature_name}'] = macd_df['MACDh_12_26_9']
        df[f'{smoothening_type} signal {feature_name}'] = macd_df['MACDs_12_26_9']
        df[f'{smoothening_type} {feature_name}'] = macd_df['MACD_12_26_9']

    if show_plot == True and show_original_Feature_in_plot == True:
        df[[feature_name] + [i for i in list(df.columns) if i.split(" ")[-1] == feature_name and i.split(" ")[0][0:len(
            smoothening_type)] == smoothening_type]].plot(kind='line', figsize=(25, 5))
        plt.grid()
        plt.title(f'Feature Smoothening-{feature_name} by {smoothening_type}')
        plt.xticks([])
        plt.show()

    elif show_plot == True and show_original_Feature_in_plot == False:
        df[[i for i in list(df.columns) if
            i.split(" ")[-1] == feature_name and i.split(" ")[0][0:len(smoothening_type)] == smoothening_type]].plot(
            kind='line', figsize=(25, 5))
        plt.grid()
        plt.title(f'Feature Smoothening-{feature_name} by {smoothening_type}')
        plt.xticks([])
        plt.show()

data = pd.read_csv('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\data_blockchain_features.csv')

print(data.head())

feature_list = ['opening_price','highest_price','lowest_price','closing_price']
print(feature_list)

#Simple Moving Average
for feature in feature_list:
    feature_smoothening(data,feature,'sma',show_plot=False)

#Weighted Moving Average
for feature in feature_list:
    feature_smoothening(data,feature,'wma',show_plot=False)

#Exponential Moving Average
for feature in feature_list:
    feature_smoothening(data,feature,'ema',show_plot=False)

#Relative Strength Index
for feature in feature_list:
    feature_smoothening(data,feature,'rsi',show_plot=False,show_original_Feature_in_plot=False)

#Bollinger Bands
for feature in feature_list:
    feature_smoothening(data,feature,'bband_lower',show_plot=False)

for feature in feature_list:
    feature_smoothening(data,feature,'bband_upper',show_plot=False)

#Moving Average Convergence Divergence
for feature in feature_list:
    feature_smoothening(data,feature,'macd',show_plot=False,show_original_Feature_in_plot=False)

data = data.fillna(method='bfill')

data.to_csv('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\eng_data.csv',index=False)


