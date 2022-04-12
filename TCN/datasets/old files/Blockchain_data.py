import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import quandl
import investpy
from datetime import date


URL_array = set()


def link2df(URL, col_name, join_df, join=True, check_column=True, check_URL=True, clear_URL_array=False,
            show_details=False):
    '''This function scraps the given link and returns dataframe
    __________
    Parameters:
        URL(string): URL to be scrapped from bitcoin website
        col_name(string): column name for dataframe
        join_df(variable)= dataframe withwhich output dataframe will be left joined on Date
        join(boolean)= iF True,join, else don't join
        check_column(boolean)= check if column name already exists
        check_URL(boolean)= check if URL is already processed
        clear_URL_array(boolean)= if true URL_processed array will be cleared
        show_details(boolean)= various details wil be printed such as scrapping first and last details, df head & df tail
        '''

    print(f'processing {col_name}')

    # clear URL append array
    if clear_URL_array == True:
        URL_array.clear()

    # set join parameters if false
    if join == False:
        join_df = None
        check_column = False

    # process column name by making it lowercase and replacing spaces,commas, full stops
    col_name = col_name.lower().replace(',', '').replace(" ", "_").replace(".", "_")

    # col_name validation if exists already
    if check_column == True and col_name in list(join_df.columns):
        print(f'column {col_name} already esists in dataframe, stopped here')
        return join_df

    # URL validation if processes already
    elif check_URL == True and URL in list(URL_array):
        print(f'{URL} is already processed, stopped here')
        return join_df

        # web scrapping
    page = requests.get(URL)
    soup = page.content
    soup = str(soup)
    scraped_output = (soup.split('[[')[1]).split('{labels')[0][0:-2]
    if show_details == True:
        print('head')
        print({scraped_output[0:30]})
        print('tail')
        print({scraped_output[-30:]})

    processed_str = scraped_output.replace('new Date(', '')
    processed_str = processed_str.replace(')', '')
    processed_str = processed_str.replace('[', '')
    processed_str = processed_str.replace(']', '')
    processed_str = processed_str.replace('"', '')

    processed_str_list = processed_str.split(',')
    date_list, data_list = processed_str_list[::2], processed_str_list[1::2]

    # validate column lengths
    if len(date_list) != len(data_list):
        print(f'date & data length:{len(date_list), len(data_list), len(date_list) == len(data_list)}')

    # convert list data to a dataframe
    if join == False:
        df = pd.DataFrame()
        df['Date'] = pd.to_datetime(date_list)
        df[col_name] = data_list
        URL_array.add(URL)
        if show_details == True:
            print('*' * 100)
            print('df head')
            print(df.head(1))
            print('*' * 100)
            print('df tail')
            print(df.tail(1))
            print('*' * 100)
            print(f'df shape{df.shape}')
            print('=' * 100)

        return df

    elif col_name not in list(join_df.columns) and join == True:
        df = pd.DataFrame()
        df['Date'] = pd.to_datetime(date_list)
        df[col_name] = data_list
        join_df = pd.merge(join_df, df, on=['Date'], how='left')
        URL_array.add(URL)
        if show_details == True:
            print('*' * 100)
            print('df head')
            print(df.head(1))
            print('*' * 100)
            print('df tail')
            print(df.tail(1))
            print('*' * 100)
            print(f'output df shape= {df.shape},joined_df shape = {join_df.shape}')
            print('=' * 100)
            print(f'Number of duplicate columns in dataframe {df.columns.duplicated().sum()}')
            print('=' * 100)

        return join_df

##################################################################################################################
today = date.today()
Start_date = '01/02/2018'
End_date = str((today.strftime("%d/%m/%Y")))


# 01.Price
final_df = investpy.get_crypto_historical_data(crypto='bitcoin',from_date=Start_date,to_date=End_date)
final_df = final_df.reset_index()
final_df.drop(['Currency','Volume'],inplace=True,axis=1)
final_df.columns = ['Date','opening_price','highest_price','lowest_price','closing_price']

print(final_df.tail())


# 02.Number of transactions in blockchain per day
final_df = link2df('https://bitinfocharts.com/comparison/bitcoin-transactions.html',
                   'transactions in blockchain',join_df=final_df,join=True)

# 03.Average block size
final_df = link2df('https://bitinfocharts.com/comparison/size-btc.html',
                   'avg block size',join_df=final_df,join=True)
# 04.Number of unique (from) addresses per day
final_df = link2df('https://bitinfocharts.com/comparison/sentbyaddress-btc.html',
                   'sent by adress',join_df=final_df,join=True)
# 05.Average mining difficulty per day
final_df = link2df('https://bitinfocharts.com/comparison/bitcoin-difficulty.html',
                   'avg mining difficulty',join_df=final_df,join=True)
# 06.Average hashrate (hash/s) per day
final_df = link2df('https://bitinfocharts.com/comparison/bitcoin-hashrate.html',
                   'avg hashrate',join_df=final_df,join=True)
# 07.Mining Profitability USD/Day for 1 Hash/s
final_df = link2df('https://bitinfocharts.com/comparison/bitcoin-mining_profitability.html',
                   'mining profitability',join_df=final_df,join=True)
# 08.Sent coins in USD per day
final_df = link2df('https://bitinfocharts.com/comparison/sentinusd-btc.html',
                   'Sent coins in USD',join_df=final_df,join=True)
# 09.Average transaction fee, USD
final_df = link2df('https://bitinfocharts.com/comparison/bitcoin-transactionfees.html',
                   'avg transaction fees',join_df=final_df,join=True)

# 10.Median transaction fee, USD
final_df = link2df('https://bitinfocharts.com/comparison/bitcoin-median_transaction_fee.html',
                   'median transaction fees',join_df=final_df,join=True)
# 11.Average block time (minutes)
final_df = link2df('https://bitinfocharts.com/comparison/bitcoin-confirmationtime.html',
                   'avg block time',join_df=final_df,join=True)
# 12.Avg. Transaction Value, USD
final_df = link2df('https://bitinfocharts.com/comparison/transactionvalue-btc.html',
                   'avg transaction value',join_df=final_df,join=True)
# 13.Median Transaction Value, USD
final_df = link2df('https://bitinfocharts.com/comparison/mediantransactionvalue-btc.html',
                   'median transaction value',join_df=final_df,join=True)
# 14.Number of unique (from or to) addresses per day
final_df = link2df('https://bitinfocharts.com/comparison/activeaddresses-btc.html',
                   'active addresses',join_df=final_df,join=True)
# 17.Top 100 Richest Addresses to Total coins %
final_df = link2df('https://bitinfocharts.com/comparison/top100cap-btc.html',
                   'top100 to total percentage',join_df=final_df,join=True)
# 18.Average Fee Percentage in Total Block Reward
final_df = link2df('https://bitinfocharts.com/comparison/fee_to_reward-btc.html',
                   'avg fee to reward',join_df=final_df,join=True)
# 19.Total number of bitcoins in circulation
btc_in_circulation_df = quandl.get("BCHAIN/TOTBC",authtoken='9ztFCcK4_e1xGo_gjzK7')
btc_in_circulation_df = btc_in_circulation_df.rename(columns={'Value': 'number_of_coins_in_circulation'})
# 20.Bitcoin Miners Revenue
miners_revenue_df = quandl.get("BCHAIN/MIREV",authtoken='9ztFCcK4_e1xGo_gjzK7')
miners_revenue_df = miners_revenue_df.rename(columns={'Value': 'miner_revenue'})

# fear and greed index
r = requests.get('http://api.alternative.me/fng/?limit=0')

fear_and_greed = pd.DataFrame(r.json()['data'])
fear_and_greed.value = fear_and_greed.value.astype(int)
fear_and_greed.timestamp = pd.to_datetime(fear_and_greed.timestamp,unit = 's')
fear_and_greed.set_index('timestamp',inplace = True)
fear_and_greed = fear_and_greed[::-1]
fear_and_greed = fear_and_greed.reset_index()
fear_and_greed = fear_and_greed.rename(columns={'timestamp': 'Date'})

#Filtering data as we are considering this peiod only
final_df = final_df[(final_df['Date'] >= '2018-02-01')].reset_index(drop=True)

final_df = pd.merge(final_df,btc_in_circulation_df,on=['Date'],how='left')
final_df = pd.merge(final_df,miners_revenue_df,on=['Date'],how='left')

final_df = pd.merge(final_df,fear_and_greed,on=['Date'],how='left')

final_df = final_df.drop(['value_classification', 'time_until_update'], axis = 1)

final_df.replace(to_replace='null', value=np.nan,inplace=True)
final_df.drop(final_df.tail(1).index,inplace=True)

final_df = final_df.rename(columns={'value': 'fear_and_greed_index'})

#imputing missing values
missing_values = pd.DataFrame(final_df.isna().sum(),columns=['missing_count'])
missing_values.sort_values(by='missing_count',ascending=False)
print(missing_values)

final_df['active_addresses'].fillna(final_df['active_addresses'].rolling(14, min_periods=1).mean()).astype(float).plot(x=final_df['Date'],y='active_addresses',figsize=(25,5),grid=True)
for i in list(final_df.loc[pd.isna(final_df['active_addresses']),:].index):
    plt.axvline(x=i,color='r',alpha=0.1)
plt.ylabel('number of active adresses')
plt.title('Date vs number of active adresses(with highlighted imputation)')
plt.show()
final_df['active_addresses'] = final_df['active_addresses'].fillna(final_df['active_addresses'].rolling(14, min_periods=1).mean())

final_df['fear_and_greed_index'].fillna(final_df['fear_and_greed_index'].rolling(14, min_periods=1).mean()).astype(float).plot(x=final_df['Date'],y='fear_and_greed_index',figsize=(25,5),grid=True)
for i in list(final_df.loc[pd.isna(final_df['fear_and_greed_index']),:].index):
    plt.axvline(x=i,color='r',alpha=0.1)
plt.ylabel(' fear_and_greed_index')
plt.title('Date vs fear_and_greed_index(with highlighted imputation)')
plt.show()
final_df['fear_and_greed_index'] = final_df['fear_and_greed_index'].fillna(final_df['fear_and_greed_index'].rolling(24, min_periods=1).mean())

missing_values = pd.DataFrame(final_df.isna().sum(),columns=['missing_count'])
missing_values.sort_values(by='missing_count',ascending=False)
print(missing_values)

print(final_df.tail())
PATH = 'C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\data_blockchain_features.csv'
final_df.to_csv(PATH,index=False)
