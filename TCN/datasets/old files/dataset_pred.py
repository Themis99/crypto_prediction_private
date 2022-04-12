import requests
import pandas as pd
import yfinance as yf

r = requests.get('http://api.alternative.me/fng/?limit=0')

df = pd.DataFrame(r.json()['data'])

df.value = df.value.astype(int)

df.timestamp = pd.to_datetime(df.timestamp,unit = 's')

df.set_index('timestamp',inplace = True)

df = df[::-1]

df_btc = yf.download('BTC-USD')

df_btc.index.name = 'timestamp'


merged = df.merge(df_btc,on = 'timestamp')

merged.head()

merged = merged.drop(['value_classification', 'time_until_update'], axis = 1)

merged = merged[['Open','High','Low','Close','Adj Close','Volume','value']]

print(merged.tail())

missing_values = pd.DataFrame(merged.isna().sum(),columns=['missing_count'])
missing_values.sort_values(by='missing_count',ascending=False)
print(missing_values)

merged.to_csv('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\old_data.csv')

