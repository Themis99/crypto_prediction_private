import pandas as pd

class rolling_zscore:

    def __init__(self, window):
        self.window = window

        m = pd.DataFrame()
        s = pd.DataFrame()
        self.m = m
        self.s = s
    def fit(self, data):
        columns = data.columns
        data_transformed = pd.DataFrame()

        for col in columns:
            roll = data[col].rolling(self.window)
            self.m[col] = roll.mean()
            self.s[col] = roll.std()
            data_transformed[col] = (data[col] - roll.mean()) / roll.std()

        data_transformed = data_transformed.fillna(0)
        return data_transformed



    def inv_fit(self, data):
        columns = data.columns
        data_transformed = pd.DataFrame()
        for col in columns:
            data_transformed[col] = self.s[col]*data[col] + self.m[col]
        return data_transformed

if __name__ == "__main__":
    data = pd.read_csv('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\old_data.csv')
    data.reset_index(inplace=True)
    data = data.drop('timestamp',axis=1)

    roll_z = rolling_zscore(window=24)
    data_trans = roll_z.fit(data)
    print(data_trans.head(24))





