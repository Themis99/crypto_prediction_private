import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\eng_data.csv')

print(data['macd hist closing_price'])

def _VIF(df,drop_f,filter =10):
    #VIF
    X = df.drop(drop_f,axis=1)

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    return_best = vif_data.loc[vif_data["VIF"] <= filter]
    return_worst = vif_data.loc[vif_data["VIF"] > filter]

    return return_best, return_worst

drop = ['Date','closing_price']
best , _ = _VIF(data,drop_f = drop)

print(best)

#keep best
best_dataset = data[best['feature']]

#print(best_dataset.head())

#save best dataset
best_dataset['target_price'] = data['closing_price']
best_dataset.insert(loc = 0, column = 'Date', value = data['Date'].tolist())
print(best_dataset.tail())
#best_dataset.to_csv('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\final_df.csv',index=False)




