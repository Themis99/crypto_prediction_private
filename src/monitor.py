from src import data_collector
from src.predictors.predictor_2 import predictor_2

def winloss(prev_pred, previous_signal):

    #take data
    data = data_collector.retrieve_data()
    #take prev close
    close_r = round(data.tail(1)['Close'].item(),2)
    #take comp day
    close_p = round(data[:-1].tail(1)['Close'].item(),2)

    real_s = 'UP' if close_p < close_r else 'DOWN'

    trade = 'WIN' if real_s == previous_signal else 'LOSS'
    out = abs(close_r - prev_pred) if real_s == previous_signal else None

    return trade,out

if __name__ == "__main__":
    LAG = 58 # days to look back
    PATH = 'C:\\Users\\Themis\\Desktop\\Cuda_bitcoin_pred\\models\\model_exp2' # model path ( in case you want to use relative path => ./models/model_exp1 )
    pred = predictor_2(LAG = 58) #object predictor
    signal, prediction, end_date, prev_close, prev_date = pred.predict(PATH=PATH) #predict

    print('BTC AT: '+str(prediction)+' FOR '+end_date+' SIGNAL: '+signal+' PREVIOUS CLOSE AT: '+str(prev_close)+' FOR '+prev_date)


    prev_p = 39960 #χθεσινη προβλεψη (θα τα περνεις απο json)
    prev_s = 'UP'  #χθεσινο σημα (θα το περνεις απο json

    initial_trade = None # Αν κανουμε την πρωτη μας προβλεψη (πχ με ενα νεο μποτ) επειδη δεν εχουμε χθεσινη (προηγουμενη) προβλεψη πρεπει να αλλαξουμε την συνθηκη με το χερι

    if initial_trade is not None:
        winloss_yes,out = winloss(prev_p,prev_s) #γυρναει win αν αν ηταν σωστο το signal (δηλαδη αν επεσε η ανεβηκε και οντως το προβλεψε σωστα) και out ποσο μακρια επεσε. Αν επεσε εξω τοτε το out ειναι None

