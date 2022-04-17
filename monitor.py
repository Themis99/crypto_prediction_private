from predictor import predictor

if __name__ == "__main__":
    LAG = 58 # days to look back
    PATH = 'C:\\Users\\Themis\\Desktop\\Cuda_bitcoin_pred\\models\\model_exp1' # model path
    pred = predictor(LAG = 58) #object predictor
    signal, prediction, end_date, prev_close, prev_date = pred.predict(PATH=PATH) #predict

    print('BTC AT: '+str(prediction)+' FOR '+end_date+' SIGNAL: '+signal+' PREVIOUS CLOSE AT: '+str(prev_close)+' FOR '+prev_date)

