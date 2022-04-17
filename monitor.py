from predictor import predictor

model = 'model_exp1'
if __name__ == "__main__":
    LAG = 58  # days to look back
    PATH = './' + model  # model path
    pred = predictor(LAG=58)  # object predictor
    signal, prediction, end_date, prev_close, prev_date = pred.predict(PATH=PATH)  # predict

    print('Model : ' + model
          + ' BTC Price: ' + str(prediction)
          + ' FOR ' + end_date
          + ' UP/DOWN: ' + signal
          + ' PREVIOUS CLOSE AT: ' + str(prev_close)
          + ' FOR '+prev_date
          )

