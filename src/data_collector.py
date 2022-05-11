from src import yfdata

def retrieve_data():
    data = yfdata.yahoo_retriever()
    return data

def retrieve_data_win_loss():
    data = yfdata.yahoo_retriever()
    data = data[:-1]
    return data

def retrieve_data_past_dates(past):
    data = yfdata.yahoo_retriever()
    data = data[:-past]
    return data
