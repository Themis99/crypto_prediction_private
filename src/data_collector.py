from src import yfdata

def retrieve_data():
    data = yfdata.yahoo_retriever()
    return data

def retrieve_data2(past):
    data = yfdata.yahoo_retriever()
    data = data[:-past]
    return data
