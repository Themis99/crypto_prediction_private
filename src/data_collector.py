from src import yfdata

def retrieve_data():
    data = yfdata.yahoo_retriever()
    data = data[:-1]
    return data
