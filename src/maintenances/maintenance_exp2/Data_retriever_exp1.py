import yahoo_fandg

def retrieve_data():
    data = yahoo_fandg.yahoo_retriever()
    data = data[:-1]
    return data
