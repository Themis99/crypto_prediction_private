import os
import random
import time
from os import path

import data_exporter_library


# We use this function to generate random data from
def generate_mock_data(model):
    for x in range(1, 10):
        random_number = random.randint(1, 30)
        random_btc_price = random.randint(35000, 45000)

        signal = 'UP' if random.randint(1, 2) == 1 else 'DOWN'
        prediction = str(random_btc_price) if signal == 'UP' else str((random_btc_price - random.randint(500, 1600)))
        prev_close = str((random_btc_price - random.randint(500, 1600))) if signal == 'UP' else str(random_btc_price)

        prev_date = str((random_number - 1)) + '-05-2022'
        end_date = str(random_number) + '-05-2022'

        time.sleep(0.25)
        data_exporter_library.export_data(True, model, prediction, prev_close, signal, prev_date, end_date)


if __name__ == "__main__":
    # Delete fake data json if it exists
    file_path = './fake_data/data.json'
    if path.exists(file_path):
        os.remove(file_path)

    generate_mock_data('model_exp1')
    generate_mock_data('model_exp2')
    generate_mock_data('model_exp3')
