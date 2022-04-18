import random
import data_exporter


# We use this function to generate random data from
def generate_mock_data(model):
    for x in range(1, 30):
        random_number = random.randint(1, 30)
        random_btc_price = random.randint(35000, 45000)
        prediction = str(random_btc_price)
        prev_close = str((random_btc_price - 1250))
        signal = 'UP'
        prev_date = str((random_number - 1)) + '-05-2022'
        end_date = str(random_number) + '-05-2022'

        data_exporter.export_data(True, model, prediction, prev_close, signal, prev_date, end_date)


if __name__ == "__main__":
    generate_mock_data('model_exp1')
    generate_mock_data('model_2')
    generate_mock_data('model_3')
