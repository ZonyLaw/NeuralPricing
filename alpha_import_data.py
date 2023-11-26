import requests
import pandas as pd
from email.message import EmailMessage
from config import ALPHA_API
import numpy as np


def market_price(from_currency, to_currency):
    url = "https://www.alphavantage.co/query"

    API_KEY = ALPHA_API
    url_params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_currency,
        "to_currency": to_currency,
        "apikey": API_KEY,
    }

    try:
        extract = requests.get(url, params=url_params)
        data = extract.json()
        normal_data = pd.json_normalize(data)
        print(normal_data)
        return normal_data.iloc[0, 4], normal_data
    except:
        print('fail to retrieve')

def save_data(data, file_name):

    # Specify the file name and mode ('a' for append)
    with open(file_name, 'a') as file:
        file.write(data)

    print("Data appended to", file_name)


def save_as_csv(data, file_name):
    
    data.to_csv(file_name, index=False, sep='\t')


def main():
    
    
    fromCur = "USD"
    toCur = "JPY"
    currentPrice, all_prices = market_price(fromCur, toCur)
    currentPrice = float(currentPrice)
    print(currentPrice)
    save_as_csv(all_prices, "USDJPY_prices.txt")



if __name__ == '__main__':
    main()