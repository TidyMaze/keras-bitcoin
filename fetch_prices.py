from ast import parse
from datetime import datetime
from csv_loader import load as load_csv
from pair_history import PairHistory
from price_item import PriceItem
from dateutil.parser import parse


def load():
    data = load_csv('BTC-USD-5years.csv')
    data.reverse()
    # print(data)
    history = [PriceItem(datetime.strptime(item['Date'], '%d/%m/%Y'), float(
        item['Dernier'].replace(".", "").replace(",", "."))) for item in data]
    return PairHistory('btc/usdt', history)
