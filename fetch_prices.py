from ast import parse
from datetime import datetime
from csv_loader import load as load_csv
from pair_history import PairHistory
from price_item import PriceItem
from dateutil.parser import parse


def load():
    data = load_csv('bitcoin_history_2021.csv')
    # print(data)
    return PairHistory('btc/usdt', list(map(lambda item: PriceItem(datetime.strptime(item['Date'], '%d/%m/%Y'), float(item['Dernier'].replace(".", "").replace(",", "."))), data)))
