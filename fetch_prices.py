from ast import parse
from datetime import datetime
from csv_loader import load as load_csv
from pair_history import PairHistory
from price_item import PriceItem
from dateutil.parser import parse

def parse_numbers_with_suffix(raw):
    if 'K' in raw:
        return float(raw.replace('K', '')) * 1000
    elif 'M' in raw:
        return float(raw.replace('M', '')) * 1000000
    elif 'B' in raw:
        return float(raw.replace('B', '')) * 1000000000
    else:
        return float(raw)

def parseNumberWithSerparators(raw):
    return float(raw.replace(".", "").replace(',', '.'))

def load():
    data = load_csv('BTC-USD-5years.csv')
    data.reverse()
    # print(data)
    history = [
        PriceItem(
            datetime.strptime(item['Date'], '%d/%m/%Y'),
            parseNumberWithSerparators(item['Ouv.']),
            parseNumberWithSerparators(item['Dernier']),
            parseNumberWithSerparators(item['Plus Haut']),
            parseNumberWithSerparators(item['Plus Bas']),
            parse_numbers_with_suffix(item['Vol.'].replace(",", "."))
        )
        for item in data
    ]
    return PairHistory('btc/usdt', history)
