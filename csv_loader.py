import csv


def load(filename):
    """
    Loads a CSV file.
    """
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=',')
        data = list(reader)
    return data
