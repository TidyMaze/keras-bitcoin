from dataclasses import dataclass
from datetime import datetime


@dataclass
class PriceItem:
    timestamp: datetime
    price_first: float
    price_last: float
    price_min: float
    price_max: float
    volume: float

    def __repr__(self):
        return f"{self.timestamp} {self.price_first} {self.price_last} {self.price_min} {self.price_max} {self.volume}"
