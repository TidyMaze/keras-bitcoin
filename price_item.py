from dataclasses import dataclass
from datetime import datetime


@dataclass
class PriceItem:
    price: float
    timestamp: datetime

    def __repr__(self):
        return f"{self.price} at {self.timestamp}"
