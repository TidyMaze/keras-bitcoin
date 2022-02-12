from dataclasses import dataclass
from datetime import datetime


@dataclass
class PriceItem:
    timestamp: datetime
    price: float

    def __repr__(self):
        return f"{self.price} at {self.timestamp}"
