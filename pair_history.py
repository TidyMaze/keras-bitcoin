from dataclasses import dataclass

from price_item import PriceItem


@dataclass
class PairHistory:
    pair: str
    history: list[PriceItem]

    def __repr__(self):
        return f"{self.pair}:\n{self.history}"
