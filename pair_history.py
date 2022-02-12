from dataclasses import dataclass


@dataclass
class PairHistory:
    pair: str
    history: list

    def __repr__(self):
        return f"{self.pair}:\n{self.history}"
