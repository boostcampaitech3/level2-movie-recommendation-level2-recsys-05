import torch


class Optimizer:
    def __init__(self) -> None:
        self.adam = torch.optim.Adam
