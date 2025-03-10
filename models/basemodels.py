from abc import ABC, abstractmethod
from typing import Any, Self

import torch

from models.types import Device


class BaseModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def _load_model(self) -> Any: ...


class TorchModel(BaseModel):

    def __init__(self, device: Device = None) -> None:
        super().__init__()
        self.device = device

    @abstractmethod
    def _load_model(self) -> Any: ...

    def to(self, device: Device) -> Self:
        self.device = device
        self.model = self.model.to(device)
        return self

    def cpu(self) -> Self:
        self.device = torch.device("cpu")
        self.model = self.model.cpu()
        return self

    def cuda(self) -> Self:
        self.device = torch.device("cuda")
        self.model = self.model.cuda()
        return self

    def train(self) -> Self:
        self.model.train()
        return self

    def eval(self) -> Self:
        self.model.eval()
        return self

    def half(self) -> Self:
        self.model.half()
        return self

    def float(self) -> Self:
        self.model.float()
        return self

    def double(self) -> Self:
        self.model.double()
        return self
