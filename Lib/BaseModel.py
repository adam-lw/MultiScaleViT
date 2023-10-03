import torch
from .DataHandler import DataHandler
from torch.utils.data import Dataset


class BaseModel:
    def __init__(self):
        self.config = None

    def get_custom_params(self) -> dict:
        return None
    
    def set_config(self, config) -> None:
        self.config = config

    def load_transform_data(self, train_transform, test_transform, dataset):
        dh = DataHandler(self.config)
        self.train_data: Dataset = dh.getDataset(self.config["train_transform"], self.config["train_path"], self.config["train_folder_type"])
        self.test_data: Dataset = dh.getDataset(self.config["test_transform"], self.config["test_path"], self.config["test_folder_type"])

    def build_model(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass
