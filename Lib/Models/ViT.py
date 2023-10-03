import torch
from ..DataHandler import DataHandler
from ..BaseModel import BaseModel
from torch.utils.data import Dataset


class ViT(BaseModel):

    custom_settings_dict = {
        "vitPretrainedModel": "google/vit-base-patch16-224-in21k",
        "classifierHead": {
            "name": "torch.nn.Linear",
            "inst": "False"
        },
        "selfAttentionLayer": "5",
        "maxWindowSize": "0.5",
        "interpolationLimit": "1",
        "trainAvgThreshold": "0.4",
        "vitNumHeads": "12",
        "embedding_dim": "768"
    }
    def __init__(self):
        self.config = None

    def get_custom_params(self) -> dict:
        return {}

    def set_config(self, config) -> None:
        self.config = config

    def load_transform_data(self, train_transform, test_transform, dataset):
        dh = DataHandler(self.config)
        self.train_data: Dataset = dh.getDataset(
            self.config["train_transform"], self.config["train_path"], self.config["train_folder_type"])
        self.test_data: Dataset = dh.getDataset(
            self.config["test_transform"], self.config["test_path"], self.config["test_folder_type"])

    def build_model(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass
