from BaseModel import BaseModel
import Utils
from Lib.DataHandler import DataHandler
from torch.utils.data import Dataset
from transformers import ViTModel, ViTConfig, ViTForImageClassification, AutoModel, ViTImageProcessor
from torch import nn
from Lib.UtilClasses import BaseModelOutputWithPoolingToTensor


class GenericHFModel(BaseModel):

    def __init__(self, config: dict):
        self.config = config
        self.train_data = None
        self.test_data = None

    def load_custom_params(self) -> dict:
        customFilePath = "../Config/GenericHFModel.yaml"
        return Utils.parseYAML(customFilePath)

    def build_model(self):
        modelComponent = Utils.instantialise(self.config["hfModel"])

        hfModel = nn.Sequential(
            modelComponent,
            BaseModelOutputWithPoolingToTensor(),
            nn.LayerNorm(normalized_shape=self.embedding_dim),
            self.classifierHead,
        )

        # Load model onto GPU
        hfModel.to(self.device)

        # Initialise our optimiser function based on provided experiment parameters
        modelOptimiser = self.optimiser(
            hfModel.parameters(), lr=self.learningRate)

    def train(self):
        pass

    def evaluate(self):
        pass
