from . import Utils
from .BaseModel import BaseModel
from .DataHandler import DataHandler
from typing import Type


class ExperimentInterface:

    # Used for creating the default settings configuration
    # This will only be used to create a new settings file

    # Any dictionaries with a "name", "params" or "inst" key will be
    # automatically loaded as classes
    default_settings_dict = {
        "modelConfigDir": "Config",
        "savePath": "SavedModels",
        "resultsDir": "Results",
        "datasetDir": "Datasets/RNSA",
        "model": "ViT",
        "train_transforms": [{
            "name": "torchvision.transforms.Compose",
            "params": [{
                "name": "torchvision.transforms.Resize.Resize",
                "params": ["(224, 224)"]
            }]
        }],
        "test_transforms": [{
            "name": "torchvision.transforms.Compose",
            "params": [{
                "name": "torchvision.transforms.Resize.Resize",
                "params": ["(224, 224)"]
            }]
        }],
        "optimiser": {
            "name": "torch.optim.AdamW",
            "inst": "False"
        },
        "lossFunction": {
            "name": "torch.nn.CrossEntropyLoss",
            "params": []
        },
        "learningRate": "0.0001",
        "useLearningRateScheduler": "False",
        "learningRateScheduler": "",
        "accumulationSize": "10",
        "maxEpochs": "30",
        "earlyStopThreshold": "5",
        "earlyStopDelta": "0.005",

    }

    def __init__(self, settingsPath: str = "") -> None:
        if (settingsPath != ""):
            self.experimentParams = Utils.parseYAML(settingsPath)
        else:
            self.experimentParams = None

    def updateConfig(self, settingsPath: str) -> None:
        self.experimentParams = Utils.parseYAML(settingsPath)

    # Run experiment based on provided settings dict
    # kwargs override specified default setting if provided
    def runExperiment(self, **kwargs):

        # Copy global/default experiment params to custom experiment config
        localParams = self.experimentParams

        # Update our model if it's provided
        if "model" in kwargs:
            localParams["model"] = kwargs["model"]

        # Instantiate model from string and append default model settings to the localParams
        foundClass = Utils.import_string(
            f"Lib.Models.{localParams['model']}.{localParams['model']}")
        if foundClass:
            print(f"Successfully found: {foundClass}")
            model: Type[BaseModel] = foundClass()
            localParams.update(model.get_custom_params())
        else:
            raise NameError("Model '" + localParams["model"] + "' not found.")

        # Update the rest of the params with provided experiment-specific params
        for key, item in kwargs.items():

            if key not in localParams.keys():
                raise KeyError("Key '" + str(key) +
                               "' not found in experiment configuration.")
            localParams[key] = item

        # Instantiate experiment settings
        localParams = Utils.instantiateSettings(localParams)

        # Provide model with config details
        model.set_config(localParams)

        # Do the pipeline
        model.load_transform_data()
        model.build_model()
        model.train()
        model.evaluate()

        print("Done!")

    def createSettingsFile(self, filepath):
        print("Creating settings file...")
        Utils.writeYAML(self.default_settings_dict, filepath)

    
