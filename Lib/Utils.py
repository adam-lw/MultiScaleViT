import torchvision
from typing import Callable
import yaml
from importlib import import_module
import sys


def assembleTransforms(inputLists: list[str]) -> list:
    outputTransforms = []

    # For each list of transform components, assemble them to a single transform object
    for sublist in inputLists:
        assembleTransform(sublist)

    def assembleTransform(dictList):
        transformList = []
        for subdict in dictList:
            transformList.append(import_string(
                f"torchvision.transforms.{subdict['name']}")(eval(subdict["params"])))
        print(transformList)


def parseYAML(filepath: str) -> dict:
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
        if config is None:
            raise FileNotFoundError("Provided YAML file has not been found")
    return config


def writeYAML(data: dict, filepath: str) -> None:

    with open(filepath, "w") as yaml_file:
        print(f"Writing YAML to {filepath}")
        yaml.dump(data, yaml_file)

# From Django source code


def import_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """

    print(f"Trying to import {dotted_path}")

    def cached_import(module_path, class_name):
        # Check whether module is loaded and fully initialized.
        if not (
            (module := sys.modules.get(module_path))
            and (spec := getattr(module, "__spec__", None))
            and getattr(spec, "_initializing", False) is False
        ):
            module = import_module(module_path)
        return getattr(module, class_name)

    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" %
                          dotted_path) from err

    try:
        return cached_import(module_path, class_name)
    except AttributeError as err:
        raise ImportError(
            'Module "%s" does not define a "%s" attribute/class'
            % (module_path, class_name)
        ) from err

def instantiateSettings(self, localParams):
    # Assemble transforms
    localParams["train_transforms"] = assembleTransforms(
        self.experimentParams["train_transforms"])
    localParams["test_transforms"] = assembleTransforms(
        self.experimentParams["test_transforms"])
    
    # Assemble settings 
    # If the setting key begins with 'c_', we 