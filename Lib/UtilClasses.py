from itertools import product


class ParameterGrid:
    def __init__(self, paramDict: dict) -> list[dict]:
        # Top-level list items in the dict are considered alternative

        param_dicts = []
        param_combinations = product(*paramDict.values())

        for combination in param_combinations:
            params = {param_name: value for param_name,
                      value in zip(paramDict.keys(), combination)}
            param_dicts.append(params)

        for idx, experiment in param_dicts:
            experiment["id"] = idx

        return param_dicts


class BaseModelOutputWithPoolingToTensor(nn.Module):
    # Convert Hugging Face model outputs to tensor, for use in an nn.Sequential.

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.pooler_output
