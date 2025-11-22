from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            # Support HuggingFace authentication via token
            token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            config_file = hf_hub_download(path, "pipeline.json", token=token)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            try:
                if is_local:
                    _models[k] = models.from_pretrained(f"{path}/{v}")
                else:
                    # For HuggingFace, we need to handle the path construction carefully
                    # If 'v' is a relative path inside the repo, we construct the full HF path
                    # But models.from_pretrained expects "repo_id/path/to/model"
                    _models[k] = models.from_pretrained(f"{path}/{v}")
            except Exception as e:
                # Fallback logic
                print(f"[WARNING] Failed to load model {k} from {path}/{v}: {e}")
                print(f"[INFO] Attempting fallback load for {k}...")
                if '/' in v:
                     _models[k] = models.from_pretrained(v)
                else:
                    raise e

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
