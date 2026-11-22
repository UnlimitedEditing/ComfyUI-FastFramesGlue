"""ComfyUI glue for Apple's FastFrames-S text-to-video model (github.com/apple-aiml-research/ml-fastframes).

This repo contains NO Apple code or weights. It imports the `fastframes` package from a sibling checkout of
apple-aiml-research/ml-fastframes (cloned into custom_nodes next to this repo) and reads the weights from
ComfyUI/models/fastframes/ (staged by Graydient's concept_mapping).
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
