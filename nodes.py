import dataclasses
import glob
import os
import sys

import torch

import folder_paths

ROOT = "fastframes"  # under ComfyUI/models/


def _import_fastframes():
    """Find Apple's `fastframes` package in a sibling custom_nodes checkout (ml-fastframes) and import it."""
    try:
        import fastframes.pipeline as pipeline  # noqa: F401
        return pipeline
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    custom_nodes = os.path.dirname(here)
    hits = sorted(glob.glob(os.path.join(custom_nodes, "*", "fastframes", "pipeline.py")))
    if not hits:
        raise RuntimeError(
            "Apple's fastframes package was not found. Clone https://github.com/apple-aiml-research/ml-fastframes "
            "into ComfyUI/custom_nodes/ (this node imports it from there).")
    repo_root = os.path.dirname(os.path.dirname(hits[0]))
    if repo_root not in sys.path:
        sys.path.append(repo_root)  # append, not insert: the repo also has top-level apps/ scripts/ dirs
    import fastframes.pipeline as pipeline
    return pipeline


def _p(*parts):
    return os.path.join(folder_paths.models_dir, ROOT, *parts)


class FastFramesLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("FASTFRAMES",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "FastFrames"

    def load(self):
        pipeline = _import_fastframes()
        spec = pipeline.MODELS["ff-s"]
        paths = dict(
            wan_dir=_p("Wan2.1-T2V-1.3B"),
            dit=_p(spec.dit),
            refiner=_p(spec.refiner),
            decoder=_p(spec.decoder),
        )
        for name, path in paths.items():
            if not os.path.exists(path):
                raise RuntimeError(f"FastFrames-S weight missing ({name}): {path}")
        pipeline.configure_backends(spec.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = pipeline.FastFrames.from_pretrained(
            spec.config, paths["wan_dir"], paths["dit"], paths["refiner"], paths["decoder"], device=device)
        return (model,)


class FastFramesT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("FASTFRAMES",),
            "prompt": ("STRING", {"multiline": True, "default": "A corgi surfing a wave at sunset"}),
            "rng_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "length": ("INT", {"default": 81, "min": 5, "max": 81, "step": 4}),
            "width": ("INT", {"default": 0, "min": 0, "max": 1664, "step": 16}),
            "height": ("INT", {"default": 0, "min": 0, "max": 960, "step": 16}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "FastFrames"

    def generate(self, model, prompt, rng_seed, length, width, height):
        pipeline = _import_fastframes()
        base = model.config
        # pixel frames = 4 * latent_frames - 3 (the decoder emits 4 RGB frames per latent frame and drops the first 3)
        latent_frames = max(2, (int(length) - 1) // 4 + 1)
        w = int(width) if int(width) > 0 else 832
        h = int(height) if int(height) > 0 else 480
        w, h = max(256, round(w / 16) * 16), max(256, round(h / 16) * 16)  # latent (w/8, h/8) must be even
        cfg = dataclasses.replace(
            base, num_frames=latent_frames, latent_size=(h // 8, w // 8), num_pixel_frames=4 * latent_frames - 3)
        run = pipeline.FastFrames(cfg, model.dit, model.refiner, model.decoder, model.text_encoder)

        run.text_encoder.to(run.device)
        embeds = run.encode_prompt(prompt)
        run.text_encoder.to("cpu")  # free ~11 GB of VRAM for the KV cache + decoder
        torch.cuda.empty_cache()

        _, features, modulation = run.rollout(embeds, int(rng_seed))
        pixels = run.decode(features, modulation)[0]  # [T, 3, H, W] in [-1, 1]
        images = ((pixels.float().clamp(-1, 1) + 1.0) * 0.5).permute(0, 2, 3, 1).contiguous().cpu()
        return (images,)


NODE_CLASS_MAPPINGS = {
    "FastFramesLoad": FastFramesLoad,
    "FastFramesT2V": FastFramesT2V,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FastFramesLoad": "FastFrames S Loader",
    "FastFramesT2V": "FastFrames S Text to Video",
}
