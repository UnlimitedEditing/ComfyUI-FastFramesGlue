# ComfyUI-FastFramesGlue

Glue nodes that run Apple's **FastFrames-S** (1.3B, two-step autoregressive text-to-video, 832x480, 81 frames, 16 fps)
inside ComfyUI. Contains no Apple code or weights.

- Needs `https://github.com/apple-aiml-research/ml-fastframes` cloned next to this repo in `custom_nodes/` (Apple Sample Code License).
- Needs weights under `ComfyUI/models/fastframes/` (layout of Apple's `checkpoints/` dir):
  - `Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1-T2V-1.3B/google/umt5-xxl/*` (Wan-AI/Wan2.1-T2V-1.3B)
  - `causal-forcing++/framewise-2step.pt` (zhuhz22/Causal-Forcing)
  - `fastframes-s/kv_refiner.safetensors`, `fastframes-s/pixel_decoder.safetensors` (apple/FastFrames-S)
- Python deps: `einops`, `ftfy`, `regex`, `transformers` (tokenizer only), `safetensors`. Do not install Apple's pinned torch.

Nodes: `FastFrames S Loader` -> `FastFrames S Text to Video` (IMAGE batch; wire to CreateVideo at 16 fps).
`length` is 4k+1 frames (5-81). `width`/`height` default to 832x480 (the trained size); other sizes are untested.
