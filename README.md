# Comfy EverAnimate

Native ComfyUI custom nodes for experimenting with EverAnimate-style latent anchor and motion memory on top of native WanAnimate workflows.

## Nodes

- **Comfy EverAnimate**: native `WanAnimateToVideo`-style conditioning node with EverAnimate anchor latents, latent motion memory, pose strength, and face strength.
- **Comfy EverAnimate Trim Images**: trims duplicated handoff frames after VAE decode.

## Intended Workflow

```text
Comfy EverAnimate -> native KSampler -> TrimVideoLatent -> VAEDecode -> Comfy EverAnimate Trim Images
```

For the next chunk, connect the previous native sampler output into the next **Comfy EverAnimate** `prev_samples` input.

## Defaults

- `num_video_anchor_latents`: `4`
- `num_motion_latents`: `1`
- `video_frame_offset`: `0`
- `pose_strength`: `1.0`
- `face_strength`: `1.0`

## Install

Clone this repository into your ComfyUI `custom_nodes` folder, then restart ComfyUI.

```bash
git clone https://github.com/younestft/Comfy_EverAnimate.git Comfy-EverAnimate
```

This node pack is native-ComfyUI focused and does not depend on WanVideoWrapper.
