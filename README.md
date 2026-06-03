# Comfy EverAnimate

Native ComfyUI custom nodes for experimenting with EverAnimate-style latent anchors plus native WanAnimate image carry-over on top of native sampler workflows.

## Nodes

- **Comfy EverAnimate**: native `WanAnimateToVideo`-style conditioning node with EverAnimate anchor latents, native image carry-over, optional latent motion memory, pose strength, and face strength.
- **Comfy EverAnimate Trim Images**: trims duplicated handoff frames after VAE decode.

## Intended Workflow

```text
Comfy EverAnimate -> native KSampler -> TrimVideoLatent -> VAEDecode -> Comfy EverAnimate Trim Images
```

For smoother chunk boundaries, connect the previous chunk's trimmed decoded images into the next **Comfy EverAnimate** `continue_motion` input. This uses native WanAnimate-style image carry-over and takes priority over `prev_samples`.

For chunk 1, you can connect the reference image into `continue_motion` and set `continue_motion_max_frames` to `1` to reduce startup flashes. For later chunks, `continue_motion_max_frames` defaults to `5`.

## Defaults

- `num_video_anchor_latents`: `4`
- `num_motion_latents`: `1`
- `video_frame_offset`: `0`
- `pose_strength`: `1.0`
- `face_strength`: `1.0`
- `motion_handoff_strength`: `1.0`
- `continue_motion_max_frames`: `5`

## Install

Clone this repository into your ComfyUI `custom_nodes` folder, then restart ComfyUI.

```bash
git clone https://github.com/younestft/Comfy_EverAnimate.git Comfy-EverAnimate
```

This node pack is native-ComfyUI focused and does not depend on WanVideoWrapper.
