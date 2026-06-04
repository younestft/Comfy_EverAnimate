# Comfy EverAnimate

Native ComfyUI custom nodes for experimenting with EverAnimate-style latent anchors plus native WanAnimate image carry-over on top of native sampler workflows.

## Nodes

- **Comfy EverAnimate**: native `WanAnimateToVideo`-style conditioning node with EverAnimate anchor latents, native image carry-over, optional latent motion memory, pose strength, and face strength.
- **Comfy EverAnimate Trim Images**: trims duplicated handoff frames after VAE decode.
- **EverAnimate AIO**: runs chunked native WanAnimate sampling internally, carries decoded images plus latent motion memory between chunks, optionally saves temp MP4 previews after each completed chunk, and returns one final image batch.
- **EverAnimate Master Settings**: collects shared model, video, sampler, anchor, pose/face, and guide settings into an initial settings socket.
- **EverAnimate Initial Chunk**: generates the first chunk from the initial settings, owns the startup carry settings, and displays how many extension chunks are needed when a guide length is connected.
- **EverAnimate Extension Chunk**: generates later chunks from the previous cumulative images, owns the handoff settings, and carries sampler state forward.

## Intended Workflow

```text
Comfy EverAnimate -> native KSampler -> TrimVideoLatent -> VAEDecode -> Comfy EverAnimate Trim Images
```

For smoother chunk boundaries, connect the previous chunk's trimmed decoded images into the next **Comfy EverAnimate** `continue_motion` input. This uses native WanAnimate-style image carry-over and takes priority over `prev_samples`.

For chunk 1, you can connect the reference image into `continue_motion` and set `continue_motion_max_frames` to `1` to reduce startup flashes. For later chunks, `continue_motion_max_frames` defaults to `5`.

## EverAnimate AIO Workflow

```text
EverAnimate AIO -> Video Combine
```

The AIO node keeps the same native sampler controls inside the node, so connect your Wan model, positive/negative conditioning, VAE, reference image, and optional pose/face/background/mask guides directly to it.

Chunk modes:

- `auto_target_frames`: calculates enough chunks to reach `target_frames`.
- `auto_guide_length`: uses the shortest connected guide length from pose, face, background, or mask.
- `custom_chunks`: runs exactly `chunk_count` chunks.

When `save_chunk_previews` is enabled, the AIO node uses VideoHelperSuite to save temporary MP4 previews named like:

```text
Comfy_EverAnimate_AIO_chunk_001_preview.mp4
Comfy_EverAnimate_AIO_chunk_002_preview.mp4
```

Chunk previews require ComfyUI-VideoHelperSuite. The AIO node does not encode the final video or attach audio; connect its final `images` output to your normal Video Combine node for the final audio/video.

## Master + Chunk Workflow

```text
EverAnimate Master Settings -> EverAnimate Initial Chunk -> EverAnimate Extension Chunk -> Video Combine
```

Connect the Master `initial settings` output to the Initial Chunk `initial settings` input. The Initial Chunk then outputs normal `settings` for Extension Chunk nodes. These two settings sockets are intentionally different types, so the initial settings cannot be plugged into Extension chunks and normal settings cannot be plugged back into Initial.

For longer videos, add more **EverAnimate Extension Chunk** nodes. Connect each previous chunk `settings` output to the next chunk `settings` input, and each previous chunk `images` output to the next chunk `images` input. The final Video Combine connects to the last chunk `images`.

The initial chunk has no image input because it uses the reference image startup carry. Extension chunks use the previous cumulative images plus latent motion memory stored in the settings socket.

Connect an INT value to the Master `frame count` socket under `video_anchor_latent`, then click `calculate chunks` on the Initial Chunk. It shows `extension chunks needed` inside the node without running generation. If an Extension Chunk is already connected, the button uses that node's `continue_motion_max_frames`; otherwise it assumes the default `5`.

The seed is in Master Sampling Settings and is shared by all chunks. Keep it fixed across chunks for the most stable boundaries. Chunk nodes do not encode previews internally; connect any chunk `images` output to a normal Preview Image or Video Combine node when you want to inspect progress.

Master `chunk length` is the per-chunk Wan window length, not the total final video length. `ref image background` defaults to enabled; when it is on, the Master ignores connected `background_video` and `character_mask` inputs.

The flash-sensitive settings are on the chunk nodes:

- Initial chunk: `startup_carry_frames` defaults to `1`, so the first generated frames are anchored to the reference image instead of starting from an empty handoff.
- Extension chunk: `num_motion_latents` defaults to `1`, `continue_motion_max_frames` defaults to `5`, and `motion_handoff_strength` defaults to `0.75`. This keeps one latent of motion memory and softly carries five decoded frames across the boundary.

## Defaults

- `num_video_anchor_latents`: `4`
- `video_frame_offset`: `0`
- Master `width`: `480`
- Master `height`: `832`
- Master `chunk length`: `81`
- Master `frame count`: optional INT socket, default `0` when unconnected
- Master `ref image background`: `true`
- Master `seed`: `42`, fixed
- `pose_strength`: `1.0`
- `face_strength`: `1.0`
- Initial `startup_carry_frames`: `1`
- Extension `num_motion_latents`: `1`
- Extension `continue_motion_max_frames`: `5`
- Extension `motion_handoff_strength`: `0.75`
- AIO `startup_carry_frames`: `1`
- AIO `save_chunk_previews`: `true`

## Install

Clone this repository into your ComfyUI `custom_nodes` folder, then restart ComfyUI.

```bash
git clone https://github.com/younestft/Comfy_EverAnimate.git Comfy-EverAnimate
```

This node pack is native-ComfyUI focused and does not depend on WanVideoWrapper.
