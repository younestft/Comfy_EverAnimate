# Comfy EverAnimate

Native ComfyUI custom nodes for experimenting with EverAnimate-style latent anchors plus native WanAnimate image carry-over on top of native sampler workflows.

## Nodes

- **Comfy EverAnimate**: native `WanAnimateToVideo`-style conditioning node with EverAnimate anchor latents, native image carry-over, optional latent motion memory, pose strength, and face strength.
- **EverAnimate Master Settings**: collects shared model, video, sampler, anchor, pose/face, and guide settings into an initial settings socket.
- **EverAnimate Initial Chunk**: generates the first chunk from the initial settings and owns the startup carry setting.
- **EverAnimate Extension Chunk**: generates later chunks from the previous cumulative images, owns the handoff settings, and carries sampler state forward.
- **EverAnimate Color Correction**: applies native Transfer Color only around chunk boundaries to reduce color/brightness flashes without changing the whole video.
- **EverAnimate Chunks Calculator**: previews the total number of chunks needed from the Master pose video frame count and chunk length.

## Intended Workflow

```text
Comfy EverAnimate -> native KSampler -> TrimVideoLatent -> VAEDecode -> Video Combine
```

For smoother chunk boundaries, connect the previous chunk's trimmed decoded images into the next **Comfy EverAnimate** `continue_motion` input. This uses native WanAnimate-style image carry-over and takes priority over `prev_samples`.

For chunk 1, you can connect the reference image into `continue_motion` and set `continue_motion_max_frames` to `1` to reduce startup flashes. For later chunks, `continue_motion_max_frames` defaults to `5`.

## Master + Chunk Workflow

```text
EverAnimate Master Settings -> EverAnimate Initial Chunk -> EverAnimate Extension Chunk -> Video Combine
```

Connect the Master `initial settings` output to the Initial Chunk `initial settings` input. The Initial Chunk then outputs normal `settings` for Extension Chunk nodes. These two settings sockets are intentionally different types, so the initial settings cannot be plugged into Extension chunks and normal settings cannot be plugged back into Initial.

For longer videos, add more **EverAnimate Extension Chunk** nodes. Connect each previous chunk `settings` output to the next chunk `settings` input, and each previous chunk `images` output to the next chunk `images` input. The final Video Combine connects to the last chunk `images`.

Optional chunk calculation:

```text
EverAnimate Master Settings chunks calculator -> EverAnimate Chunks Calculator
```

The calculator previews `Total chunks needed` followed by the bold chunk count inside the node. Its output socket is an `INT` containing only the chunk count. It rounds partial chunks up and counts the Initial Chunk as chunk 1.

Optional boundary color correction:

```text
Last Extension Chunk images + settings -> EverAnimate Color Correction -> Video Combine
```

Use the same reference images you would use with native `Transfer Color`. The correction node defaults to 12 frames before each join, 16 frames after each join, `mkl_lab`, `per_frame`, strength `1.0`, and only processes frames near each chunk join.

The initial chunk has no image input because it uses the reference image startup carry. Extension chunks use the previous cumulative images plus latent motion memory stored in the settings socket.

The seed is in Master Sampling Settings and is shared by all chunks. Keep it fixed across chunks for the most stable boundaries. Chunk nodes do not encode previews internally; connect any chunk `images` output to a normal Preview Image or Video Combine node when you want to inspect progress.

Master `chunk length` is the per-chunk Wan window length, not the total final video length. `ref image background` is at the top of the video settings, defaults to enabled, and when it is on, the Master ignores connected `background_video` and `character_mask` inputs.

The flash-sensitive settings are on the chunk nodes:

- Initial and Extension chunks have a `use custom settings` toggle. When it is off, the custom rows stay visible but are locked/gray and the backend uses the original defaults. Turn it on to unlock those rows and use your custom values.
- Initial chunk: `startup_carry_frames` defaults to `1`, so the first generated frames are anchored to the reference image instead of starting from an empty handoff.
- Extension chunk: `num_motion_latents` defaults to `1`, `continue_motion_max_frames` defaults to `5`, and `motion_handoff_strength` defaults to `0.75`. This keeps one latent of motion memory and softly carries five decoded frames across the boundary.
- Extension chunks use WanAnimate's returned `video_frame_offset` internally so pose, face, background, and mask guides stay aligned after carry-frame backtracking.

## Defaults

- `num_video_anchor_latents`: `4`
- `video_frame_offset`: `0`
- Master `width`: `480`
- Master `height`: `832`
- Master `chunk length`: `81`
- Master `ref image background`: `true`
- Master `seed`: `42`, fixed
- `pose_strength`: `1.0`
- `face_strength`: `1.0`
- Initial `use custom settings`: `false`
- Initial `startup_carry_frames`: `1`
- Extension `use custom settings`: `false`
- Extension `num_motion_latents`: `1`
- Extension `continue_motion_max_frames`: `5`
- Extension `motion_handoff_strength`: `0.75`
- Color Correction `frames_before`: `12`
- Color Correction `frames_after`: `16`
- Color Correction `method`: `mkl_lab`
- Color Correction `source_stats`: `per_frame`
- Color Correction `strength`: `1.0`

## Install

Clone this repository into your ComfyUI `custom_nodes` folder, then restart ComfyUI.

```bash
git clone https://github.com/younestft/Comfy_EverAnimate.git Comfy-EverAnimate
```

This node pack is native-ComfyUI focused and does not depend on WanVideoWrapper.
