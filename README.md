# Comfy EverAnimate

Native ComfyUI nodes for chunked EverAnimate-style WanAnimate generation.

The current release focuses on the Master + Initial Chunk + Extension Chunk workflow.

## Included Nodes

- **Comfy EverAnimate**: low-level WanAnimate conditioning node for manual workflows.
- **EverAnimate Master Settings**: collects the model, conditioning, VAE, video size, sampler settings, anchor settings, and guide inputs into one settings socket.
- **EverAnimate Initial Chunk**: generates the first chunk from the master settings.
- **EverAnimate Extension Chunk**: generates each later chunk from the previous cumulative images and settings.
- **EverAnimate Color Correction**: applies optional frame color matching for chunk continuity.

## Default Workflow

The packaged workflow is:

```text
examples/Comfy_EverAnimate_Chunked.json
```

It uses one Initial Chunk followed by Extension Chunk nodes. Add or remove Extension Chunk nodes to change the final duration.

Basic flow:

```text
EverAnimate Master Settings
  -> EverAnimate Initial Chunk
  -> EverAnimate Extension Chunk
  -> EverAnimate Extension Chunk
  -> Video Combine
```

The chunk nodes return cumulative image batches. Connect the final chunk image output to your video combine/output node.

## How It Works

EverAnimate keeps long WanAnimate generations more stable by carrying two kinds of memory between chunks:

- decoded image carry-over from the end of the previous chunk
- latent motion memory from the previous sampler output

The Initial Chunk uses the reference image as the startup carry frame. Later Extension Chunk nodes use the previous generated images and stored settings to continue the video.

Recommended defaults are already set in the workflow:

- `num_video_anchor_latents`: `4`
- `startup_carry_frames`: `1`
- `num_motion_latents`: `1`
- `continue_motion_max_frames`: `5`
- `motion_handoff_strength`: `0.75`
- `chunk_length`: `81`
- `steps`: `4`
- `cfg`: `1.0`

## Workflow Dependencies

This node pack provides only the EverAnimate nodes listed above. The default workflow also uses common ComfyUI helper packs, including:

- ComfyUI-KJNodes
- ComfyUI-VideoHelperSuite
- rgthree-comfy
- SAM2 / segmentation nodes
- pose and face detection nodes
- Wan model, CLIP, VAE, and LoRA loader nodes from your ComfyUI setup

Install any missing workflow nodes through ComfyUI Manager.

## Install

Clone this repository into your ComfyUI `custom_nodes` folder:

```bash
git clone https://github.com/younestft/Comfy_EverAnimate.git Comfy_EverAnimate
```

Restart ComfyUI after installation or update.

## Notes

- This pack is native-ComfyUI focused.
- It does not depend on WanVideoWrapper.
- Trimming is handled inside the chunk nodes.
