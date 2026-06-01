import logging

import comfy.model_management
import comfy.utils
import node_helpers
import nodes
import torch
from comfy_api.latest import io


log = logging.getLogger(__name__)


def _latent_samples(latent, name):
    if latent is None:
        return None
    samples = latent.get("samples") if isinstance(latent, dict) else latent
    if samples is None:
        raise ValueError(f"{name} does not contain latent samples.")
    if samples.ndim != 5:
        raise ValueError(f"{name} must be a LATENT with shape [B, C, T, H, W].")
    if samples.shape[1] != 16:
        raise ValueError(f"{name} must have 16 channels for Wan latents, got {samples.shape[1]}.")
    return samples


def _normalize_temporal_count(samples, count, name):
    if samples.shape[2] == count:
        return samples
    if samples.shape[2] > count:
        return samples[:, :, :count]
    if samples.shape[2] == 0:
        raise ValueError(f"{name} has no latent frames.")

    pad = samples[:, :, -1:].repeat((1, 1, count - samples.shape[2], 1, 1))
    log.warning("%s had fewer latent frames than requested; repeating the last frame.", name)
    return torch.cat((samples, pad), dim=2)


def _assert_latent_geometry(samples, latent_height, latent_width, name):
    if samples.shape[-2] != latent_height or samples.shape[-1] != latent_width:
        raise ValueError(
            f"{name} has latent size {samples.shape[-1]}x{samples.shape[-2]}, "
            f"but this node needs {latent_width}x{latent_height}. "
            "Use the same width/height as the previous chunk."
        )


def _upscale_images(images, width, height, length):
    return comfy.utils.common_upscale(
        images[:length].movedim(-1, 1),
        width,
        height,
        "area",
        "center",
    ).movedim(1, -1)


class ComfyEverAnimate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyEverAnimate",
            display_name="Comfy EverAnimate",
            category="model/conditioning/video_models",
            description="Native WanAnimate conditioning node with EverAnimate anchor and latent motion memory.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=77, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Int.Input(
                    "num_video_anchor_latents",
                    default=4,
                    min=1,
                    max=16,
                    step=1,
                    tooltip="EverAnimate N. The paper/LoRA default is 4 identity anchor latent slots.",
                ),
                io.Int.Input(
                    "num_motion_latents",
                    default=1,
                    min=0,
                    max=16,
                    step=1,
                    tooltip="EverAnimate M. Number of previous sampler latents to carry into the next chunk.",
                ),
                io.Int.Input(
                    "video_frame_offset",
                    default=0,
                    min=0,
                    max=nodes.MAX_RESOLUTION,
                    step=1,
                    tooltip="Connect this from the previous Comfy EverAnimate video_frame_offset output.",
                ),
                io.Float.Input(
                    "pose_strength",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.001,
                    tooltip="Multiplier for native WanAnimate pose latents. 0 disables pose influence.",
                ),
                io.Float.Input(
                    "face_strength",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.001,
                    tooltip="Strength for native WanAnimate face guide. 0 sends a neutral face guide.",
                ),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("reference_image", optional=True),
                io.Image.Input("face_video", optional=True),
                io.Image.Input("pose_video", optional=True),
                io.Image.Input("background_video", optional=True),
                io.Mask.Input("character_mask", optional=True),
                io.Latent.Input(
                    "prev_samples",
                    optional=True,
                    tooltip="Previous native Wan sampler output. The node takes the last M latents as motion memory.",
                ),
                io.Latent.Input(
                    "video_anchor_latent",
                    optional=True,
                    tooltip="Advanced: prebuilt N anchor latents. Leave empty to repeat the reference image latent.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_latent"),
                io.Int.Output(display_name="trim_image"),
                io.Int.Output(display_name="video_frame_offset"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        num_video_anchor_latents,
        num_motion_latents,
        video_frame_offset,
        pose_strength,
        face_strength,
        reference_image=None,
        clip_vision_output=None,
        face_video=None,
        pose_video=None,
        background_video=None,
        character_mask=None,
        prev_samples=None,
        video_anchor_latent=None,
    ) -> io.NodeOutput:
        latent_length = ((length - 1) // 4) + 1
        latent_width = width // 8
        latent_height = height // 8
        anchor_count = int(num_video_anchor_latents)
        motion_count = min(int(num_motion_latents), latent_length)
        pose_strength = float(pose_strength)
        face_strength = float(face_strength)

        prev_latents = _latent_samples(prev_samples, "prev_samples")
        has_motion_memory = prev_latents is not None and motion_count > 0
        trim_image = (motion_count - 1) * 4 + 1 if has_motion_memory else 0
        effective_frame_offset = max(0, int(video_frame_offset) - trim_image)

        if reference_image is None:
            reference_image = torch.zeros((1, height, width, 3))

        if video_anchor_latent is not None:
            anchor_stack = _latent_samples(video_anchor_latent, "video_anchor_latent")[:1]
            _assert_latent_geometry(anchor_stack, latent_height, latent_width, "video_anchor_latent")
            anchor_stack = _normalize_temporal_count(anchor_stack, anchor_count, "video_anchor_latent")
        else:
            ref_image = _upscale_images(reference_image[:1], width, height, 1)
            anchor_stack = vae.encode(ref_image[:, :, :, :3]).repeat((1, 1, anchor_count, 1, 1))

        anchor_stack = anchor_stack.to(dtype=anchor_stack.dtype)
        cond_device = anchor_stack.device
        cond_dtype = anchor_stack.dtype

        image = torch.ones((length, height, width, 3), device=cond_device, dtype=reference_image.dtype) * 0.5

        if background_video is not None and background_video.shape[0] > effective_frame_offset:
            background_video = background_video[effective_frame_offset:]
            background_video = _upscale_images(background_video, width, height, length).to(device=cond_device)
            if background_video.shape[0] > trim_image:
                image[trim_image:background_video.shape[0]] = background_video[trim_image:]

        window_latents = vae.encode(image[:, :, :, :3]).to(device=cond_device, dtype=cond_dtype)

        if has_motion_memory:
            prev_latents = prev_latents[:1]
            _assert_latent_geometry(prev_latents, latent_height, latent_width, "prev_samples")
            motion_stack = prev_latents[:, :, -motion_count:].to(device=cond_device, dtype=cond_dtype)
            window_latents[:, :, :motion_count] = motion_stack

        anchor_mask = torch.zeros(
            (1, 4, anchor_count, latent_height, latent_width),
            device=cond_device,
            dtype=cond_dtype,
        )
        window_mask_frames = torch.ones(
            (1, 1, latent_length * 4, latent_height, latent_width),
            device=cond_device,
            dtype=cond_dtype,
        )
        if has_motion_memory:
            window_mask_frames[:, :, : motion_count * 4] = 0.0

        if character_mask is not None:
            if character_mask.shape[0] > effective_frame_offset or character_mask.shape[0] == 1:
                if character_mask.shape[0] == 1:
                    character_mask = character_mask.repeat((length,) + (1,) * (character_mask.ndim - 1))
                else:
                    character_mask = character_mask[effective_frame_offset:]
                if character_mask.ndim == 3:
                    character_mask = character_mask.unsqueeze(1).movedim(0, 1)
                if character_mask.ndim == 4:
                    character_mask = character_mask.unsqueeze(1)
                character_mask = comfy.utils.common_upscale(
                    character_mask[:, :, :length],
                    latent_width,
                    latent_height,
                    "nearest-exact",
                    "center",
                )
                protected_frames = motion_count * 4 if has_motion_memory else 0
                if character_mask.shape[2] > protected_frames:
                    window_mask_frames[:, :, protected_frames:character_mask.shape[2]] = character_mask[
                        :, :, protected_frames:
                    ]

        window_mask = window_mask_frames.view(
            1,
            window_mask_frames.shape[2] // 4,
            4,
            latent_height,
            latent_width,
        ).transpose(1, 2)

        concat_latent_image = torch.cat((anchor_stack, window_latents), dim=2)
        concat_mask = torch.cat((anchor_mask, window_mask), dim=2)

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        if pose_video is not None:
            if pose_video.shape[0] <= effective_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[effective_frame_offset:]

        if pose_video is not None:
            pose_video = _upscale_images(pose_video, width, height, length)
            if pose_video.shape[0] < length:
                pose_video = torch.cat((pose_video,) + (pose_video[-1:],) * (length - pose_video.shape[0]), dim=0)
            pose_video_latent = vae.encode(pose_video[:, :, :, :3]) * pose_strength
            extra_pose_latents = max(0, anchor_count - 1)
            if extra_pose_latents > 0:
                pose_pad = torch.zeros(
                    (
                        pose_video_latent.shape[0],
                        pose_video_latent.shape[1],
                        extra_pose_latents,
                        pose_video_latent.shape[3],
                        pose_video_latent.shape[4],
                    ),
                    device=pose_video_latent.device,
                    dtype=pose_video_latent.dtype,
                )
                pose_video_latent = torch.cat((pose_pad, pose_video_latent), dim=2)
            positive = node_helpers.conditioning_set_values(positive, {"pose_video_latent": pose_video_latent})
            negative = node_helpers.conditioning_set_values(negative, {"pose_video_latent": pose_video_latent})

        if face_video is not None:
            if face_video.shape[0] <= effective_frame_offset:
                face_video = None
            else:
                face_video = face_video[effective_frame_offset:]

        if face_video is not None:
            face_video = comfy.utils.common_upscale(
                face_video[:length].movedim(-1, 1),
                512,
                512,
                "area",
                "center",
            ) * 2.0 - 1.0
            if face_strength != 1.0:
                face_video = -1.0 + (face_video + 1.0) * face_strength
                face_video = face_video.clamp(-1.0, 1.0)
            face_video = face_video.movedim(0, 1).unsqueeze(0)
            extra_face_frames = max(0, (anchor_count - 1) * 4)
            if extra_face_frames > 0:
                face_pad = torch.full(
                    (
                        face_video.shape[0],
                        face_video.shape[1],
                        extra_face_frames,
                        face_video.shape[3],
                        face_video.shape[4],
                    ),
                    -1.0,
                    device=face_video.device,
                    dtype=face_video.dtype,
                )
                face_video = torch.cat((face_pad, face_video), dim=2)
            positive = node_helpers.conditioning_set_values(positive, {"face_video_pixels": face_video})
            negative = node_helpers.conditioning_set_values(negative, {"face_video_pixels": face_video * 0.0 - 1.0})

        positive = node_helpers.conditioning_set_values(
            positive,
            {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask},
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask},
        )

        out_latent = {
            "samples": torch.zeros(
                [batch_size, 16, latent_length + anchor_count, latent_height, latent_width],
                device=comfy.model_management.intermediate_device(),
            )
        }
        return io.NodeOutput(
            positive,
            negative,
            out_latent,
            anchor_count,
            trim_image,
            effective_frame_offset + length,
        )


class ComfyEverAnimateTrimImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyEverAnimateTrimImages",
            display_name="Comfy EverAnimate Trim Images",
            category="image/video",
            description="Trim duplicated handoff frames after decoding Comfy EverAnimate chunks.",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("trim_amount", default=0, min=0, max=99999),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, images, trim_amount) -> io.NodeOutput:
        trim_amount = max(0, int(trim_amount))
        if trim_amount <= 0:
            return io.NodeOutput(images)
        if trim_amount >= images.shape[0]:
            return io.NodeOutput(images[-1:].clone())
        return io.NodeOutput(images[trim_amount:])


NODE_CLASS_MAPPINGS = {
    "ComfyEverAnimate": ComfyEverAnimate,
    "ComfyEverAnimateTrimImages": ComfyEverAnimateTrimImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyEverAnimate": "Comfy EverAnimate",
    "ComfyEverAnimateTrimImages": "Comfy EverAnimate Trim Images",
}
