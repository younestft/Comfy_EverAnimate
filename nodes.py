import logging
import math

import comfy.samplers
import comfy.model_management
import comfy.utils
import node_helpers
import nodes
import torch
from comfy_api.latest import io

log = logging.getLogger(__name__)


EverAnimateInitialSettings = io.Custom("EVERANIMATE_INITIAL_SETTINGS")
EverAnimateSettings = io.Custom("EVERANIMATE_SETTINGS")


DEFAULT_BATCH_SIZE = 1
DEFAULT_STARTUP_CARRY_FRAMES = 1
DEFAULT_NUM_MOTION_LATENTS = 1
DEFAULT_CONTINUE_MOTION_MAX_FRAMES = 5
DEFAULT_MOTION_HANDOFF_STRENGTH = 0.75


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


def _build_everanimate_chunk(
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
    motion_handoff_strength,
    continue_motion_max_frames,
    reference_image=None,
    clip_vision_output=None,
    face_video=None,
    pose_video=None,
    background_video=None,
    character_mask=None,
    continue_motion=None,
    prev_samples=None,
    video_anchor_latent=None,
    allow_latent_memory_with_image_carry=False,
):
    latent_length = ((length - 1) // 4) + 1
    latent_width = width // 8
    latent_height = height // 8
    anchor_count = int(num_video_anchor_latents)
    motion_count = min(int(num_motion_latents), latent_length)
    pose_strength = float(pose_strength)
    face_strength = float(face_strength)
    motion_handoff_strength = max(0.0, min(1.0, float(motion_handoff_strength)))

    continue_motion_frames = 0
    continue_motion_latents = 0
    has_image_carry = continue_motion is not None and continue_motion.shape[0] > 0
    if has_image_carry:
        continue_motion_frames = min(int(continue_motion_max_frames), continue_motion.shape[0], length)
        continue_motion_latents = ((continue_motion_frames - 1) // 4) + 1

    prev_latents = None
    if not has_image_carry or allow_latent_memory_with_image_carry:
        prev_latents = _latent_samples(prev_samples, "prev_samples")
    has_motion_memory = prev_latents is not None and motion_count > 0
    if has_image_carry:
        trim_image = max(0, continue_motion_latents * 4 - 3)
        effective_frame_offset = max(0, int(video_frame_offset) - continue_motion_frames)
    else:
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

    if has_image_carry:
        carry_images = continue_motion[-continue_motion_frames:]
        carry_images = _upscale_images(carry_images, width, height, continue_motion_frames).to(device=cond_device)
        image[: carry_images.shape[0]] = carry_images

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
    if has_image_carry:
        window_mask_frames[:, :, : continue_motion_latents * 4] = 1.0 - motion_handoff_strength
    elif has_motion_memory:
        window_mask_frames[:, :, : motion_count * 4] = 1.0 - motion_handoff_strength

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
            if has_image_carry:
                protected_frames = continue_motion_latents * 4
            elif has_motion_memory:
                protected_frames = motion_count * 4
            else:
                protected_frames = 0
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
    return positive, negative, out_latent, anchor_count, trim_image, effective_frame_offset + length


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
                io.Float.Input(
                    "motion_handoff_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the previous chunk's motion latents are locked into the next chunk. "
                        "1.0 is the original hard handoff; 0.6-0.8 is usually smoother for low-step distill runs."
                    ),
                ),
                io.Int.Input(
                    "continue_motion_max_frames",
                    default=5,
                    min=1,
                    max=nodes.MAX_RESOLUTION,
                    step=4,
                    tooltip=(
                        "When continue_motion images are connected, carry this many final RGB frames into the next chunk. "
                        "This matches native WanAnimate's image carry-over behavior."
                    ),
                ),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("reference_image", optional=True),
                io.Image.Input("face_video", optional=True),
                io.Image.Input("pose_video", optional=True),
                io.Image.Input("background_video", optional=True),
                io.Mask.Input("character_mask", optional=True),
                io.Image.Input(
                    "continue_motion",
                    optional=True,
                    tooltip=(
                        "Native-style image carry-over. Connect the previous chunk's final decoded images here. "
                        "When connected, this takes priority over prev_samples."
                    ),
                ),
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
        motion_handoff_strength,
        continue_motion_max_frames,
        reference_image=None,
        clip_vision_output=None,
        face_video=None,
        pose_video=None,
        background_video=None,
        character_mask=None,
        continue_motion=None,
        prev_samples=None,
        video_anchor_latent=None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *_build_everanimate_chunk(
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
                motion_handoff_strength,
                continue_motion_max_frames,
                reference_image=reference_image,
                clip_vision_output=clip_vision_output,
                face_video=face_video,
                pose_video=pose_video,
                background_video=background_video,
                character_mask=character_mask,
                continue_motion=continue_motion,
                prev_samples=prev_samples,
                video_anchor_latent=video_anchor_latent,
            )
        )


def _trim_latent(latent, trim_amount):
    trim_amount = max(0, int(trim_amount))
    out = latent.copy()
    out["samples"] = latent["samples"][:, :, trim_amount:]
    return out


def _decode_latent_images(vae, latent):
    images = vae.decode(latent["samples"])
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _trim_image_batch(images, trim_amount):
    trim_amount = max(0, int(trim_amount))
    if trim_amount <= 0:
        return images
    if trim_amount >= images.shape[0]:
        return images[-1:].clone()
    return images[trim_amount:]


def _safe_image_batch(images, height, width):
    if images is not None and images.shape[0] > 0:
        return images
    return torch.zeros((1, height, width, 3), dtype=torch.float32)


def _guide_length(value):
    if value is None:
        return None
    if hasattr(value, "shape") and len(value.shape) > 0:
        return int(value.shape[0])
    return None


def _shortest_guide_length(*guides):
    lengths = [length for length in (_guide_length(guide) for guide in guides) if length is not None and length > 0]
    return min(lengths) if lengths else None


def _repeat_reference_for_startup(reference_image, startup_carry_frames):
    if reference_image is None or int(startup_carry_frames) <= 0:
        return None
    return reference_image[:1].repeat((int(startup_carry_frames), 1, 1, 1))


def _sampler_default():
    return "uni_pc" if "uni_pc" in comfy.samplers.KSampler.SAMPLERS else comfy.samplers.KSampler.SAMPLERS[0]


def _scheduler_default():
    return "normal" if "normal" in comfy.samplers.KSampler.SCHEDULERS else comfy.samplers.KSampler.SCHEDULERS[0]


def _chunk_trim_amount_from_frames(frame_count):
    if frame_count <= 0:
        return 0
    latent_count = ((max(1, int(frame_count)) - 1) // 4) + 1
    return max(0, latent_count * 4 - 3)


def _settings_output_target(settings):
    return None


def _settings_expected_chunk_count(settings):
    return int(settings.get("chunk_index", 0)) + 1


def _extension_chunks_required(settings, startup_carry_frames, continue_motion_max_frames=DEFAULT_CONTINUE_MOTION_MAX_FRAMES):
    frame_count = int(settings.get("frame_count", 0))
    if frame_count <= 0:
        frame_count = _shortest_guide_length(
            settings.get("pose_video"),
            settings.get("face_video"),
            settings.get("background_video"),
            settings.get("character_mask"),
        ) or 0
    if frame_count <= 0:
        return 0

    if settings.get("reference_image") is not None and int(startup_carry_frames) > 0:
        startup_trim_image = _chunk_trim_amount_from_frames(startup_carry_frames)
    else:
        startup_trim_image = 0
    continue_trim_image = _chunk_trim_amount_from_frames(continue_motion_max_frames)
    first_kept = max(1, int(settings["length"]) - int(startup_trim_image))
    later_kept = max(1, int(settings["length"]) - int(continue_trim_image))
    if int(frame_count) <= first_kept:
        return 0
    return int(math.ceil((int(frame_count) - first_kept) / later_kept))


def _copy_settings(settings):
    copied = dict(settings)
    return copied


def _make_master_settings(
    model,
    positive,
    negative,
    vae,
    width,
    height,
    chunk_length,
    ref_image_background,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    denoise,
    num_video_anchor_latents,
    pose_strength,
    face_strength,
    clip_vision_output=None,
    reference_image=None,
    face_video=None,
    pose_video=None,
    background_video=None,
    character_mask=None,
    video_anchor_latent=None,
    frame_count=0,
):
    settings = {
        "model": model,
        "positive": positive,
        "negative": negative,
        "vae": vae,
        "width": int(width),
        "height": int(height),
        "length": int(chunk_length),
        "frame_count": int(frame_count),
        "ref_image_background": bool(ref_image_background),
        "batch_size": DEFAULT_BATCH_SIZE,
        "seed": int(seed),
        "steps": int(steps),
        "cfg": float(cfg),
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "denoise": float(denoise),
        "num_video_anchor_latents": int(num_video_anchor_latents),
        "pose_strength": float(pose_strength),
        "face_strength": float(face_strength),
        "clip_vision_output": clip_vision_output,
        "reference_image": reference_image,
        "face_video": face_video,
        "pose_video": pose_video,
        "background_video": None if ref_image_background else background_video,
        "character_mask": None if ref_image_background else character_mask,
        "video_anchor_latent": video_anchor_latent,
        "chunk_index": 0,
        "video_frame_offset": 0,
        "previous_samples": None,
    }
    return settings


def _run_settings_chunk(
    settings,
    input_images=None,
    is_initial=False,
    startup_carry_frames=1,
    num_motion_latents=1,
    continue_motion_max_frames=5,
    motion_handoff_strength=0.75,
):
    settings = _copy_settings(settings)

    if is_initial:
        continue_motion = _repeat_reference_for_startup(settings.get("reference_image"), startup_carry_frames)
        carry_frame_count = max(1, int(startup_carry_frames)) if continue_motion is not None else 1
        motion_latent_count = 0
        handoff_strength = 1.0
        previous_samples = None
    else:
        if input_images is None or input_images.shape[0] == 0:
            raise ValueError("EverAnimate Extension Chunk needs the previous chunk images connected to its images input.")
        continue_motion = input_images
        carry_frame_count = int(continue_motion_max_frames)
        motion_latent_count = int(num_motion_latents)
        handoff_strength = float(motion_handoff_strength)
        previous_samples = settings.get("previous_samples")

    (
        chunk_positive,
        chunk_negative,
        latent,
        trim_latent,
        trim_image,
        _next_video_frame_offset,
    ) = _build_everanimate_chunk(
        settings["positive"],
        settings["negative"],
        settings["vae"],
        settings["width"],
        settings["height"],
        settings["length"],
        settings["batch_size"],
        settings["num_video_anchor_latents"],
        motion_latent_count,
        settings.get("video_frame_offset", 0),
        settings["pose_strength"],
        settings["face_strength"],
        handoff_strength,
        carry_frame_count,
        reference_image=settings.get("reference_image"),
        clip_vision_output=settings.get("clip_vision_output"),
        face_video=settings.get("face_video"),
        pose_video=settings.get("pose_video"),
        background_video=settings.get("background_video"),
        character_mask=settings.get("character_mask"),
        continue_motion=continue_motion,
        prev_samples=previous_samples,
        video_anchor_latent=settings.get("video_anchor_latent"),
        allow_latent_memory_with_image_carry=not is_initial,
    )

    sampled = nodes.common_ksampler(
        settings["model"],
        int(settings["seed"]),
        settings["steps"],
        settings["cfg"],
        settings["sampler_name"],
        settings["scheduler"],
        chunk_positive,
        chunk_negative,
        latent,
        denoise=settings["denoise"],
    )[0]
    decoded = _decode_latent_images(settings["vae"], _trim_latent(sampled, trim_latent))
    kept_images = _trim_image_batch(decoded, trim_image)
    kept_images = _safe_image_batch(kept_images, settings["height"], settings["width"])
    output = kept_images if is_initial else torch.cat((input_images, kept_images), dim=0)

    settings["previous_samples"] = sampled
    settings["chunk_index"] = int(settings.get("chunk_index", 0)) + 1
    settings["video_frame_offset"] = int(settings.get("video_frame_offset", 0)) + int(kept_images.shape[0])
    settings["last_frame_count"] = int(output.shape[0])

    return output, settings


class ComfyEverAnimateMasterSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyEverAnimateMasterSettings",
            display_name="EverAnimate Master Settings",
            category="model/conditioning/video_models",
            description="Collects all EverAnimate native chunk settings into one initial settings socket.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("chunk_length", display_name="chunk length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Boolean.Input(
                    "ref_image_background",
                    display_name="ref image background",
                    default=True,
                    tooltip="When enabled, background_video and character_mask inputs are ignored and the reference image/background carry is used instead.",
                ),
                io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, control_after_generate=io.ControlAfterGenerate.fixed),
                io.Int.Input("steps", default=4, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="lcm" if "lcm" in comfy.samplers.KSampler.SAMPLERS else _sampler_default()),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default=_scheduler_default()),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("num_video_anchor_latents", default=4, min=1, max=16, step=1),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0, step=0.001),
                io.Float.Input("face_strength", default=1.0, min=0.0, max=10.0, step=0.001),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("reference_image", optional=True),
                io.Image.Input("face_video", optional=True),
                io.Image.Input("pose_video", optional=True),
                io.Image.Input("background_video", optional=True),
                io.Mask.Input("character_mask", optional=True),
                io.Latent.Input("video_anchor_latent", optional=True),
                io.Int.Input(
                    "frame_count",
                    display_name="frame count",
                    default=0,
                    min=0,
                    max=999999,
                    step=1,
                    optional=True,
                    force_input=True,
                    tooltip="Optional socket-only planning frame count. Connect an INT value to calculate how many Extension Chunk nodes are needed without running the workflow.",
                ),
            ],
            outputs=[
                EverAnimateInitialSettings.Output(display_name="initial settings"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, *args, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(_make_master_settings(*args, **kwargs))


class ComfyEverAnimateInitialChunk(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyEverAnimateInitialChunk",
            display_name="EverAnimate Initial Chunk",
            category="model/conditioning/video_models",
            description="Generates the first EverAnimate chunk from master settings.",
            inputs=[
                EverAnimateInitialSettings.Input("initial_settings", display_name="initial settings"),
                io.Int.Input(
                    "startup_carry_frames",
                    default=DEFAULT_STARTUP_CARRY_FRAMES,
                    min=0,
                    max=nodes.MAX_RESOLUTION,
                    step=1,
                    tooltip="Carries the reference image into the first chunk. 1 reduces startup flashes without adding extra duplicate frames.",
                ),
            ],
            outputs=[
                EverAnimateSettings.Output(display_name="settings"),
                io.Image.Output(display_name="images"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, initial_settings, startup_carry_frames) -> io.NodeOutput:
        output, next_settings = _run_settings_chunk(
            initial_settings,
            input_images=None,
            is_initial=True,
            startup_carry_frames=startup_carry_frames,
        )
        return io.NodeOutput(next_settings, output)


class ComfyEverAnimateContinueChunk(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyEverAnimateContinueChunk",
            display_name="EverAnimate Extension Chunk",
            category="model/conditioning/video_models",
            description="Generates the next EverAnimate extension chunk from cumulative images and settings.",
            inputs=[
                EverAnimateSettings.Input("settings"),
                io.Int.Input(
                    "num_motion_latents",
                    default=DEFAULT_NUM_MOTION_LATENTS,
                    min=0,
                    max=16,
                    step=1,
                    tooltip="EverAnimate M. 1 is the recommended motion-memory default for smoother chunk handoff.",
                ),
                io.Int.Input(
                    "continue_motion_max_frames",
                    default=DEFAULT_CONTINUE_MOTION_MAX_FRAMES,
                    min=1,
                    max=nodes.MAX_RESOLUTION,
                    step=4,
                    tooltip="Carries this many final decoded frames into the next chunk. 5 is usually smoother for WanAnimate.",
                ),
                io.Float.Input(
                    "motion_handoff_strength",
                    default=DEFAULT_MOTION_HANDOFF_STRENGTH,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="0.75 soft-locks carry frames to reduce flashing on low-step distilled runs.",
                ),
                io.Image.Input("images"),
            ],
            outputs=[
                EverAnimateSettings.Output(display_name="settings"),
                io.Image.Output(display_name="images"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        settings,
        num_motion_latents,
        continue_motion_max_frames,
        motion_handoff_strength,
        images,
    ) -> io.NodeOutput:
        output, next_settings = _run_settings_chunk(
            settings,
            input_images=images,
            is_initial=False,
            num_motion_latents=num_motion_latents,
            continue_motion_max_frames=continue_motion_max_frames,
            motion_handoff_strength=motion_handoff_strength,
        )
        return io.NodeOutput(next_settings, output)


def _match_image_stats(image, reference, strength):
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0:
        return image
    src = image.float()
    ref = reference.float()
    src_mean = src.mean(dim=(0, 1), keepdim=True)
    src_std = src.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    ref_mean = ref.mean(dim=(0, 1), keepdim=True)
    ref_std = ref.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    matched = (src - src_mean) / src_std * ref_std + ref_mean
    return torch.lerp(src, matched, strength).clamp(0.0, 1.0).to(image.dtype)


class ComfyEverAnimateColorCorrection(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyEverAnimateColorCorrection",
            display_name="EverAnimate Color Correction",
            category="image/video",
            description="Matches chunk color statistics to reference frames for smoother chunk continuity.",
            inputs=[
                EverAnimateSettings.Input("settings"),
                io.Image.Input("images"),
                io.Image.Input("reference_images", display_name="reference images", optional=True),
                io.Int.Input("source_window", default=12, min=0, max=99999),
                io.Int.Input("reference_window", default=16, min=1, max=99999),
                io.Combo.Input("method", options=["mkl_lab", "mean_std_rgb"], default="mkl_lab"),
                io.Combo.Input("mode", options=["per_frame", "global"], default="per_frame"),
                io.Int.Input("start_frame", default=0, min=0, max=99999),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        settings,
        images,
        source_window,
        reference_window,
        method,
        mode,
        start_frame,
        strength,
        reference_images=None,
    ) -> io.NodeOutput:
        if images is None or reference_images is None or images.shape[0] == 0 or reference_images.shape[0] == 0:
            return io.NodeOutput(images)

        corrected = images.clone()
        start = min(max(0, int(start_frame)), corrected.shape[0])
        count = int(source_window)
        end = corrected.shape[0] if count <= 0 else min(corrected.shape[0], start + count)
        refs = reference_images[-max(1, int(reference_window)) :]

        if mode == "global":
            reference = refs.mean(dim=0)
            for idx in range(start, end):
                corrected[idx] = _match_image_stats(corrected[idx], reference, strength)
        else:
            for out_idx, idx in enumerate(range(start, end)):
                ref_idx = min(out_idx, refs.shape[0] - 1)
                corrected[idx] = _match_image_stats(corrected[idx], refs[ref_idx], strength)

        return io.NodeOutput(corrected)


NODE_CLASS_MAPPINGS = {
    "ComfyEverAnimate": ComfyEverAnimate,
    "ComfyEverAnimateMasterSettings": ComfyEverAnimateMasterSettings,
    "ComfyEverAnimateInitialChunk": ComfyEverAnimateInitialChunk,
    "ComfyEverAnimateContinueChunk": ComfyEverAnimateContinueChunk,
    "ComfyEverAnimateColorCorrection": ComfyEverAnimateColorCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyEverAnimate": "Comfy EverAnimate",
    "ComfyEverAnimateMasterSettings": "EverAnimate Master Settings",
    "ComfyEverAnimateInitialChunk": "EverAnimate Initial Chunk",
    "ComfyEverAnimateContinueChunk": "EverAnimate Extension Chunk",
    "ComfyEverAnimateColorCorrection": "EverAnimate Color Correction",
}
