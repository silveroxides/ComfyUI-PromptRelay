import logging
import re
import math
import torch
import comfy.utils

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)

from .patches import detect_model_type, apply_patches
from .advanced_options import PromptRelayAdvancedOptions, RelayOptions

log = logging.getLogger(__name__)


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """Convert pixel-space segment lengths to integer latent-space lengths using the
    largest-remainder method. Targets the full `latent_frames` when the pixel sum looks
    like full coverage (within one stride of latent_frames * stride). Otherwise targets
    round(total_pixel / temporal_stride) so partial-coverage timelines stay partial.
    """
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    # Within one frame of full → user clearly intended full coverage; pin to latent_frames.
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    # Ensure every segment has ≥ 1 latent frame (steal from the largest if needed).
    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options=None):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"PromptRelay: '{name}' arrived as None. "
                "Likely causes: a stale workflow JSON saved with null, the timeline "
                "editor's web extension failing to load, or an upstream node returning None. "
                "Set the field to an empty string or fix the upstream connection."
            )

    locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
    if not locals_list:
        raise ValueError("At least one local prompt is required (separate with |)")

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[PromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[PromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[PromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, relay_options)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning


class PromptRelayEncode(io.ComfyNode):
    """Encodes temporal local prompts and patches the model for Prompt Relay."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayEncode",
            display_name="Prompt Relay Encode",
            category="conditioning/prompt_relay",
            description=(
                "Encodes a global prompt combined with temporal local prompts and patches the model "
                "for Prompt Relay temporal control. Local prompts are separated by |. "
                "Use a standard CLIPTextEncode for the negative prompt."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Ordered prompts for each temporal segment, separated by |",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Comma-separated pixel space frame counts per segment. Leave empty to auto-distribute evenly.",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
                RelayOptions.Input(
                    "relay_options", optional=True,
                    tooltip="Optional advanced per-stream tuning. Connect a Prompt Relay Advanced Options node.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options=None) -> io.NodeOutput:
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options,
        )
        return io.NodeOutput(patched, conditioning)


class PromptRelayEncodeTimeline(io.ComfyNode):
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayEncodeTimeline",
            display_name="Prompt Relay Encode (Timeline)",
            category="conditioning/prompt_relay",
            description=(
                "Same as Prompt Relay Encode, but local prompts and segment lengths are edited "
                "visually as draggable blocks on a timeline. The max_frames input only sets the "
                "timeline scale (pixel space) — actual frame count is still read from the latent."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.Int.Input(
                    "max_frames", default=129, min=1, max=10000, step=1,
                    tooltip="Total timeline length in pixel-space frames. Used by the editor for visual scale only.",
                ),
                io.String.Input(
                    "timeline_data", default="",
                    tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand).",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Auto-populated from the timeline editor.",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Auto-populated from the timeline editor (pixel-space frame counts).",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
                io.Float.Input(
                    "fps", default=24.0, min=0.1, max=240.0, step=0.1, optional=True,
                    tooltip="Frames per second — only affects how time is displayed in the timeline editor when time_units is set to 'seconds'.",
                ),
                io.Combo.Input(
                    "time_units", options=["frames", "seconds"], default="frames", optional=True,
                    tooltip="Display the ruler, segment ranges, length input, and total in frames or seconds. Internal storage is always pixel-space frames.",
                ),
                RelayOptions.Input(
                    "relay_options", optional=True,
                    tooltip="Optional advanced per-stream tuning. Connect a Prompt Relay Advanced Options node.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )


    @classmethod
    def execute(cls, model, clip, latent, global_prompt, max_frames, timeline_data, local_prompts, segment_lengths, epsilon, fps=24.0, time_units="frames", relay_options=None) -> io.NodeOutput:
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options,
        )
        return io.NodeOutput(patched, conditioning)


def _encode_relay_advanced(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, vlm_resolution, image_inputs, relay_options=None):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(f"PromptRelay: '{name}' arrived as None.")

    # 1. Parse autogrow images (0-indexed or 1-indexed)
    raw_images = {}
    if image_inputs is not None:
        for k, v in image_inputs.items():
            if v is not None:
                digits = re.findall(r'\d+', k)
                idx = int(digits[0]) if digits else 1
                raw_images[idx] = v

    is_zero_indexed = 0 in raw_images

    # 2. Sequential left-to-right keyword replacement in global and local prompts
    pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)
    sub_token = "<img><image_soft_token><end_of_image>"
    images_vl_raw = []

    def replace_func(match):
        num = int(match.group(1))
        dict_key = num - 1 if is_zero_indexed else num
        if dict_key in raw_images:
            images_vl_raw.append(raw_images[dict_key])
            return sub_token
        return ""

    modified_global = pattern.sub(replace_func, global_prompt)

    locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
    if not locals_list:
        raise ValueError("At least one local prompt is required (separate with |)")

    modified_locals_list = []
    for lp in locals_list:
        mod_lp = pattern.sub(replace_func, lp)
        modified_locals_list.append(mod_lp)

    # 3. Handle latent geometry and segment distribution
    arch, patch_size, temporal_stride = detect_model_type(model)
    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    # 4. Token mapping
    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, modified_global, modified_locals_list)

    # 5. Tokenize text without passing images, getting raw 262144 token IDs
    tokens = clip.tokenize(full_prompt, skip_template=True)

    # 6. Preprocess and manually inject images sequentially into the 262144 tokens
    if len(images_vl_raw) > 0:
        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Original": 896
            }
            samples = image.movedim(-1, 1)
            vlm_size = VLM_RESOLUTIONS.get(res, 896)
            total_vlm = vlm_size * vlm_size
            scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
            width_vlm = round(samples.shape[3] * scale_by_vlm)
            height_vlm = round(samples.shape[2] * scale_by_vlm)

            interp = "area" if res == "Original" else "bicubic"
            s_vlm = comfy.utils.common_upscale(samples, width_vlm, height_vlm, interp, "disabled")
            return s_vlm.movedim(1, -1)[:, :, :, :3]

        processed_images = [process_vlm_image(img, vlm_resolution) for img in images_vl_raw]

        for key, val in tokens.items():
            if isinstance(val, list):
                embed_count = 0
                for r in val:
                    if isinstance(r, list):
                        for i, token in enumerate(r):
                            if isinstance(token, tuple) and len(token) > 0:
                                if token[0] == 262144 and embed_count < len(processed_images):
                                    r[i] = ({"type": "image", "data": processed_images[embed_count]},) + token[1:]
                                    embed_count += 1

    # 7. Encode and patch
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    effective_lengths = distribute_segment_lengths(len(modified_locals_list), latent_frames, parsed_lengths)

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, relay_options)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning


class PromptRelayEncodeAdvanced(io.ComfyNode):
    """Encodes temporal local prompts containing dynamic image inputs and patches the model (Advanced)."""

    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="PromptRelayEncodeAdvanced",
            display_name="Prompt Relay Encode (Advanced)",
            category="conditioning/prompt_relay",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Supports image_input_X keywords.",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Ordered prompts for each temporal segment, separated by |. Supports image_input_X keywords.",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Comma-separated pixel space frame counts per segment. Leave empty to auto-distribute evenly.",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                ),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Original"],
                    default="Fast (384)",
                ),
                RelayOptions.Input(
                    "relay_options", optional=True,
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, vlm_resolution, image_inputs, relay_options=None) -> io.NodeOutput:
        patched, conditioning = _encode_relay_advanced(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, vlm_resolution, image_inputs, relay_options,
        )
        return io.NodeOutput(patched, conditioning)


NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "PromptRelayEncodeAdvanced": PromptRelayEncodeAdvanced,
    "PromptRelayAdvancedOptions": PromptRelayAdvancedOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "PromptRelayEncodeAdvanced": "Prompt Relay Encode (Advanced)",
    "PromptRelayAdvancedOptions": "Prompt Relay Advanced Options",
}
