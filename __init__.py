from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes import PromptRelayEncode, PromptRelayEncodeTimeline, PromptRelayAdvancedOptions, PromptRelayEncodeAdvanced
from .smart_nodes import PromptRelaySmartEncode, PromptRelaySmartEncodeTest, PromptRelaySmartEncodeAdvanced
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PromptRelayEncode,
            PromptRelayEncodeTimeline,
            PromptRelayEncodeAdvanced,
            PromptRelaySmartEncode,
            PromptRelaySmartEncodeTest,
            PromptRelaySmartEncodeAdvanced,
            PromptRelayAdvancedOptions
        ]


async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()

NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "PromptRelayEncodeAdvanced": PromptRelayEncodeAdvanced,
    "PromptRelaySmartEncode": PromptRelaySmartEncode,
    "PromptRelaySmartEncodeAdvanced": PromptRelaySmartEncodeAdvanced,
    "PromptRelaySmartEncodeTest": PromptRelaySmartEncodeTest,
    "PromptRelayAdvancedOptions": PromptRelayAdvancedOptions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "PromptRelayEncodeAdvanced": "Prompt Relay Encode (Advanced)",
    "PromptRelaySmartEncode": "Prompt Relay Encode (Smart)",
    "PromptRelaySmartEncodeAdvanced": "Prompt Relay Encode (Smart/Advanced)",
    "PromptRelaySmartEncodeTest": "Prompt Relay Smart Encode Test",
    "PromptRelayAdvancedOptions": "Prompt Relay Advanced Options"
}


WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
