from .nodes import (
    ComfyEverAnimate,
    ComfyEverAnimateAIO,
    ComfyEverAnimateContinueChunk,
    ComfyEverAnimateInitialChunk,
    ComfyEverAnimateMasterSettings,
    ComfyEverAnimateTrimImages,
)
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class ComfyEverAnimateExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ComfyEverAnimate,
            ComfyEverAnimateTrimImages,
            ComfyEverAnimateAIO,
            ComfyEverAnimateMasterSettings,
            ComfyEverAnimateInitialChunk,
            ComfyEverAnimateContinueChunk,
        ]


async def comfy_entrypoint() -> ComfyEverAnimateExtension:
    return ComfyEverAnimateExtension()


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "comfy_entrypoint",
]


WEB_DIRECTORY = "./web"
