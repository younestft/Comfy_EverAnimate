from .nodes import ComfyEverAnimate, ComfyEverAnimateTrimImages
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class ComfyEverAnimateExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ComfyEverAnimate,
            ComfyEverAnimateTrimImages,
        ]


async def comfy_entrypoint() -> ComfyEverAnimateExtension:
    return ComfyEverAnimateExtension()


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "comfy_entrypoint",
]
