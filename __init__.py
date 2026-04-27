from typing import override
from comfy_api.latest import ComfyExtension, io

from .pbn_node import PaintByNumberNode

class PBNExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PaintByNumberNode]

async def comfy_entrypoint() -> PBNExtension:
    return PBNExtension()
