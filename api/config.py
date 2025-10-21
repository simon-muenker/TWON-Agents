import os
import typing

import torch
import pydantic


os.environ['CUDA_VISIBLE_DEVICES'] = "2"


class Config(pydantic.BaseModel):
    title: str = "TWON Agents API (post, reply)"
    version: str = "0.1.0"

    trust_origins: typing.List[str] = ["*"]

    device: torch.device = torch.device("cuda:0")

    pipeline_ids: typing.Dict[str, str] = {
        "post": "simon-muenker/TWON-Agent-OSN-Post-en",
        "reply": "simon-muenker/TWON-Agent-OSN-Replies-en"
    }
    
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
