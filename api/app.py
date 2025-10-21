import typing

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import transformers

from .config import Config
from .schemas import PostRequest, ReplyRequest

CFG: Config = Config()

pipelines: typing.Dict[str, transformers.pipeline] = {}
for task, model_id in CFG.pipeline_ids.items():
    pipelines[task] = transformers.pipeline("text-generation", model_id)
    pipelines[task].model.load_adapter(model_id)


def generate(pipeline: transformers.pipeline, instructions: typing.List[typing.Dict]):
    return (
        pipeline(
            (
                pipeline
                .tokenizer
                .apply_chat_template(instructions["messages"], tokenize=False)
            ), 
            max_new_tokens=254,
            return_full_text=False
        )
        [0]
        ["generated_text"]
        .replace("assistant\n\n", "")
    )


app = FastAPI(title=CFG.title, version=CFG.version, docs_url="/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG.trust_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/post")
def post(request: PostRequest) -> str:
    return generate(pipelines["post"], request.get_instruction())


@app.post("/reply")
def reply(request: ReplyRequest) -> str:
    return generate(pipelines["reply"], request.get_instruction())


if __name__ == "__main__":
    uvicorn.run(app)