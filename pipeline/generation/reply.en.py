import random
import uuid
import json

import transformers 

import rich.progress

import twon_agents


MODEL = "simon-muenker/TWON-Agent-OSN-Replies-en"
DATA_PATH = "data/processed/twitter.english.dataset.enriched.csv"

dataset = twon_agents.data.format_reply_instructions_dataset(DATA_PATH)

print(len(dataset))

pipeline = transformers.pipeline("text-generation", MODEL, device="cuda:0")
pipeline.model.load_adapter(MODEL)

for sample in rich.progress.track(random.sample(dataset, k=1000)):
    reply: str = (
        pipeline(
            (
                pipeline
                .tokenizer
                .apply_chat_template(sample["messages"][:-1], tokenize=False)
            ), 
            max_new_tokens=64,
            return_full_text=False,
            do_sample=False
        )
        [0]
        ["generated_text"]
    )

    open(f"pipeline/generation/data/{uuid.uuid4()}.json", "w").write(
        json.dumps({
            "prompt": sample["messages"][:-1],
            "user": sample["messages"][-1]["content"],
            "synthetic": reply.replace("assistant\n\n", ""),
        }, indent=4, ensure_ascii=False)
    )
    