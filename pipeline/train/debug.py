import os

import twon_agents


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

twon_agents.align_content_generation.Pipeline(
    task="reply",
    dataset=dict(
        train_path="data/processed/twitter.german.dataset.enriched.csv",
        eval_path="data/processed/twitter.german.dataset.enriched.csv",
        max_samples_train=20,
        max_samples_eval=5
    ),
    models=dict(
        base="meta-llama/Llama-3.2-3B-Instruct",
        adapter="simon-muenker/TWON-Agent-OSN-debug",
    ),
    training=dict(logging_steps=5),
    testing=dict(num_repitions=2, num_samples=5),
)()
