"""Optional on-device EmbeddingGemma 2 text runtime backed by Apple MLX."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


MODEL_REPOSITORY = "mlx-community/embeddinggemma-2-8bit"
# Pin both the model conversion and its upstream model revision. This keeps
# local verse caches stable and avoids silently changing retrieval behavior.
MODEL_REVISION = "7505ef2f8ddef45efef6d060865f27989b3c9cec"
RUNTIME_ID = f"mlx-vlm-3d87e884-{MODEL_REVISION}-8bit"


def ensure_model(model_dir: Path | None = None) -> Path:
    """Get the pinned text encoder snapshot, reusing Hugging Face's local cache."""
    if model_dir is not None:
        if model_dir.is_dir():
            return model_dir
        raise FileNotFoundError(f"EmbeddingGemma model directory does not exist: {model_dir}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config_sentence_transformers.json",
            "sentence_bert_config.json",
            "modules.json",
            "1_Pooling/config.json",
            "2_Normalize/config.json",
        ],
    ))


class EmbeddingGemma2:
    """Text-only EmbeddingGemma 2 encoder for Apple silicon.

    The model's optional audio and vision encoders are disabled: Devar embeds
    transcript queries and Bible verse text only. MLX keeps inference local.
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        try:
            import mlx.core as mx
            from mlx_vlm.embedding_loader import load_embedding_model
            from mlx_vlm.utils import load_config
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional 'embedding' extra to use on-device EmbeddingGemma"
            ) from exc

        path = ensure_model(model_dir)
        config = load_config(str(path))
        config["audio_config"] = None
        config["vision_config"] = None
        self._mx = mx
        self._model = load_embedding_model(path, config=config)
        self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.identity = f"{MODEL_REPOSITORY}@{MODEL_REVISION}-{RUNTIME_ID}-768"
        mx.eval(self._model.parameters())
        mx.set_cache_limit(512 * 1024 * 1024)
        # Pay one-time MLX kernel compilation during app preparation.
        self.embed_texts(["task: search result | query: warm up"])

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        import numpy as np

        rows = []
        for start in range(0, len(texts), 16):
            batch = list(texts[start:start + 16])
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            inputs = {name: self._mx.array(values) for name, values in tokens.items()}
            output = self._model(**inputs)
            rows.append(np.asarray(output.text_embeds.astype(self._mx.float32)))
        return np.concatenate(rows, axis=0).tolist()

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._mx.clear_cache()
