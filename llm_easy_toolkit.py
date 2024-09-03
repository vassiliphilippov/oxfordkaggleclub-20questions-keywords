from typing import Any, TypeVar, Optional
from datetime import datetime, timezone

import jinja2
from pydantic import BaseModel, ValidationError
import transformers
import torch

PROMPT_PATH = "prompts"


def load(template_path: str, **kwargs: Any) -> str:
    """Load a template and render it with the given keyword arguments."""
    with open(template_path, "r", encoding="utf-8") as f:
        template = jinja2.Template(f.read())

    return template.render(**kwargs)


class LLamaLLM:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name: str = model_name
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def __call__(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
        )

        return outputs[0]["generated_text"][-1]["content"]


def run(llm, template_path: str, **kwargs: Any) -> str:
    """Run the model with the given template and keyword arguments."""
    prompt: str = load(template_path, **kwargs)
    return llm(prompt)

