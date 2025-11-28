"""GPU client wrapper for GPT-OSS 20B model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


@dataclass
class GPTOSSClientConfig:
    """Configuration for instantiating the GPU model."""

    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    system_prompt: str = (
        "You are a meticulous Python execution reasoning assistant. "
        "Given a program plus assertion you must reason step-by-step, then respond "
        "exactly using the requested format (e.g., [ANSWER] tags or JSON)."
    )


@dataclass
class GPTOSSInvocationParams:
    """Per-request inference settings."""

    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    do_sample: bool = True
    seed: Optional[int] = None


@dataclass
class GPTOSSResponse:
    text: str
    latency_s: float
    raw_output: Dict[str, Any]


class GPTOSSClient:
    """Thin wrapper over HuggingFace transformers for GPT-OSS."""

    def __init__(self, config: GPTOSSClientConfig | None = None) -> None:
        self.config = config or GPTOSSClientConfig()
        
        print(f"Loading {self.config.model_id}...")
        start = time.time()
        
        # Set torch dtype
        if self.config.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        
        elapsed = time.time() - start
        print(f"✓ Model loaded in {elapsed:.1f}s")
        
        # Check GPU memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            print(f"✓ GPU memory used: {memory_allocated:.2f} GB")

    def invoke(self, prompt: str, params: GPTOSSInvocationParams | None = None) -> GPTOSSResponse:
        params = params or GPTOSSInvocationParams()
        
        # Prepare input
        full_prompt = f"{self.config.system_prompt}\n\n{prompt}"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        # Set seed for reproducibility
        if params.seed is not None:
            torch.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(params.seed)
        
        # Generate
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                do_sample=params.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.time() - start
        
        # Decode output
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove input prompt)
        generated_text = full_output[len(full_prompt):].strip()
        
        return GPTOSSResponse(
            text=generated_text,
            latency_s=latency,
            raw_output={"full_output": full_output}
        )


__all__ = [
    "DEFAULT_MODEL_ID",
    "GPTOSSClientConfig",
    "GPTOSSInvocationParams",
    "GPTOSSResponse",
    "GPTOSSClient",
]