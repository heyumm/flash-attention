"""Flash Attention CUTE (CUDA Template Engine) implementation."""

__version__ = "0.1.0"
__heyu_commit__ = "afef8da+"
__heyu_tag__ = "heyu/qat: QAT_DENSE_P fakequant (no +7 trick)"

import os as _os
if int(_os.environ.get("RANK", "0")) == 0:
    print(
        f"[flash_attn_cute] {__heyu_tag__} @ {__heyu_commit__} | {__file__}",
        flush=True,
    )

import cutlass.cute as cute

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

from flash_attn_cute.cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
