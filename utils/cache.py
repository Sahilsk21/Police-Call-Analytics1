import torch
from functools import lru_cache

@lru_cache(maxsize=4)
def load_model(key, loader_func):
    """Cache models in memory"""
    print(f"Loading model: {key}")
    torch.cuda.empty_cache()
    return loader_func()
