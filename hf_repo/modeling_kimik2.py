"""
Minimal HuggingFace wrapper for KimiK2 model recognition
"""
import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class KimiK2Config(PretrainedConfig):
    model_type = "kimi-k2"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class KimiK2ForCausalLM(PreTrainedModel):
    config_class = KimiK2Config
    
    def __init__(self, config):
        super().__init__(config)
        # This is just for HF recognition - actual loading happens via direct PyTorch
        print("Note: Use the direct PyTorch loading method shown in the README for this model.")
    
    def forward(self, input_ids, **kwargs):
        # Placeholder for HF compatibility
        batch_size, seq_len = input_ids.shape
        vocab_size = getattr(self.config, 'vocab_size', 50304)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return CausalLMOutputWithPast(logits=logits)
