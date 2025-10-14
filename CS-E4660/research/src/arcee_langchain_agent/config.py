"""
Configuration for Arcee Agent with LangChain
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple


class ArceeAgentConfig:
    """Configuration class for Arcee Agent"""
    
    MODEL_NAME = "arcee-ai/Arcee-Agent"
    MAX_NEW_TOKENS = 2048
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    @staticmethod
    def load_model(device: str = "cuda") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load Arcee Agent model and tokenizer
        
        Args:
            device: Device to load model on ('cuda' or 'cpu')
            
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading Arcee Agent model on {device}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            ArceeAgentConfig.MODEL_NAME,
            trust_remote_code=True
        )
        
        # Check if CUDA is available
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            ArceeAgentConfig.MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        print(f"Model loaded successfully on {device}")
        return model, tokenizer
    
    @staticmethod
    def load_model_quantized(device: str = "cuda"):
        """
        Load Arcee Agent with 4-bit quantization for memory efficiency
        Requires bitsandbytes library
        
        Args:
            device: Device to load model on
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            print("Warning: bitsandbytes not installed. Loading without quantization.")
            return ArceeAgentConfig.load_model(device)
        
        print("Loading Arcee Agent with 4-bit quantization...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            ArceeAgentConfig.MODEL_NAME,
            trust_remote_code=True
        )
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            ArceeAgentConfig.MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Quantized model loaded successfully")
        return model, tokenizer
