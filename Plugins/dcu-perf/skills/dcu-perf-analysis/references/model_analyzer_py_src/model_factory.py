

import json
from typing import Dict, Any, Optional
from pathlib import Path
from enum import Enum


class ModelType(Enum):
    """支持的模型类型"""
    DEEPSEEK_V3 = "deepseek_v3"
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    QWEN = "qwen"
    CUSTOM = "custom"


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """加载模型配置文件"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def create_deepseek_v3_config() -> Dict[str, Any]:
        """创建DeepSeek-V3默认配置"""
        return {
            "model_type": "deepseek_v3",
            "vocab_size": 129280,
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "num_routed_experts": 256,
            "num_shared_experts": 1,
            "num_activated_experts": 8,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "max_position_embeddings": 163840,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "dtype": "bfloat16",
            "use_cache": True,
            "moe_layer_freq": 1,
            "topk_group": 4,
            "scoring_func": "sigmoid"
        }
    
    @staticmethod
    def create_llama2_config() -> Dict[str, Any]:
        """创建LLaMA2默认配置"""
        return {
            "model_type": "llama2",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "num_routed_experts": 0,  # 不是MoE模型
            "num_activated_experts": 0,
            "q_lora_rank": 0,
            "kv_lora_rank": 0,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 128,
            "v_head_dim": 128,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "dtype": "float16",
            "use_cache": True
        }
    
    @classmethod
    def get_model_config(cls, model_type: ModelType, 
                        config_path: Optional[str] = None) -> Dict[str, Any]:
        """获取模型配置"""
        if config_path:
            config = cls.load_config(config_path)
        else:
            if model_type == ModelType.DEEPSEEK_V3:
                config = cls.create_deepseek_v3_config()
            elif model_type == ModelType.LLAMA2:
                config = cls.create_llama2_config()
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
        
        default_config = {
            "model_type": "unknown",
            "vocab_size": 0,
            "hidden_size": 0,
            "intermediate_size": 0,
            "moe_intermediate_size": 0,
            "num_hidden_layers": 0,
            "num_attention_heads": 0,
            "num_key_value_heads": 0,
            "num_routed_experts": 0,
            "num_shared_experts": 0,
            "num_activated_experts": 0,
            "q_lora_rank": 0,
            "kv_lora_rank": 0,
            "qk_nope_head_dim": 0,
            "qk_rope_head_dim": 0,
            "v_head_dim": 0,
            "max_position_embeddings": 0,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "dtype": "float16",
            "use_cache": True,
            "moe_layer_freq": 1,
            "topk_group": 1,
            "scoring_func": "softmax"
        }
        
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    
    @staticmethod
    def get_model_info(config: Dict[str, Any]) -> str:
        """获取模型信息字符串"""
        info = f"模型类型: {config.get('model_type', 'unknown')}\n"
        info += f"参数量: {cls._estimate_params(config):,}\n"
        info += f"隐藏层大小: {config.get('hidden_size', 0)}\n"
        info += f"层数: {config.get('num_hidden_layers', 0)}\n"
        info += f"注意力头数: {config.get('num_attention_heads', 0)}\n"
        
        if config.get('num_routed_experts', 0) > 0:
            info += f"MoE专家数: {config.get('num_routed_experts', 0)}\n"
            info += f"激活专家数: {config.get('num_activated_experts', 0)}\n"
        
        return info
    
    @staticmethod
    def _estimate_params(config: Dict[str, Any]) -> int:
        """估计模型参数量"""
        hidden_size = config.get('hidden_size', 0)
        num_layers = config.get('num_hidden_layers', 0)
        vocab_size = config.get('vocab_size', 0)
        num_routed_experts = config.get('num_routed_experts', 0)
        num_activated_experts = config.get('num_activated_experts', 0)
        
        if num_routed_experts > 0:
            moe_intermediate_size = config.get('moe_intermediate_size', 0)
            expert_params = (hidden_size * moe_intermediate_size * 3 * 2  # w1, w2, w3
                           + hidden_size + moe_intermediate_size)  # biases
            
            moe_layer_freq = config.get('moe_layer_freq', 1)
            moe_layers = num_layers // moe_layer_freq if moe_layer_freq > 0 else 0
            dense_layers = num_layers - moe_layers
            
            total_params = (
                # Embedding
                vocab_size * hidden_size * 2 +  # token embedding + output embedding
                # Attention layers
                num_layers * (hidden_size * hidden_size * 4 +  # Q, K, V, O projections
                             hidden_size * 2) +  # norms
                # Dense layers
                dense_layers * (hidden_size * config.get('intermediate_size', 0) * 3 +
                               hidden_size + config.get('intermediate_size', 0)) +
                # MoE layers
                moe_layers * (num_routed_experts * expert_params +
                             hidden_size * hidden_size)  # gate
            )
        else:
            intermediate_size = config.get('intermediate_size', 0)
            total_params = (
                vocab_size * hidden_size * 2 +  # embeddings
                num_layers * (hidden_size * hidden_size * 4 +  # attention
                             hidden_size * intermediate_size * 3 +  # mlp
                             hidden_size * 4)  # norms
            )
        
        return int(total_params)