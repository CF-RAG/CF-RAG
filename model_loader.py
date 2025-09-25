import torch
import logging
from typing import Dict, Any, Tuple
import gc
from pathlib import Path

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from sentence_transformers import CrossEncoder, SentenceTransformer
import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

def get_quantization_config():
    if config.MODEL_LOAD_IN_4BIT:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif config.MODEL_LOAD_IN_8BIT:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None

def check_device_availability(device: str) -> bool:
    if device == "cpu":
        return True
    
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            return False
        
        try:
            device_id = int(device.split(":")[1])
            return device_id < torch.cuda.device_count()
        except (IndexError, ValueError):
            return False
    
    return False

def print_gpu_memory_info(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        device_id = int(device.split(":")[1])
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        
        logger.info(f"GPU {device_id} Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

def load_llm() -> Tuple[Any, Any]:
    logger.info(f"Loading LLM model: {config.LLM_MODEL_ID}")
    logger.info(f"Target device: {config.LLM_DEVICE}")
    
    if not check_device_availability(config.LLM_DEVICE):
        logger.warning(f"Device {config.LLM_DEVICE} not available, falling back to CPU")
        device = "cpu"
    else:
        device = config.LLM_DEVICE
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL_ID,
            use_fast=config.USE_FAST_TOKENIZER,
            trust_remote_code=True,
            cache_dir=config.MODEL_CACHE_DIR if config.USE_MODEL_CACHE else None
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        quantization_config = get_quantization_config()
        
        logger.info("Loading LLM model...")
        model_kwargs = {
            "pretrained_model_name_or_path": config.LLM_MODEL_ID,
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "cache_dir": config.MODEL_CACHE_DIR if config.USE_MODEL_CACHE else None,
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            logger.info(f"Using quantization: 4bit={config.MODEL_LOAD_IN_4BIT}, 8bit={config.MODEL_LOAD_IN_8BIT}")
        else:
            model_kwargs["device_map"] = device
        
        if config.USE_FLASH_ATTENTION:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        if quantization_config is None and device != "cpu":
            model = model.to(device)
        
        if config.ENABLE_GRADIENT_CHECKPOINTING and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        model.eval()
        
        if config.TORCH_COMPILE and hasattr(torch, 'compile'):
            logger.info(f"Compiling model with mode: {config.TORCH_COMPILE_MODE}")
            model = torch.compile(model, mode=config.TORCH_COMPILE_MODE)
        
        print_gpu_memory_info(device)
        
        logger.info(f"LLM model loaded successfully on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load LLM model: {e}")
        raise RuntimeError(f"LLM model loading failed: {e}")

def load_reranker() -> Any:
    logger.info(f"Loading Reranker model: {config.RERANKER_MODEL_ID}")
    logger.info(f"Target device: {config.RERANKER_DEVICE}")
    
    if not check_device_availability(config.RERANKER_DEVICE):
        logger.warning(f"Device {config.RERANKER_DEVICE} not available, falling back to CPU")
        device = "cpu"
    else:
        device = config.RERANKER_DEVICE
    
    try:
        logger.info("Loading reranker model...")
        reranker = CrossEncoder(
            config.RERANKER_MODEL_ID,
            max_length=512,
            device=device,
            trust_remote_code=True
        )
        
        if hasattr(reranker, 'model'):
            reranker.model.eval()
        
        print_gpu_memory_info(device)
        
        logger.info(f"Reranker model loaded successfully on {device}")
        
        return reranker
        
    except Exception as e:
        logger.error(f"Failed to load reranker model: {e}")
        raise RuntimeError(f"Reranker model loading failed: {e}")

def load_embedding_model() -> Any:
    logger.info(f"Loading Embedding model: {config.EMBEDDING_MODEL_ID}")
    logger.info(f"Target device: {config.EMBEDDING_DEVICE}")
    
    if not check_device_availability(config.EMBEDDING_DEVICE):
        logger.warning(f"Device {config.EMBEDDING_DEVICE} not available, falling back to CPU")
        device = "cpu"
    else:
        device = config.EMBEDDING_DEVICE
    
    try:
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL_ID,
            device=device,
            trust_remote_code=True,
            cache_folder=config.MODEL_CACHE_DIR if config.USE_MODEL_CACHE else None
        )
        
        if hasattr(embedding_model, '_modules'):
            for module in embedding_model._modules.values():
                if hasattr(module, 'eval'):
                    module.eval()
        
        print_gpu_memory_info(device)
        
        logger.info(f"Embedding model loaded successfully on {device}")
        
        return embedding_model
        
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise RuntimeError(f"Embedding model loading failed: {e}")

def load_all_models() -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Starting CF-RAG Model Loading Process")
    logger.info("=" * 60)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_properties(i).name
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    models = {}
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("\n" + "─" * 50)
        logger.info("Step 1/3: Loading Large Language Model")
        logger.info("─" * 50)
        
        llm_model, llm_tokenizer = load_llm()
        models["llm_model"] = llm_model
        models["llm_tokenizer"] = llm_tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("\n" + "─" * 50)
        logger.info("Step 2/3: Loading Reranker Model")
        logger.info("─" * 50)
        
        reranker_model = load_reranker()
        models["reranker_model"] = reranker_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("\n" + "─" * 50)
        logger.info("Step 3/3: Loading Embedding Model")
        logger.info("─" * 50)
        
        embedding_model = load_embedding_model()
        models["embedding_model"] = embedding_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("\n" + "=" * 60)
        logger.info("Final GPU Memory Usage")
        logger.info("=" * 60)
        
        devices = set([config.LLM_DEVICE, config.RERANKER_DEVICE, config.EMBEDDING_DEVICE])
        for device in devices:
            if device.startswith("cuda"):
                print_gpu_memory_info(device)
        
        logger.info("\n" + "🎉" * 20)
        logger.info("All models loaded successfully!")
        logger.info("CF-RAG system is ready to use.")
        logger.info("🎉" * 20)
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        
        logger.info("Cleaning up partially loaded models...")
        for model_name, model_obj in models.items():
            try:
                if hasattr(model_obj, 'cpu'):
                    model_obj.cpu()
                del model_obj
            except Exception as cleanup_e:
                logger.warning(f"Failed to cleanup {model_name}: {cleanup_e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        raise RuntimeError(f"Model loading process failed: {e}")

def verify_models(models: Dict[str, Any]) -> bool:
    logger.info("Verifying loaded models...")
    
    required_keys = ["llm_model", "llm_tokenizer", "reranker_model", "embedding_model"]
    
    for key in required_keys:
        if key not in models:
            logger.error(f"Missing required model: {key}")
            return False
        
        if models[key] is None:
            logger.error(f"Model {key} is None")
            return False
    
    try:
        logger.info("Running basic functionality tests...")
        
        test_text = "Hello, world!"
        tokens = models["llm_tokenizer"](test_text, return_tensors="pt")
        logger.info(f"Tokenizer test passed: {len(tokens['input_ids'][0])} tokens")
        
        embeddings = models["embedding_model"].encode([test_text])
        logger.info(f"Embedding model test passed: shape {embeddings.shape}")
        
        scores = models["reranker_model"].predict([(test_text, test_text)])
        logger.info(f"Reranker test passed: score {scores[0]:.4f}")
        
        logger.info("All model verification tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False

def get_model_info(models: Dict[str, Any]) -> Dict[str, Any]:
    info = {
        "model_ids": {
            "llm": config.LLM_MODEL_ID,
            "reranker": config.RERANKER_MODEL_ID,
            "embedding": config.EMBEDDING_MODEL_ID
        },
        "devices": {
            "llm": config.LLM_DEVICE,
            "reranker": config.RERANKER_DEVICE,
            "embedding": config.EMBEDDING_DEVICE
        },
        "model_types": {},
        "parameters": {}
    }
    
    try:
        for key, model in models.items():
            if model is not None:
                info["model_types"][key] = str(type(model).__name__)
        
        if "llm_model" in models and models["llm_model"] is not None:
            llm_params = sum(p.numel() for p in models["llm_model"].parameters())
            info["parameters"]["llm"] = f"{llm_params / 1e9:.2f}B"
        
        if "embedding_model" in models and models["embedding_model"] is not None:
            embed_dim = models["embedding_model"].get_sentence_embedding_dimension()
            info["embedding_dimension"] = embed_dim
            
    except Exception as e:
        logger.warning(f"Failed to get complete model info: {e}")
    
    return info

if __name__ == "__main__":
    try:
        models = load_all_models()
        
        if verify_models(models):
            print("\n✅ All models loaded and verified successfully!")
            
            model_info = get_model_info(models)
            print("\n📊 Model Information:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
        else:
            print("\n❌ Model verification failed!")
            
    except Exception as e:
        print(f"\n💥 Error: {e}")
        exit(1)