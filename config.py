import os
from pathlib import Path

LLM_MODEL_ID = "./model_cache/llama3-8b-instruct-local/"
RERANKER_MODEL_ID = "./model_cache/bge-reranker-large-local/"
EMBEDDING_MODEL_ID = "./model_cache/bge-large-en-v1.5-local/"

LLM_DEVICE = "cuda:1"
RERANKER_DEVICE = "cuda:1"
EMBEDDING_DEVICE = "cuda:1"

MAX_NEW_TOKENS = 512
MAX_INPUT_LENGTH = 4096
TOP_P = 0.9
TOP_K = 50

EXPLANATORY_TEMPERATURE = 0.1
COUNTERFACTUAL_TEMPERATURE = 0.7

RETRIEVAL_TOP_K = 15
RERANKER_TOP_K = 5
RERANKER_WEIGHT = 0.7

RELEVANCE_THRESHOLD = 0.1

NUM_CLUSTERS = 4
NUM_DRAFTS = 4
MIN_DOCS_PER_DRAFT = 4
USE_SEMANTIC_CLUSTERING = True
CLUSTER_OVERLAP_RATIO = 0.3
INCLUDE_TOP_EVIDENCE = True
VERIFICATION_WEIGHT = 0.4
ENABLE_MULTI_ASPECT_CONSISTENCY = False
CONSISTENCY_ASPECTS = [
    "factual_support",
    "logical_coherence",
    "completeness"
]

ENABLE_SIMPLIFIED_MODE = False
SIMPLIFIED_DRAFT_COUNT = 1

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SUPPORTED_DOCUMENT_FORMATS = ['.txt', '.md', '.json']
MAX_DOCUMENT_LENGTH = 50000

MAX_COUNTERFACTUAL_QUERIES = 3
MAX_EVIDENCE_DOCS = 5

PROJECT_ROOT = Path(__file__).parent.absolute()

KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index"

SUPPORTED_DOCUMENT_FORMATS = [".txt", ".md", ".pdf", ".docx", ".json"]

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_DOCUMENT_LENGTH = 10000

MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"
USE_MODEL_CACHE = True

DRAFT_PROMPT_TEMPLATE = """You are an expert analyst. Based on the provided evidence, give a comprehensive answer to the question with detailed reasoning and evidence citations.

**Question:** {original_query}

**Evidence:**
{evidence_documents}

**Instructions:**
1. Analyze each piece of evidence carefully
2. Cite specific evidence using [Evidence X] format
3. Provide step-by-step reasoning
4. Give a clear, well-supported answer

**Response Format:**
**Reasoning:**
[Provide detailed step-by-step analysis citing specific evidence]

**Answer:**
[Give the final answer based on your reasoning]

**Your Analysis:**
"""

COUNTERFACTUAL_PROMPT_TEMPLATE = """
You are an expert in counterfactual reasoning and critical thinking. Generate {num_queries} high-quality counterfactual questions that will help identify the most decisive evidence for answering the original question.

**Original Question:** {original_query}

**Guidelines for counterfactual questions:**
1. **Semantic similarity**: Use similar entities, concepts, and vocabulary as the original
2. **Logical opposition**: Ask about the opposite scenario, alternative entities, or different outcomes  
3. **Evidence discrimination**: Help distinguish between relevant and irrelevant evidence
4. **Specificity**: Be as specific as the original question

**Examples:**
- Original: "Are X and Y both from the United States?" → Counterfactual: "Are X and Y from different countries?"
- Original: "Who was the director of movie Z?" → Counterfactual: "Who was NOT involved in directing movie Z?"
- Original: "When did event A occur?" → Counterfactual: "When did event A NOT occur?"

**Generate {num_queries} counterfactual questions (one per line, no numbering):**
"""

EXPLANATORY_PROMPT_TEMPLATE = """You are an expert AI assistant specializing in evidence-based reasoning. Provide a comprehensive answer with detailed analysis, evidence citations, and clear reasoning.

**Question:** {original_query}

**Selected Evidence:**
{evidence_documents}

**Instructions:**
1. Analyze the evidence systematically
2. Cite specific evidence using [Evidence X] format  
3. Show your reasoning process step-by-step
4. Address potential counterarguments if relevant
5. Provide a clear, well-supported conclusion

**Response Format:**

**Evidence Analysis:**
[Analyze each piece of evidence and its relevance]

**Reasoning Process:**
[Show step-by-step logical reasoning, citing evidence]

**Conclusion:**
[Provide the final answer with supporting rationale]

**Your Response:**"""

LOG_LEVEL = "INFO"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "cfrag.log"

BATCH_SIZE = 1

ENABLE_GRADIENT_CHECKPOINTING = True
USE_FLASH_ATTENTION = False

API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

ENABLE_DEBUG_MODE = False
SAVE_INTERMEDIATE_RESULTS = False
EXPERIMENT_OUTPUT_DIR = PROJECT_ROOT / "experiments"

RETRIEVAL_TEMPERATURE = 0.0

MODEL_LOAD_IN_8BIT = False
MODEL_LOAD_IN_4BIT = False
USE_FAST_TOKENIZER = True

TORCH_COMPILE = False
TORCH_COMPILE_MODE = "default"

TEST_QUERIES = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What are the benefits of renewable energy?",
]

EVALUATION_METRICS = [
    "relevance_score",
    "factual_accuracy", 
    "explanation_quality",
    "counterfactual_effectiveness"
]

def get_env_config():
    env_config = {}
    
    if os.getenv("CF_RAG_LLM_MODEL"):
        env_config["LLM_MODEL_ID"] = os.getenv("CF_RAG_LLM_MODEL")
    
    if os.getenv("CF_RAG_RERANKER_MODEL"):
        env_config["RERANKER_MODEL_ID"] = os.getenv("CF_RAG_RERANKER_MODEL")
    
    if os.getenv("CF_RAG_EMBEDDING_MODEL"):
        env_config["EMBEDDING_MODEL_ID"] = os.getenv("CF_RAG_EMBEDDING_MODEL")
    
    if os.getenv("CF_RAG_DEVICE"):
        device = os.getenv("CF_RAG_DEVICE")
        env_config["LLM_DEVICE"] = device
        env_config["RERANKER_DEVICE"] = device
        env_config["EMBEDDING_DEVICE"] = device
    
    if os.getenv("CF_RAG_KNOWLEDGE_BASE"):
        env_config["KNOWLEDGE_BASE_PATH"] = Path(os.getenv("CF_RAG_KNOWLEDGE_BASE"))
    
    if os.getenv("CF_RAG_FAISS_INDEX"):
        env_config["FAISS_INDEX_PATH"] = Path(os.getenv("CF_RAG_FAISS_INDEX"))
    
    return env_config

def validate_config():
    valid_devices = ["cpu"] + [f"cuda:{i}" for i in range(8)]
    if LLM_DEVICE not in valid_devices:
        raise ValueError(f"Invalid LLM_DEVICE: {LLM_DEVICE}")
    
    if RERANKER_DEVICE not in valid_devices:
        raise ValueError(f"Invalid RERANKER_DEVICE: {RERANKER_DEVICE}")
    
    if EMBEDDING_DEVICE not in valid_devices:
        raise ValueError(f"Invalid EMBEDDING_DEVICE: {EMBEDDING_DEVICE}")
    
    if not 0 <= EXPLANATORY_TEMPERATURE <= 2.0:
        raise ValueError(f"Invalid EXPLANATORY_TEMPERATURE: {EXPLANATORY_TEMPERATURE}, should be in [0, 2.0]")
    
    if not 0 < COUNTERFACTUAL_TEMPERATURE <= 2.0:
        raise ValueError(f"Invalid COUNTERFACTUAL_TEMPERATURE: {COUNTERFACTUAL_TEMPERATURE}, should be in (0, 2.0]")
    
    if not 0 <= RERANKER_WEIGHT <= 1.0:
        raise ValueError(f"Invalid RERANKER_WEIGHT: {RERANKER_WEIGHT}, should be in [0, 1.0]")
    
    if RETRIEVAL_TOP_K <= 0:
        raise ValueError(f"Invalid RETRIEVAL_TOP_K: {RETRIEVAL_TOP_K}, should be > 0")
    
    if RERANKER_TOP_K <= 0:
        raise ValueError(f"Invalid RERANKER_TOP_K: {RERANKER_TOP_K}, should be > 0")
    
    if not KNOWLEDGE_BASE_PATH.exists():
        print(f"Warning: KNOWLEDGE_BASE_PATH does not exist: {KNOWLEDGE_BASE_PATH}")
    
    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_INTERMEDIATE_RESULTS:
        EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def init_config():
    env_config = get_env_config()
    globals().update(env_config)
    
    validate_config()
    
    print("CF-RAG Configuration initialized successfully")
    print(f"LLM Model: {LLM_MODEL_ID}")
    print(f"Reranker Model: {RERANKER_MODEL_ID}")
    print(f"Embedding Model: {EMBEDDING_MODEL_ID}")
    print(f"Device: {LLM_DEVICE}")
    print(f"Knowledge Base: {KNOWLEDGE_BASE_PATH}")

if __name__ != "__main__":
    try:
        init_config()
    except Exception as e:
        print(f"Warning: Config initialization failed: {e}")