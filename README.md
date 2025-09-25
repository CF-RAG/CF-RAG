# CF-RAG: Counterfactual-Driven Retrieval-Augmented Generation

## Overview

CF-RAG is a new framework that enhances Retrieval-Augmented Generation (RAG) with causal reasoning to overcome a critical vulnerability in existing systems: the Correlation Trap.


## Supported Datasets

CF-RAG has been evaluated on multiple benchmark datasets:

- **HotpotQA**: Multi-hop question answering dataset
- **TriviaQA**: Trivia question answering with evidence passages
- **PopQA**: Popular entity-centric question answering
- **MusiQue**: Multi-step question answering requiring reasoning
- **PubHealth**: Health claim verification dataset


## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 24GB GPU memory for optimal performance

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/CF-RAG/CF-RAG.git
cd CF-RAG
```

2. **Create and activate a conda environment**
```bash
conda create -n cfrag python=3.9
conda activate cfrag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download models** (Optional - models will be downloaded automatically)
```bash
# The system will automatically download required models to model_cache/
# Ensure you have sufficient disk space (~20GB for all models)
```

4. **Prepare knowledge base**
```bash
# Create knowledge base directory and add your documents
mkdir -p knowledge_base
# Add your .txt, .md, or .json files to the knowledge_base directory
```

## Usage

### Dataset Evaluation

```bash
# Evaluate on multiple datasets
python run_evaluation.py --dataset hotpotqa --num-samples 50
python run_evaluation.py --dataset triviaqa --num-samples 50
python run_evaluation.py --dataset popqa --num-samples 50
python musique_evaluation.py --num-samples 50
python run_evaluation.py --dataset pubhealth --num-samples 50
```

## Configuration

Key configuration parameters in `config.py`:

### Model Configuration
```python
# Model paths (will download automatically if not present)
LLM_MODEL_ID = "./model_cache/llama3-8b-instruct-local/"
RERANKER_MODEL_ID = "./model_cache/bge-reranker-large-local/"
EMBEDDING_MODEL_ID = "./model_cache/bge-large-en-v1.5-local/"

# Device allocation
LLM_DEVICE = "cuda:0"  # Adjust based on your GPU setup
RERANKER_DEVICE = "cuda:0"
EMBEDDING_DEVICE = "cuda:0"
```

### Pipeline Parameters
```python
# Generation parameters
EXPLANATORY_TEMPERATURE = 0.1    # Final answer generation
COUNTERFACTUAL_TEMPERATURE = 0.7 # Counterfactual generation

# Retrieval parameters
RETRIEVAL_TOP_K = 15             # Initial retrieval count
RERANKER_TOP_K = 5               # Final evidence count
RERANKER_WEIGHT = 0.7            # Relevance vs refutation weight

# Multi-draft parameters
NUM_CLUSTERS = 4                 # Evidence clustering
NUM_DRAFTS = 4                   # Number of answer drafts
```

### Advanced Features
```python
# Enable simplified mode (single-stage evaluation)
ENABLE_SIMPLIFIED_MODE = False

# Enable semantic clustering
USE_SEMANTIC_CLUSTERING = True

# Multi-aspect consistency evaluation
ENABLE_MULTI_ASPECT_CONSISTENCY = False
```

## Evaluation Metrics

CF-RAG uses standard QA evaluation metrics:

- **Exact Match (EM)**: Binary match between predicted and ground truth answers
- **F1 Score**: Token-level F1 score between prediction and ground truth
- **Processing Time**: Average time per question
- **Smart Matching**: Enhanced matching for various answer formats

## Project Structure

```
CF-RAG/
├── cfrag_pipeline.py           # Core CF-RAG pipeline implementation
├── main.py                     # Main entry point and CLI
├── config.py                   # Configuration parameters
├── model_loader.py             # Model loading utilities
├── retriever.py                # Knowledge base retriever
├── evaluation.py               # Evaluation utilities
├── data_loader.py              # Dataset loading functions
├── run_evaluation.py           # Multi-dataset evaluation
├── musique_evaluation.py       # MusiQue evaluation
├── requirements.txt            # Python dependencies
├── knowledge_base/             # Document storage
├── faiss_index/                # Vector index storage
├── model_cache/                # Downloaded models
├── logs/                       # System logs
└── eval_results/               # Evaluation results
```

## Hardware Requirements

### Minimum Requirements
- GPU: 8GB VRAM (RTX 3080, RTX 4070, etc.)
- RAM: 16GB
- Storage: 50GB free space

### Recommended Requirements (Current Testing Configuration)
- **GPU: A100 80GB VRAM**  (Currently Tested)
  - Supports full precision models
  - Enables parallel draft generation
  - Optimal for complete 5-stage CF-RAG pipeline
- **RAM: 32GB+**
- **Storage: 100GB+ free space** (for models and datasets)

### Alternative Configurations
- **High-End Consumer**: RTX 4090 (24GB) - requires some optimizations
- **Professional**: RTX A6000 (48GB) - good performance with minor adjustments
- **Data Center**: H100, A100 80GB - excellent performance

## Performance

CF-RAG demonstrates strong performance across multiple datasets:

- **HotpotQA**: Improved multi-hop reasoning capabilities
- **TriviaQA**: Enhanced factual accuracy through counterfactual validation
- **PopQA**: Better handling of popular entity queries
- **MusiQue**: Superior performance on complex multi-step reasoning
- **PubHealth**: Reliable health claim verification
