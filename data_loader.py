import random
import os
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_sampled_dataset(dataset, num_samples, random_seed):
    dataset_list = list(dataset)
    if num_samples is not None and num_samples < len(dataset_list):
        random.seed(random_seed)
        dataset_list = random.sample(dataset_list, num_samples)
        logger.info(f"Randomly sampled {num_samples} examples from the dataset.")
    else:
        logger.info(f"Using the full dataset with {len(dataset_list)} examples.")
    return dataset_list

def configure_huggingface_mirror():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    logger.info("Configured HuggingFace to use mirror endpoint: https://hf-mirror.com")

def load_hotpotqa_data(cache_dir, num_samples=None, random_seed=42):
    logger.info("Loading HotpotQA dataset...")
    try:
        dataset = load_dataset("hotpot_qa", "distractor", cache_dir=cache_dir, split="validation")
    except Exception as e:
        logger.error(f"Failed to load HotpotQA dataset: {e}")
        logger.info("Please download the dataset manually and place it in the cache directory.")
        raise
    
    dataset_list = _get_sampled_dataset(dataset, num_samples, random_seed)
    
    processed_data = []
    for example in dataset_list:
        context_docs = ["".join(para) for para in example['context']['sentences']]
        processed_data.append({
            "question": example['question'],
            "answer": [example['answer']],
            "context": context_docs
        })
    return processed_data

def load_triviaqa_data(cache_dir, num_samples=None, random_seed=42):
    logger.info("Loading TriviaQA dataset...")
    try:
        dataset = load_dataset("trivia_qa", "unfiltered", cache_dir=cache_dir, split="validation")
    except Exception as e:
        logger.error(f"Failed to load TriviaQA dataset: {e}")
        logger.info("Please download the dataset manually and place it in the cache directory.")
        raise
        
    dataset_list = _get_sampled_dataset(dataset, num_samples, random_seed)

    processed_data = []
    for example in dataset_list:
        context_docs = []
        
        if 'entity_pages' in example and example['entity_pages']:
            if 'wiki_context' in example['entity_pages']:
                wiki_contexts = example['entity_pages']['wiki_context']
                if wiki_contexts:
                    context_docs.extend(wiki_contexts)
        
        if not context_docs and 'search_results' in example and example['search_results']:
            if 'description' in example['search_results']:
                descriptions = example['search_results']['description']
                if descriptions:
                    context_docs.extend(descriptions)
        
        if not context_docs:
            context_docs = [f"Question context: {example['question']}"]
            logger.warning(f"No context found for question: {example['question'][:50]}...")
        
        processed_data.append({
            "question": example['question'],
            "answer": list(example['answer']['aliases']),
            "context": context_docs
        })
    return processed_data

def load_popqa_data(cache_dir, num_samples=None, random_seed=42):
    logger.info("Loading PopQA dataset...")
    try:
        dataset = load_dataset("akariasai/PopQA", cache_dir=cache_dir, split="test")
    except Exception as e:
        logger.error(f"Failed to load PopQA dataset: {e}")
        logger.info("Please download the dataset manually and place it in the cache directory.")
        raise
        
    dataset_list = _get_sampled_dataset(dataset, num_samples, random_seed)

    processed_data = []
    for example in dataset_list:
        answers = [example['obj']]
        if 'possible_answers' in example and example['possible_answers']:
            try:
                import ast
                possible_answers = ast.literal_eval(example['possible_answers'])
                if isinstance(possible_answers, list):
                    answers.extend(possible_answers)
            except:
                answers.append(example['possible_answers'])
        answers = list(dict.fromkeys(answers))
        
        context_text = f"Question: {example['question']} Subject: {example['subj']} ({example.get('s_wiki_title', '')})"
        
        processed_data.append({
            "question": example['question'],
            "answer": answers,
            "context": [context_text]
        })
    return processed_data

def load_musique_data(cache_dir, num_samples=None, random_seed=42):
    logger.info("Loading MusiQue-QA dataset...")
    
    import os
    import json
    from pathlib import Path
    
    local_musique_path = Path(cache_dir) / "musique_local"
    
    if local_musique_path.exists():
        logger.info("Found local MusiQue directory, checking for official data...")
        
        official_files = [
            local_musique_path / "musique_ans_v1.0_dev.jsonl",
            local_musique_path / "musique_full_v1.0_dev.jsonl"
        ]
        
        for file_path in official_files:
            if file_path.exists():
                logger.info(f"Loading official MusiQue data from: {file_path}")
                return _load_musique_from_jsonl(file_path, num_samples, random_seed)
        
        sample_files = [
            local_musique_path / "dev.json",
            local_musique_path / "validation.json"
        ]
        
        for file_path in sample_files:
            if file_path.exists():
                logger.info(f"Loading sample MusiQue data from: {file_path}")
                return _load_musique_from_json(file_path, num_samples, random_seed)
    
    try:
        logger.info("Trying to load MusiQue from HuggingFace Hub...")
        possible_names = [
            "musique",
            "musique-qa",
            "mu-sique/musique-qa",
            "allenai/musique"
        ]
        
        dataset = None
        for name in possible_names:
            try:
                logger.info(f"Trying dataset name: {name}")
                dataset = load_dataset(name, cache_dir=cache_dir, split="validation")
                logger.info(f"Successfully loaded MusiQue dataset as: {name}")
                break
            except Exception as e:
                logger.info(f"Failed to load with name {name}: {str(e)[:100]}...")
                continue
        
        if dataset is None:
            raise Exception("Could not find MusiQue dataset with any known name")
            
        dataset_list = _get_sampled_dataset(dataset, num_samples, random_seed)
        
        processed_data = []
        for example in dataset_list:
            if 'paragraphs' in example:
                context_docs = [p['paragraph_text'] for p in example['paragraphs']]
            elif 'context' in example:
                context_docs = [example['context']]
            else:
                context_docs = [f"Question context: {example['question']}"]
                logger.warning(f"No standard context found for MusiQue question: {example['question'][:50]}...")
                
            processed_data.append({
                "question": example['question'],
                "answer": [example.get('answer', example.get('answer_text', 'Unknown'))],
                "context": context_docs
            })
        return processed_data
        
    except Exception as e:
        logger.error(f"Failed to load MusiQue dataset: {e}")
        logger.info("Please check the correct dataset name or download manually.")
        logger.info("You may need to manually download MusiQue from https://github.com/stonybrooknlp/musique")
        raise

def _load_musique_from_jsonl(file_path, num_samples=None, random_seed=42):
    import json
    import random
    
    logger.info(f"Loading MusiQue from JSONL: {file_path}")
    
    raw_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    logger.info(f"Loaded {len(raw_data)} examples from {file_path}")
    
    if num_samples is not None and num_samples < len(raw_data):
        random.seed(random_seed)
        raw_data = random.sample(raw_data, num_samples)
        logger.info(f"Randomly sampled {num_samples} examples from MusiQue dataset.")
    
    processed_data = []
    for example in raw_data:
        context_docs = []
        supporting_docs = []
        non_supporting_docs = []
        
        if 'paragraphs' in example and example['paragraphs']:
            for p in example['paragraphs']:
                if 'paragraph_text' in p:
                    if 'title' in p and p['title']:
                        enhanced_text = f"Title: {p['title']}\nContent: {p['paragraph_text']}"
                    else:
                        enhanced_text = p['paragraph_text']
                    
                    if p.get('is_supporting', False):
                        supporting_docs.append(enhanced_text)
                    else:
                        non_supporting_docs.append(enhanced_text)
            
            context_docs = supporting_docs + non_supporting_docs
        
        answer_list = []
        if 'answer' in example and example['answer']:
            answer_list.append(example['answer'])
        
        if 'answer_aliases' in example and example['answer_aliases']:
            answer_list.extend(example['answer_aliases'])
        
        seen = set()
        unique_answers = []
        for ans in answer_list:
            if ans.lower() not in seen:
                unique_answers.append(ans)
                seen.add(ans.lower())
        
        if not unique_answers:
            unique_answers = ['Unknown']
        
        if not context_docs:
            context_docs = [f"Question context: {example.get('question', 'Unknown question')}"]
        
        processed_data.append({
            "question": example.get('question', 'Unknown question'),
            "answer": unique_answers,
            "context": context_docs
        })
    
    logger.info(f"Successfully processed {len(processed_data)} MusiQue examples.")
    return processed_data

def _load_musique_from_json(file_path, num_samples=None, random_seed=42):
    import json
    import random
    
    logger.info(f"Loading MusiQue from JSON: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, dict) and 'data' in raw_data:
        raw_data = raw_data['data']
    elif isinstance(raw_data, dict) and 'examples' in raw_data:
        raw_data = raw_data['examples']
    
    if num_samples is not None and num_samples < len(raw_data):
        random.seed(random_seed)
        raw_data = random.sample(raw_data, num_samples)
        logger.info(f"Randomly sampled {num_samples} examples from sample MusiQue dataset.")
    
    processed_data = []
    for example in raw_data:
        context_docs = []
        if 'paragraphs' in example:
            context_docs = [p.get('paragraph_text', p.get('text', str(p))) for p in example['paragraphs']]
        elif 'context' in example:
            if isinstance(example['context'], list):
                context_docs = example['context']
            else:
                context_docs = [example['context']]
        
        if not context_docs:
            context_docs = [f"Question context: {example.get('question', 'Unknown question')}"]
        
        processed_data.append({
            "question": example.get('question', 'Unknown question'),
            "answer": [example.get('answer', example.get('answer_text', 'Unknown'))],
            "context": context_docs
        })
    
    logger.info(f"Successfully loaded {len(processed_data)} examples from sample MusiQue dataset.")
    return processed_data
    
def load_arc_c_data(cache_dir, num_samples=None, random_seed=42):
    logger.info("Loading ARC-C dataset...")
    try:
        dataset = load_dataset("ai2_arc", "ARC-Challenge", cache_dir=cache_dir, split="test")
    except Exception as e:
        logger.error(f"Failed to load ARC-C dataset: {e}")
        logger.info("Please download the dataset manually and place it in the cache directory.")
        raise
        
    dataset_list = _get_sampled_dataset(dataset, num_samples, random_seed)

    processed_data = []
    for example in dataset_list:
        correct_answer = example['choices']['text'][example['choices']['label'].index(example['answerKey'])]
        processed_data.append({
            "question": example['question'],
            "answer": [correct_answer],
            "context": None
        })
    return processed_data

def load_pubhealth_data(cache_dir, num_samples=None, random_seed=42):
    logger.info("Loading PubHealth dataset...")
    try:
        pubhealth_cache_dir = os.path.join(cache_dir, "pubhealth_cache")
        dataset = load_dataset("bigbio/pubhealth", "pubhealth_source", cache_dir=pubhealth_cache_dir, split="test")
    except Exception as e:
        logger.error(f"Failed to load PubHealth dataset: {e}")
        logger.info("Please download the dataset manually and place it in the cache directory.")
        raise
        
    dataset_list = _get_sampled_dataset(dataset, num_samples, random_seed)

    processed_data = []
    for example in dataset_list:
        claim = example['claim']
        
        label_map = {
            0: "False",
            1: "True", 
            2: "Mixture",
            3: "Unproven"
        }
        correct_answer = label_map.get(example['label'], "Unknown")
        
        context_parts = []
        if example.get('explanation'):
            context_parts.append(f"Explanation: {example['explanation']}")
        if example.get('main_text'):
            context_parts.append(f"Main text: {example['main_text']}")
        
        if not context_parts:
            context_parts = [f"Fact-checking claim: {claim}"]
            
        processed_data.append({
            "question": f"Is this claim true or false? {claim}",
            "answer": [correct_answer],
            "context": context_parts
        })
    
    logger.info(f"Successfully loaded {len(processed_data)} PubHealth examples.")
    return processed_data

DATASET_LOADERS = {
    "hotpotqa": load_hotpotqa_data,
    "triviaqa": load_triviaqa_data,
    "popqa": load_popqa_data,
    "musique": load_musique_data,
    "arc_c": load_arc_c_data,
    "pubhealth": load_pubhealth_data
}

AVAILABLE_DATASETS = ["hotpotqa", "triviaqa", "popqa", "arc_c", "pubhealth"]

def load_and_process_data(dataset_name: str, cache_dir: str, num_samples: int = None, random_seed: int = 42, use_mirror: bool = True):
    if use_mirror:
        configure_huggingface_mirror()
    
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets are: {list(DATASET_LOADERS.keys())}")
    
    loader_fn = DATASET_LOADERS[dataset_name]
    return loader_fn(cache_dir, num_samples, random_seed)