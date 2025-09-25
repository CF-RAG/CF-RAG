import argparse
import json
import time
import os
from tqdm import tqdm
from pathlib import Path

import config
import model_loader
from retriever import KnowledgeBaseRetriever
from cfrag_pipeline import CFRAGPipeline
from evaluation import evaluate
from improved_data_loader import load_and_process_data

def ensure_datasets_dependency():
    try:
        import datasets
        print(f"✅ datasets library version: {datasets.__version__}")
    except ImportError:
        print("❌ datasets library not found. Installing...")
        os.system("conda install -c huggingface datasets -y")
        print("✅ datasets library installed.")

def main():
    ensure_datasets_dependency()
    
    parser = argparse.ArgumentParser(description="Run evaluation for the CF-RAG model.")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["hotpotqa", "triviaqa", "popqa", "musique", "pubhealth"], 
                       help="Name of the dataset to evaluate on.")
    parser.add_argument("--num_samples", type=int, default=None, 
                       help="Number of samples to run evaluation on. Defaults to all.")
    parser.add_argument("--cache_dir", type=str, default="./huggingface_cache", 
                       help="Directory to cache Hugging Face datasets.")
    parser.add_argument("--output_dir", type=str, default="./eval_results", 
                       help="Directory to save evaluation results.")
    parser.add_argument("--random_seed", type=int, default=42, 
                       help="Random seed for sampling.")
    parser.add_argument("--use_mirror", action="store_true", default=True,
                       help="Use HuggingFace mirror for Chinese servers.")
    parser.add_argument("--skip_download", action="store_true", default=False,
                       help="Skip downloading and use cached datasets only.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.output_file = output_dir / f"{args.dataset}_results_{timestamp}.jsonl"

    print(f"Starting evaluation for dataset: {args.dataset}")
    print(f"Results will be saved to: {args.output_file}")
    print(f"Samples: {args.num_samples if args.num_samples else 'All'}")
    print(f"Use mirror: {args.use_mirror}")

    print("\nInitializing models and pipeline...")
    try:
        models = model_loader.load_all_models()
        retriever = KnowledgeBaseRetriever(models["embedding_model"])
        pipeline = CFRAGPipeline(models, retriever)
        print("✅ System initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return 1

    print(f"\nLoading {args.dataset} dataset...")
    try:
        eval_data = load_and_process_data(
            dataset_name=args.dataset,
            cache_dir=args.cache_dir,
            num_samples=args.num_samples,
            random_seed=args.random_seed,
            use_mirror=args.use_mirror
        )
        print(f"✅ Loaded {len(eval_data)} examples.")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        if args.skip_download:
            print("💡 Try running without --skip_download flag.")
        print("💡 Or manually download the dataset to the cache directory.")
        return 1

    if args.dataset == 'arc_c':
        print("ARC-C dataset detected. Loading the general knowledge base...")
        try:
            retriever.load_index()
            if not retriever.is_ready():
                print("\n❌ ERROR: FAISS index not found for ARC-C.")
                print("💡 Please run 'python build_knowledge_base.py' first to create the index.")
                return 1
        except Exception as e:
            print(f"❌ Failed to load knowledge base: {e}")
            return 1

    print(f"\nStarting evaluation on {len(eval_data)} examples...")
    all_results = []
    total_em = 0
    total_f1 = 0
    successful_runs = 0

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(tqdm(eval_data, desc=f"Evaluating on {args.dataset}")):
            question = example['question']
            ground_truths = example['answer']
            
            if example['context'] is not None:
                try:
                    retriever.build_temp_index_from_docs(example['context'])
                except Exception as e:
                    print(f"\n⚠️ Warning: Failed to build temp index for question {i}: {e}")

            try:
                start_time = time.time()
                prediction = pipeline.run(question)
                latency = time.time() - start_time
                
                em, f1 = evaluate(prediction, ground_truths, question)
                total_em += em
                total_f1 += f1
                successful_runs += 1
                
                if i < 3:
                    print(f"\nExample {i}:")
                    print(f"Q: {question[:100]}...")
                    print(f"A: {prediction[:100]}...")
                    print(f"GT: {ground_truths[0][:100] if ground_truths else 'N/A'}...")
                    print(f"EM: {em}, F1: {f1:.3f}")
            
            except Exception as e:
                print(f"\n❌ Error processing question {i}: {question[:50]}...")
                print(f"Error: {str(e)[:100]}...")
                prediction = "ERROR"
                latency = -1
                em, f1 = 0, 0

            result = {
                "id": i,
                "question": question,
                "prediction": prediction,
                "ground_truths": ground_truths,
                "em": em,
                "f1": f1,
                "latency": latency
            }
            all_results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if successful_runs > 0:
        avg_em = (total_em / len(eval_data)) * 100
        avg_f1 = (total_f1 / len(eval_data)) * 100
    else:
        avg_em = avg_f1 = 0

    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Total examples: {len(eval_data)}")
    print(f"Successful runs: {successful_runs}")
    if args.num_samples:
        print(f"Sampling: {args.num_samples} random samples")
    print(f"Average EM: {avg_em:.2f}%")
    print(f"Average F1: {avg_f1:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)