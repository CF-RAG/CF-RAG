import argparse
import json
import time
import os
from tqdm import tqdm
from pathlib import Path

import model_loader
from retriever import KnowledgeBaseRetriever
from cfrag_pipeline import CFRAGPipeline
from evaluation import normalize_answer
from improved_data_loader import load_and_process_data

def enhanced_evaluate_musique(prediction, ground_truths):
    pred_normalized = normalize_answer(prediction)
    
    best_em = 0
    best_f1 = 0
    
    for gt in ground_truths:
        gt_normalized = normalize_answer(gt)
        
        em = 0
        if gt_normalized in pred_normalized or pred_normalized in gt_normalized:
            em = 1
        else:
            gt_words = set(gt_normalized.split())
            pred_words = set(pred_normalized.split())
            
            if len(gt_words) > 0:
                word_overlap = len(gt_words & pred_words) / len(gt_words)
                if word_overlap >= 0.6:
                    em = 1
        
        if gt_normalized and pred_normalized:
            gt_tokens = gt_normalized.split()
            pred_tokens = pred_normalized.split()
            
            if len(pred_tokens) == 0:
                f1 = 0
            else:
                common = set(gt_tokens) & set(pred_tokens)
                if len(common) == 0:
                    f1 = 0
                else:
                    precision = len(common) / len(pred_tokens)
                    recall = len(common) / len(gt_tokens)
                    f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)
    
    return best_em, best_f1

def main():
    parser = argparse.ArgumentParser(description="Run CFRAG evaluation on MusiQue dataset")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    
    output_dir = Path("./eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"musique_optimized_results_{timestamp}.jsonl"

    print(f"🎵 Starting optimized MusiQue evaluation")
    print(f"📝 Results will be saved to: {output_file}")
    print(f"🔢 Samples: {args.num_samples if args.num_samples else 'All'}")
    print()

    print("⚙️ Initializing models and pipeline...")
    try:
        models = model_loader.load_all_models()
        retriever = KnowledgeBaseRetriever(models["embedding_model"])
        pipeline = CFRAGPipeline(models, retriever)
        print("✅ System initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return

    print("📊 Loading MusiQue dataset...")
    try:
        examples = load_and_process_data(
            dataset_name="musique",
            cache_dir="./huggingface_cache",
            num_samples=args.num_samples,
            random_seed=args.random_seed,
            use_mirror=True
        )
        print(f"✅ Loaded {len(examples)} examples.")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    total_em = 0
    total_f1 = 0
    successful_runs = 0
    results = []

    for i, example in enumerate(tqdm(examples, desc="Evaluating MusiQue")):
        try:
            question = example["question"]
            ground_truths = example["answer"]
            context = example["context"]
            
            print(f"\n🔍 Example {i}:")
            print(f"Q: {question[:100]}...")

            start_time = time.time()
            prediction = pipeline.run(question)
            latency = time.time() - start_time
            
            print(f"A: {prediction[:150]}...")
            print(f"GT: {ground_truths[0]}...")
       
            em, f1 = enhanced_evaluate_musique(prediction, ground_truths)
            
            print(f"EM: {em}, F1: {f1:.3f}")

            total_em += em
            total_f1 += f1
            successful_runs += 1
            
            result = {
                "id": i,
                "question": question,
                "prediction": prediction,
                "ground_truths": ground_truths,
                "em": bool(em),
                "f1": float(f1),
                "latency": latency
            }
            results.append(result)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
        except Exception as e:
            print(f"❌ Error processing example {i}: {e}")
            continue

    if successful_runs > 0:
        avg_em = (total_em / successful_runs) * 100
        avg_f1 = (total_f1 / successful_runs) * 100
    else:
        avg_em = avg_f1 = 0

    print("\n" + "=" * 50)
    print("📈 EVALUATION COMPLETE")
    print("=" * 50)
    print(f"📊 Dataset: MusiQue (Optimized)")
    print(f"🔢 Total examples: {len(examples)}")
    print(f"✅ Successful runs: {successful_runs}")
    print(f"🎲 Sampling: {args.num_samples if args.num_samples else 'All'} samples")
    print(f"🎯 Average EM: {avg_em:.2f}%")
    print(f"📏 Average F1: {avg_f1:.2f}%")
    print(f"⏱️ Results saved to: {output_file}")
    print("=" * 50)

if __name__ == "__main__":
    main()
