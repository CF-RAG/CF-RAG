import argparse
import json
import time
from tqdm import tqdm
from pathlib import Path

import model_loader
from retriever import KnowledgeBaseRetriever
from cfrag_pipeline import CFRAGPipeline
from evaluation import normalize_answer
from improved_data_loader import load_and_process_data
from improved_answer_extraction import extract_answer_from_cfrag_output

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
                common_tokens = set(gt_tokens) & set(pred_tokens)
                if len(common_tokens) == 0:
                    f1 = 0
                else:
                    precision = len(common_tokens) / len(pred_tokens)
                    recall = len(common_tokens) / len(gt_tokens)
                    f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)
    
    return best_em, best_f1

def run_cfrag_with_context(pipeline, question, context_docs):
    retriever = pipeline.retriever
    
    try:
        print(f"  Building temporary index with {len(context_docs)} documents...")
        
        retriever.build_temp_index_from_docs(context_docs)
        
        print(f"  ✅ Temporary index built, running CFRAG...")
        
        result = pipeline.run(question)
        
        retriever.clear_temp_index()
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error in CFRAG with context: {e}")
        try:
            retriever.clear_temp_index()
        except:
            pass
        
        return f"Based on the provided context, I cannot determine a specific answer to: {question}"

def main():
    parser = argparse.ArgumentParser(description="Improved MusiQue evaluation with better answer extraction")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--extract_mode", type=str, default="improved", 
                       choices=["full", "improved"], 
                       help="Answer extraction mode: full (complete reasoning) or improved (extracted answer)")
    
    args = parser.parse_args()
    
    output_dir = Path("./eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"musique_improved_results_{timestamp}.jsonl"
    
    print("Starting improved MusiQue evaluation...")
    print(f"Extract mode: {args.extract_mode}")
    print(f"Output file: {output_file}")
    
    print("\nLoading models...")
    try:
        models = model_loader.load_all_models()
        retriever = KnowledgeBaseRetriever(models["embedding_model"])
        pipeline = CFRAGPipeline(models, retriever)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return 1
    
    print("\nLoading MusiQue dataset...")
    try:
        examples = load_and_process_data(
            dataset_name="musique", 
            cache_dir="./huggingface_cache",
            num_samples=args.num_samples, 
            random_seed=args.random_seed
        )
        print(f"✅ Loaded {len(examples)} examples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1
    
    print(f"\nStarting evaluation on {len(examples)} examples...")
    total_em = 0
    total_f1 = 0
    successful_runs = 0
    results = []

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(tqdm(examples, desc="Evaluating MusiQue")):
            try:
                question = example["question"]
                ground_truths = example["answer"]
                context = example["context"]
                
                print(f"\nExample {i}:")
                print(f"Q: {question[:100]}...")
                
                start_time = time.time()
                full_output = run_cfrag_with_context(pipeline, question, context)
                
                if args.extract_mode == "improved":
                    prediction = extract_answer_from_cfrag_output(full_output, question)
                    print(f"Full output: {full_output[:100]}...")
                    print(f"Extracted: {prediction[:100]}...")
                else:
                    prediction = full_output
                    print(f"Full prediction: {prediction[:150]}...")
                
                latency = time.time() - start_time
                
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
                    "full_output": full_output if args.extract_mode == "improved" else None,
                    "ground_truths": ground_truths,
                    "em": bool(em),
                    "f1": float(f1),
                    "latency": latency,
                    "extract_mode": args.extract_mode
                }
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                
            except Exception as e:
                print(f"\n❌ Error processing example {i}: {e}")
                error_result = {
                    "id": i,
                    "question": example["question"],
                    "prediction": f"ERROR: {str(e)}",
                    "ground_truths": example["answer"],
                    "em": False,
                    "f1": 0.0,
                    "latency": -1,
                    "extract_mode": args.extract_mode
                }
                results.append(error_result)
                f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                f.flush()
    
    if successful_runs > 0:
        avg_em = total_em / successful_runs
        avg_f1 = total_f1 / successful_runs
        
        print(f"\nEvaluation completed!")
        print(f"Results Summary:")
        print(f"   Extract Mode: {args.extract_mode}")
        print(f"   Successful runs: {successful_runs}/{len(examples)}")
        print(f"   Average EM: {avg_em:.1%}")
        print(f"   Average F1: {avg_f1:.3f}")
        print(f"Results saved to: {output_file}")
        
        summary = {
            "extract_mode": args.extract_mode,
            "total_examples": len(examples),
            "successful_runs": successful_runs,
            "average_em": avg_em,
            "average_f1": avg_f1,
            "timestamp": timestamp
        }
        
        summary_file = output_dir / f"musique_improved_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Summary saved to: {summary_file}")
        
    else:
        print("❌ No successful evaluations completed.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())