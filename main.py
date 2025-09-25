import sys
import time
import logging
import traceback
from datetime import datetime
from typing import Optional

import config
import model_loader
from retriever import KnowledgeBaseRetriever
from cfrag_pipeline import CFRAGPipeline

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    print(f"Startup time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {config.PROJECT_ROOT}")
    print(f"Configuration: config.py")
    print("=" * 80)

def print_system_info():
    print("\nSystem Configuration:")
    print("─" * 50)
    print(f"LLM Model: {config.LLM_MODEL_ID}")
    print(f"Reranker Model: {config.RERANKER_MODEL_ID}")
    print(f"Embedding Model: {config.EMBEDDING_MODEL_ID}")
    print(f"Device: {config.LLM_DEVICE}")
    print(f"Knowledge Base: {config.KNOWLEDGE_BASE_PATH}")
    print(f"Index Path: {config.FAISS_INDEX_PATH}")
    print(f"Retrieval Top-K: {config.RETRIEVAL_TOP_K}")
    print(f"Reranker Top-K: {config.RERANKER_TOP_K}")
    print(f"Reranker Weight: {config.RERANKER_WEIGHT}")
    print("─" * 50)

def print_loading_animation(text: str, duration: float = 2.0):
    import time
    
    animation_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start_time = time.time()
    
    while time.time() - start_time < duration:
        for char in animation_chars:
            if time.time() - start_time >= duration:
                break
            print(f"\r{char} {text}", end="", flush=True)
            time.sleep(0.1)
    
    print(f"\r✅ {text} - Complete!")

def initialize_system() -> tuple[dict, KnowledgeBaseRetriever, CFRAGPipeline]:
    try:
        print("\nInitializing CF-RAG system...")
        print("─" * 50)
        
        print("\nStep 1/3: Loading models")
        start_time = time.time()
        
        models = model_loader.load_all_models()
        
        load_time = time.time() - start_time
        print(f"✅ Models loaded (time: {load_time:.1f}s)")
        
        print("\nStep 2/3: Initializing knowledge base retriever")
        start_time = time.time()
        
        retriever = KnowledgeBaseRetriever(embedding_model=models["embedding_model"])
        
        retriever_time = time.time() - start_time
        print(f"✅ Retriever initialized (time: {retriever_time:.1f}s)")
        
        stats = retriever.get_stats()
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Index size: {stats['index_size']}")
        print(f"   Vector dimension: {stats['embedding_dimension']}")
        if 'file_type_distribution' in stats:
            print("   File types:")
            for file_type, count in stats['file_type_distribution'].items():
                print(f"      {file_type}: {count}")
        
        print("\nStep 3/3: Initializing CF-RAG pipeline")
        start_time = time.time()
        
        pipeline = CFRAGPipeline(models=models, retriever=retriever)
        
        pipeline_time = time.time() - start_time
        print(f"✅ Pipeline initialized (time: {pipeline_time:.1f}s)")
        
        pipeline_stats = pipeline.get_pipeline_stats()
        print(f"   LLM: {pipeline_stats['model_info']['llm_model']}")
        print(f"   Reranker: {pipeline_stats['model_info']['reranker_model']}")
        print(f"   Device: {pipeline_stats['model_info']['device']}")
        
        total_init_time = load_time + retriever_time + pipeline_time
        print("\n" + "─" * 50)
        print(f"System initialization complete! Total time: {total_init_time:.1f}s")
        print("─" * 50)
        
        return models, retriever, pipeline
        
    except KeyboardInterrupt:
        print("\n\nUser interrupted initialization")
        logger.info("User interrupted initialization")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\nSystem initialization failed: {e}")
        logger.error(f"System initialization failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        print("\nPossible solutions:")
        print("1. Check GPU memory availability")
        print("2. Verify model files are properly downloaded")
        print("3. Check knowledge base directory exists")
        print("4. Review log file for detailed error information")
        print(f"   Log location: {config.LOG_FILE}")
        
        sys.exit(1)

def print_usage_instructions():
    instructions = """
Usage Instructions:
─────────────────────────────────────────────────────────────────────────────
• Enter your question directly, the system will provide detailed answers using CF-RAG
• The system automatically generates counterfactual queries to enhance robustness
• Type 'help' to see more commands
• Type 'stats' to view system statistics
• Type 'exit' or 'quit' to exit the system

CF-RAG Features:
• Counterfactual Reasoning: Explains why other answers are incorrect
• Decisive Evidence: Based on high-quality evidence that supports and refutes alternatives
• Explanatory Answers: Provides detailed causal reasoning process
─────────────────────────────────────────────────────────────────────────────
    """
    print(instructions)

def handle_special_commands(user_input: str, retriever: KnowledgeBaseRetriever, 
                          pipeline: CFRAGPipeline) -> bool:
    command = user_input.lower().strip()
    
    if command == 'help':
        print_usage_instructions()
        return True
    
    elif command == 'stats':
        print("\nSystem Statistics:")
        print("─" * 50)
        
        retriever_stats = retriever.get_stats()
        print("Retriever Stats:")
        print(f"   Documents: {retriever_stats['total_documents']}")
        print(f"   Index size: {retriever_stats['index_size']}")
        print(f"   Vector dimension: {retriever_stats['embedding_dimension']}")
        
        pipeline_stats = pipeline.get_pipeline_stats()
        print("\nPipeline Configuration:")
        print(f"   Max tokens: {pipeline_stats['config']['max_new_tokens']}")
        print(f"   Temperature: {pipeline_stats['config']['temperature']}")
        print(f"   Retrieval Top-K: {pipeline_stats['config']['retrieval_top_k']}")
        print(f"   Reranker Top-K: {pipeline_stats['config']['reranker_top_k']}")
        print(f"   Reranker Weight: {pipeline_stats['config']['reranker_weight']}")
        
        print("─" * 50)
        return True
    
    elif command in ['clear', 'cls']:
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        print_banner()
        return True
    
    elif command == 'rebuild':
        print("\nRebuilding knowledge base index...")
        try:
            retriever.rebuild_index()
            print("✅ Index rebuilt successfully!")
        except Exception as e:
            print(f"❌ Index rebuild failed: {e}")
        return True
    
    return False

def format_answer(answer: str, query: str, start_time: float) -> str:
    processing_time = time.time() - start_time
    
    formatted_output = f"""
CF-RAG Response
─────────────────────────────────────────────────────────────────────────────
Question: {query}
Processing time: {processing_time:.2f}s

Answer:
{answer}

Process: Counterfactual Generation → Synergetic Retrieval → Causal Reranking → Explanatory Generation
─────────────────────────────────────────────────────────────────────────────
    """
    
    return formatted_output

def main():
    try:
        print_banner()
        
        print_system_info()
        
        models, retriever, pipeline = initialize_system()
        
        print_usage_instructions()
        
        logger.info("CF-RAG system started successfully")
        
        print("\nSystem ready. Please enter your question:")
        print("=" * 80)
        
        session_count = 0
        
        while True:
            try:
                user_query = input(f"\n[{session_count + 1}] Your question: ").strip()
                
                if user_query.lower() in ["exit", "quit"]:
                    print("\nThank you for using CF-RAG system!")
                    logger.info("User exited the system")
                    break
                
                if not user_query:
                    print("⚠️ Please enter a valid question")
                    continue
                
                if handle_special_commands(user_query, retriever, pipeline):
                    continue
                
                start_time = time.time()
                session_count += 1
                
                print(f"\nProcessing your question...")
                logger.info(f"Processing query {session_count}: {user_query}")
                
                try:
                    final_answer = pipeline.run(user_query)
                    
                    formatted_answer = format_answer(final_answer, user_query, start_time)
                    print(formatted_answer)
                    
                    logger.info(f"Query {session_count} completed successfully")
                    
                except Exception as e:
                    error_msg = f"Error processing query: {e}"
                    print(f"\n❌ {error_msg}")
                    logger.error(f"Query processing failed: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    print("\nSuggestions:")
                    print("• Try rephrasing your question")
                    print("• Check if the question is too complex or ambiguous")
                    print("• Type 'stats' to check system status")
                
                print("\n" + "=" * 80)
                
            except KeyboardInterrupt:
                print("\n\nInterrupt signal detected")
                confirm = input("Do you want to exit? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    print("Goodbye!")
                    break
                else:
                    print("Continuing...")
                    continue
            
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                logger.error(f"Unexpected error in main loop: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                print("System will continue running, please try again...")
                continue
    
    except Exception as e:
        print(f"\n💥 Critical system error: {e}")
        logger.critical(f"Critical system error: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    finally:
        logger.info("CF-RAG system shutdown")

def test_mode():
    print("CF-RAG Test Mode")
    print("=" * 50)
    
    try:
        models, retriever, pipeline = initialize_system()
        
        test_queries = config.TEST_QUERIES
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}/{len(test_queries)}: {query}")
            print("─" * 50)
            
            start_time = time.time()
            try:
                answer = pipeline.run(query)
                processing_time = time.time() - start_time
                
                print(f"✅ Answer (time: {processing_time:.2f}s):")
                print(answer)
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
            
            print("─" * 50)
        
        print("\nTesting complete!")
        
    except Exception as e:
        print(f"❌ Test mode failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_mode()
        elif sys.argv[1] == "--help":
            print("""
CF-RAG Command Line Arguments:
  --test    Run test mode
  --help    Show this help message
  
Default: Interactive mode
            """)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help to see available arguments")
            sys.exit(1)
    else:
        main()