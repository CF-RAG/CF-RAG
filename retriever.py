import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

import faiss
import config

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

def load_document_content(file_path: Path) -> str:
    try:
        if file_path.suffix in [".txt", ".md"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.suffix == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            logger.warning(f"Unsupported file extension {file_path.suffix}, treating as text")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return ""

def safe_encode_texts(embedding_model, texts: List[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.array([])
    
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            batch_embeddings = embedding_model.encode(
                batch_texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        except Exception as e:
            logger.error(f"Error encoding batch {i//batch_size + 1}: {e}")
            embedding_dim = embedding_model.get_sentence_embedding_dimension()
            zero_embeddings = np.zeros((len(batch_texts), embedding_dim))
            embeddings.append(zero_embeddings)
    
    return np.vstack(embeddings) if embeddings else np.array([])

class KnowledgeBaseRetriever:
    
    def __init__(self, embedding_model):
        logger.info("Initializing KnowledgeBaseRetriever")
        
        self.embedding_model = embedding_model
        self.documents = None
        self.index = None
        self.document_metadata = []
        
        config.FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        
        self.index_file_path = config.FAISS_INDEX_PATH / "faiss.index"
        self.documents_file_path = config.FAISS_INDEX_PATH / "documents.pkl"
        self.metadata_file_path = config.FAISS_INDEX_PATH / "metadata.json"
        
        if self._index_exists():
            logger.info("Found existing index, loading...")
            self._load_index()
        else:
            logger.info("No existing index found, building new index...")
            self._build_index()
        
        logger.info(f"KnowledgeBaseRetriever initialized with {len(self.documents) if self.documents else 0} documents")

    def _index_exists(self) -> bool:
        return (
            self.index_file_path.exists() and 
            self.documents_file_path.exists() and 
            self.metadata_file_path.exists()
        )

    def _load_and_split_documents(self) -> List[Document]:
        logger.info(f"Loading documents from {config.KNOWLEDGE_BASE_PATH}")
        
        if not config.KNOWLEDGE_BASE_PATH.exists():
            logger.warning(f"Knowledge base path does not exist: {config.KNOWLEDGE_BASE_PATH}")
            return []
        
        all_documents = []
        
        for file_path in config.KNOWLEDGE_BASE_PATH.rglob("*"):
            if file_path.is_file() and file_path.suffix in config.SUPPORTED_DOCUMENT_FORMATS:
                try:
                    logger.info(f"Loading file: {file_path}")
                    
                    content = load_document_content(file_path)
                    
                    if content.strip():
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source_file": str(file_path.relative_to(config.KNOWLEDGE_BASE_PATH)),
                                "file_type": file_path.suffix,
                                "file_size": file_path.stat().st_size
                            }
                        )
                        all_documents.append(doc)
                        logger.info(f"Loaded document from {file_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    continue
        
        if not all_documents:
            logger.warning("No documents were loaded from the knowledge base")
            return []
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        
        logger.info("Splitting documents into chunks...")
        
        def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
            if len(text) <= chunk_size:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = min(start + chunk_size, len(text))
                
                if end < len(text):
                    for sep in ["\n\n", "\n", ". ", " "]:
                        last_sep = text.rfind(sep, start, end)
                        if last_sep > start:
                            end = last_sep + len(sep)
                            break
                
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                
                start = max(start + 1, end - chunk_overlap)
            
            return chunks
        
        split_documents = []
        for doc in all_documents:
            content = doc.page_content
            if len(content) > config.MAX_DOCUMENT_LENGTH:
                content = content[:config.MAX_DOCUMENT_LENGTH]
                logger.warning(f"Truncated document from {doc.metadata.get('source_file', 'unknown')}")
            
            text_chunks = split_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata.get('source_file', 'unknown')}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
                    }
                )
                split_documents.append(chunk_doc)
        
        logger.info(f"Documents split into {len(split_documents)} chunks")
        return split_documents

    def _build_index(self):
        logger.info("Building FAISS index...")
        
        documents = self._load_and_split_documents()
        
        if not documents:
            logger.error("No documents to build index from")
            self.documents = []
            self.index = None
            return
        
        document_texts = [doc.page_content for doc in documents]
        self.document_metadata = [doc.metadata for doc in documents]
        self.documents = document_texts
        
        logger.info("Generating embeddings for documents...")
        embeddings = safe_encode_texts(
            self.embedding_model, 
            document_texts, 
            batch_size=32
        )
        
        if embeddings.size == 0:
            logger.error("Failed to generate embeddings")
            return
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        embedding_dim = embeddings.shape[1]
        
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype(np.float32))
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
        
        self._save_index()

    def _save_index(self):
        try:
            logger.info("Saving index to disk...")
            
            faiss.write_index(self.index, str(self.index_file_path))
            
            with open(self.documents_file_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(self.metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Index saved successfully to {config.FAISS_INDEX_PATH}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def _load_index(self):
        try:
            logger.info("Loading index from disk...")
            
            self.index = faiss.read_index(str(self.index_file_path))
            
            with open(self.documents_file_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            with open(self.metadata_file_path, 'r', encoding='utf-8') as f:
                self.document_metadata = json.load(f)
            
            logger.info(f"Index loaded successfully: {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            logger.info("Rebuilding index...")
            self._build_index()

    def build_temp_index_from_docs(self, documents: List[str]):
        if not documents:
            logger.warning("Empty document list provided, cannot build temporary index.")
            self.temp_index = None
            self.temp_documents = []
            return

        logger.info(f"Building memory index for {len(documents)} temporary documents...")
        
        try:
            embeddings = safe_encode_texts(self.embedding_model, documents)
            logger.info(f"Embeddings type: {type(embeddings)}, shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'No shape'}")
            
            if embeddings.size == 0 or len(embeddings.shape) != 2:
                logger.error(f"Failed to generate embeddings for temporary documents. Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'No shape'}")
                self.temp_index = None
                self.temp_documents = []
                return
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.temp_index = None
            self.temp_documents = []
            return

        embedding_dim = embeddings.shape[1]
        temp_index = faiss.IndexFlatIP(embedding_dim)
        faiss.normalize_L2(embeddings)
        temp_index.add(embeddings.astype(np.float32))

        self.temp_index = temp_index
        self.temp_documents = documents
        logger.info("Temporary memory index built successfully!")

    def clear_temp_index(self):
        if hasattr(self, 'temp_index') and self.temp_index is not None:
            logger.info("Clearing temporary index...")
            self.temp_index = None
            self.temp_documents = []
            logger.info("Temporary index cleared")
        else:
            logger.debug("No temporary index to clear")

    def search(self, query_text: str, top_k: int = 5) -> List[str]:
        if hasattr(self, 'temp_index') and self.temp_index is not None:
            index_to_use = self.temp_index
            docs_to_use = self.temp_documents
            source = "temporary index"
        else:
            index_to_use = self.index
            docs_to_use = self.documents
            source = "global index"

        if index_to_use is None or not docs_to_use:
            logger.error(f"Cannot perform search: {source} not initialized.")
            return []
        
        if not query_text.strip():
            logger.warning("Empty query text")
            return []
        
        try:
            logger.info(f"Searching with {source}...")
            query_embedding = self.embedding_model.encode(
                [query_text], 
                convert_to_numpy=True
            )
            
            faiss.normalize_L2(query_embedding)
            
            k = min(top_k, len(docs_to_use))
            scores, indices = index_to_use.search(query_embedding.astype(np.float32), k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(docs_to_use):
                    results.append(docs_to_use[idx])
                    logger.debug(f"Result {i+1}: score={score:.4f}, doc_idx={idx}")
            
            logger.info(f"Retrieved {len(results)} documents from {source}.")
            return results
            
        except Exception as e:
            logger.error(f"Search failed with {source}: {e}")
            return []

    def search_with_metadata(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.documents is None:
            logger.error("Index not initialized")
            return []
        
        if not query_text.strip():
            logger.warning("Empty query text")
            return []
        
        try:
            query_embedding = self.embedding_model.encode(
                [query_text], 
                convert_to_numpy=True
            )
            
            faiss.normalize_L2(query_embedding)
            
            top_k = min(top_k, len(self.documents))
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    result = {
                        "content": self.documents[idx],
                        "score": float(score),
                        "metadata": self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                    }
                    results.append(result)
            
            logger.info(f"Retrieved {len(results)} documents with metadata")
            return results
            
        except Exception as e:
            logger.error(f"Search with metadata failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "total_documents": len(self.documents) if self.documents else 0,
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "index_exists": self._index_exists(),
            "knowledge_base_path": str(config.KNOWLEDGE_BASE_PATH),
            "supported_formats": config.SUPPORTED_DOCUMENT_FORMATS
        }
        
        if self.document_metadata:
            file_types = {}
            for metadata in self.document_metadata:
                file_type = metadata.get("file_type", "unknown")
                file_types[file_type] = file_types.get(file_type, 0) + 1
            stats["file_type_distribution"] = file_types
        
        return stats

    def rebuild_index(self):
        logger.info("Rebuilding index...")
        
        for file_path in [self.index_file_path, self.documents_file_path, self.metadata_file_path]:
            if file_path.exists():
                file_path.unlink()
        
        self._build_index()

    def add_documents(self, new_documents: List[Document]):
        if not new_documents:
            logger.warning("No new documents to add")
            return
        
        logger.info(f"Adding {len(new_documents)} new documents to index")
        
        def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
            if len(text) <= chunk_size:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = min(start + chunk_size, len(text))
                
                if end < len(text):
                    for sep in ["\n\n", "\n", ". ", " "]:
                        last_sep = text.rfind(sep, start, end)
                        if last_sep > start:
                            end = last_sep + len(sep)
                            break
                
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                
                start = max(start + 1, end - chunk_overlap)
            
            return chunks
        
        split_new_docs = []
        for doc in new_documents:
            text_chunks = split_text(doc.page_content, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for i, chunk_text in enumerate(text_chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "chunk_index": i}
                )
                split_new_docs.append(chunk_doc)
        
        new_texts = [doc.page_content for doc in split_new_docs]
        new_metadata = [doc.metadata for doc in split_new_docs]
        
        new_embeddings = safe_encode_texts(self.embedding_model, new_texts)
        
        if new_embeddings.size > 0:
            faiss.normalize_L2(new_embeddings)
            
            self.index.add(new_embeddings.astype(np.float32))
            
            self.documents.extend(new_texts)
            self.document_metadata.extend(new_metadata)
            
            self._save_index()
            
            logger.info(f"Successfully added {len(split_new_docs)} document chunks")
        else:
            logger.error("Failed to generate embeddings for new documents")

def create_retriever(embedding_model) -> KnowledgeBaseRetriever:
    return KnowledgeBaseRetriever(embedding_model)

if __name__ == "__main__":
    print("KnowledgeBaseRetriever module loaded successfully")