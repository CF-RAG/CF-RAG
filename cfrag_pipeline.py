import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import re
from collections import defaultdict
from sklearn.cluster import KMeans

import config
from retriever import KnowledgeBaseRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CFRAGPipeline:
    
    def __init__(self, models: Dict[str, Any], retriever: KnowledgeBaseRetriever):
        self.llm_model = models["llm_model"]
        self.llm_tokenizer = models["llm_tokenizer"]
        self.reranker = models["reranker_model"]
        self.retriever = retriever
        
        self.cf_generation_config = {
            "max_new_tokens": 128,
            "temperature": config.COUNTERFACTUAL_TEMPERATURE,
            "do_sample": True,
            "pad_token_id": self.llm_tokenizer.eos_token_id,
        }
        if config.EXPLANATORY_TEMPERATURE > 0:
            self.exp_generation_config = {
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "temperature": config.EXPLANATORY_TEMPERATURE,
                "do_sample": True,
                "pad_token_id": self.llm_tokenizer.eos_token_id,
            }
        else:
            self.exp_generation_config = {
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "do_sample": False,
                "pad_token_id": self.llm_tokenizer.eos_token_id,
            }
        
        self.draft_generation_config = {
            "max_new_tokens": 256,
            "temperature": config.COUNTERFACTUAL_TEMPERATURE,
            "do_sample": True,
            "pad_token_id": self.llm_tokenizer.eos_token_id,
        }
        
        logger.info("CF-RAG Pipeline initialized successfully")

    def _generate_counterfactual_query(self, original_query: str) -> List[str]:
        logger.info(f"Generating counterfactual queries for: {original_query}")
        
        prompt = config.COUNTERFACTUAL_PROMPT_TEMPLATE.format(
            num_queries=config.MAX_COUNTERFACTUAL_QUERIES,
            original_query=original_query
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates counterfactual queries."},
            {"role": "user", "content": prompt}
        ]
        
        input_text = self.llm_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.llm_tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=config.MAX_INPUT_LENGTH
        ).to(self.llm_model.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                **self.cf_generation_config
            )
        
        generated_text = self.llm_tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        ).strip()
        
        cf_queries = self._parse_counterfactual_queries(generated_text)
        
        logger.info(f"Generated {len(cf_queries)} counterfactual queries: {cf_queries}")
        return cf_queries

    def _parse_counterfactual_queries(self, generated_text: str) -> List[str]:
        queries = []
        
        numbered_pattern = r'^\d+\.\s*(.+)$'
        for line in generated_text.split('\n'):
            line = line.strip()
            match = re.match(numbered_pattern, line)
            if match:
                queries.append(match.group(1).strip())
        
        if not queries:
            dash_pattern = r'^-\s*(.+)$'
            for line in generated_text.split('\n'):
                line = line.strip()
                match = re.match(dash_pattern, line)
                if match:
                    queries.append(match.group(1).strip())
        
        if not queries:
            lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
            for line in lines:
                if len(line) > 10 and '?' in line:
                    queries.append(line)
        
        if not queries and generated_text.strip():
            queries = [generated_text.strip()]
        
        return queries[:config.MAX_COUNTERFACTUAL_QUERIES]

    def _synergetic_retrieval(self, original_query: str, cf_queries: List[str]) -> List[str]:
        logger.info("Executing synergetic retrieval")
        
        all_documents = []
        seen_docs = set()
        
        logger.info(f"Retrieving for original query: {original_query}")
        orig_docs = self.retriever.search(original_query, top_k=config.RETRIEVAL_TOP_K)
        
        for doc in orig_docs:
            doc_text = doc if isinstance(doc, str) else doc.get('text', str(doc))
            if doc_text not in seen_docs:
                all_documents.append(doc_text)
                seen_docs.add(doc_text)
        
        for i, cf_query in enumerate(cf_queries):
            logger.info(f"Retrieving for counterfactual query {i+1}: {cf_query}")
            cf_docs = self.retriever.search(cf_query, top_k=config.RETRIEVAL_TOP_K)
            
            for doc in cf_docs:
                doc_text = doc if isinstance(doc, str) else doc.get('text', str(doc))
                if doc_text not in seen_docs:
                    all_documents.append(doc_text)
                    seen_docs.add(doc_text)
        
        logger.info(f"Synergetic retrieval completed. Total unique documents: {len(all_documents)}")
        return all_documents

    def _cluster_and_sample_evidence(self, documents: List[str], original_query: str = "") -> List[List[str]]:
        logger.info(f"Clustering and sampling evidence from {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents to cluster")
            return []
        
        if len(documents) < config.NUM_CLUSTERS:
            logger.warning(f"Document count ({len(documents)}) < NUM_CLUSTERS ({config.NUM_CLUSTERS}), using all documents")
            return [documents] * config.NUM_DRAFTS
        
        try:
            logger.info("Encoding documents with embedding model")
            embeddings = []
            for doc in documents:
                embedding = self.retriever.embedding_model.encode([doc], convert_to_tensor=False)[0]
                embeddings.append(embedding)
            
            query_scores = []
            if original_query:
                query_embedding = self.retriever.embedding_model.encode([original_query], convert_to_tensor=False)[0]
                from sklearn.metrics.pairwise import cosine_similarity
                for embedding in embeddings:
                    similarity = cosine_similarity([embedding], [query_embedding])[0][0]
                    query_scores.append(similarity)
            else:
                query_scores = [1.0] * len(embeddings)
            
            embeddings = np.array(embeddings)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            
            if config.USE_SEMANTIC_CLUSTERING:
                logger.info(f"Performing semantic-aware clustering with {config.NUM_CLUSTERS} clusters")
                cluster_labels = self._semantic_clustering(embeddings, documents, original_query)
            else:
                logger.info(f"Performing KMeans clustering with {config.NUM_CLUSTERS} clusters")
                kmeans = KMeans(n_clusters=config.NUM_CLUSTERS, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
            
            clusters = {}
            for doc_idx, cluster_id in enumerate(cluster_labels):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(documents[doc_idx])
            
            logger.info(f"Created {len(clusters)} clusters with sizes: {[len(cluster) for cluster in clusters.values()]}")
            
            evidence_subsets = []
            
            top_relevant_docs = []
            if config.INCLUDE_TOP_EVIDENCE:
                relevance_pairs = [(original_query, doc) for doc in documents]
                relevance_scores = self.retriever.embedding_model.encode(
                    [pair[0] for pair in relevance_pairs], convert_to_tensor=False
                )
                doc_scores = list(zip(documents, range(len(documents))))
                top_relevant_docs = documents[:min(2, len(documents))]
                logger.info(f"Selected {len(top_relevant_docs)} top relevant documents to include in all drafts")
            
            for draft_idx in range(config.NUM_DRAFTS):
                evidence_subset = []
                
                if original_query:
                    doc_relevance = list(zip(documents, query_scores, range(len(documents))))
                    doc_relevance.sort(key=lambda x: x[1], reverse=True)
                    
                    core_evidence = doc_relevance[draft_idx % min(len(doc_relevance), 2)]
                    evidence_subset.append(core_evidence[0])
                    logger.debug(f"Draft {draft_idx+1}: Added core evidence with relevance {core_evidence[1]:.3f}")
                
                if config.INCLUDE_TOP_EVIDENCE:
                    evidence_subset.extend(top_relevant_docs)
                
                for cluster_id in sorted(clusters.keys()):
                    cluster_docs = clusters[cluster_id]
                    available_docs = [doc for doc in cluster_docs if doc not in evidence_subset]
                    if available_docs:
                        if draft_idx < len(available_docs):
                            selected_doc = available_docs[draft_idx % len(available_docs)]
                        else:
                            selected_doc = np.random.choice(available_docs)
                        evidence_subset.append(selected_doc)
                
                while len(evidence_subset) < config.MIN_DOCS_PER_DRAFT and len(evidence_subset) < len(documents):
                    remaining_docs = [doc for doc in documents if doc not in evidence_subset]
                    if remaining_docs:
                        evidence_subset.append(np.random.choice(remaining_docs))
                    else:
                        break
                
                evidence_subsets.append(evidence_subset)
                logger.debug(f"Created evidence subset {draft_idx + 1} with {len(evidence_subset)} documents")
            
            logger.info(f"Successfully created {len(evidence_subsets)} evidence subsets")
            return evidence_subsets
            
        except Exception as e:
            logger.error(f"Error in clustering and sampling: {e}")
            logger.info("Falling back to random document splitting")
            evidence_subsets = []
            docs_per_subset = max(1, len(documents) // config.NUM_DRAFTS)
            
            for i in range(config.NUM_DRAFTS):
                start_idx = (i * docs_per_subset) % len(documents)
                end_idx = min(start_idx + docs_per_subset, len(documents))
                if start_idx >= len(documents):
                    subset = documents[:docs_per_subset]
                else:
                    subset = documents[start_idx:end_idx]
                    if len(subset) < docs_per_subset and end_idx < len(documents):
                        remaining = docs_per_subset - len(subset)
                        subset.extend(documents[:remaining])
                evidence_subsets.append(subset)
            
            return evidence_subsets

    def _semantic_clustering(self, embeddings: np.ndarray, documents: List[str], original_query: str) -> np.ndarray:
        try:
            query_embedding = self.retriever.embedding_model.encode([original_query], convert_to_tensor=False)[0]
            
            from sklearn.metrics.pairwise import cosine_similarity
            query_similarities = cosine_similarity(embeddings, query_embedding.reshape(1, -1)).flatten()
            
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            enhanced_features = np.column_stack([
                embeddings,
                query_similarities.reshape(-1, 1),
                (query_similarities ** 2).reshape(-1, 1)
            ])
            
            scaler = StandardScaler()
            enhanced_features = scaler.fit_transform(enhanced_features)
            
            kmeans = KMeans(n_clusters=config.NUM_CLUSTERS, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(enhanced_features)
            
            top_indices = np.argsort(query_similarities)[-config.NUM_CLUSTERS:]
            
            for i, top_idx in enumerate(top_indices):
                target_cluster = i % config.NUM_CLUSTERS
                cluster_labels[top_idx] = target_cluster
                
            logger.info(f"Semantic clustering completed with query-aware distribution")
            logger.debug(f"Cluster sizes: {np.bincount(cluster_labels)}")
            logger.debug(f"Top documents distributed across clusters: {[cluster_labels[idx] for idx in top_indices]}")
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Error in semantic clustering: {e}")
            logger.info("Falling back to standard K-Means clustering")
            kmeans = KMeans(n_clusters=config.NUM_CLUSTERS, random_state=42, n_init=10)
            return kmeans.fit_predict(embeddings)

    def _simplified_evidence_selection(self, original_query: str, cf_queries: List[str], documents: List[str]) -> List[str]:
        logger.info(f"Simplified evidence selection from {len(documents)} documents")
        
        if not documents:
            return []
        
        try:
            relevance_pairs = [(original_query, doc) for doc in documents]
            all_relevance_scores = self.reranker.predict(relevance_pairs)
            
            qualified_docs_with_scores = []
            for i, score in enumerate(all_relevance_scores):
                if score >= config.RELEVANCE_THRESHOLD:
                    qualified_docs_with_scores.append((documents[i], score))
            
            logger.info(f"Filtered {len(qualified_docs_with_scores)} documents with relevance score >= {config.RELEVANCE_THRESHOLD}")
            
            if not qualified_docs_with_scores:
                logger.warning("No documents passed the relevance threshold. Using top relevance ranking.")
                sorted_docs = sorted(zip(documents, all_relevance_scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, score in sorted_docs[:config.RERANKER_TOP_K]]

            final_scores = []
            qualified_docs = [item[0] for item in qualified_docs_with_scores]
            relevance_scores = [item[1] for item in qualified_docs_with_scores]

            if cf_queries:
                cf_pairs = [(cf_query, doc) for doc in qualified_docs for cf_query in cf_queries]
                all_cf_scores = self.reranker.predict(cf_pairs)
                
                for i, doc in enumerate(qualified_docs):
                    doc_cf_scores = all_cf_scores[i*len(cf_queries) : (i+1)*len(cf_queries)]
                    avg_cf_relevance = np.mean(doc_cf_scores)
                    refutation_score = -avg_cf_relevance
                    
                    relevance_score = relevance_scores[i]
                    final_score = (config.RERANKER_WEIGHT * relevance_score) + ((1 - config.RERANKER_WEIGHT) * refutation_score)
                    final_scores.append((doc, final_score))
            else:
                final_scores = list(zip(qualified_docs, relevance_scores))

            final_scores.sort(key=lambda x: x[1], reverse=True)
            
            top_documents = [item[0] for item in final_scores[:config.RERANKER_TOP_K]]
            
            logger.info(f"Simplified evidence selection completed. Selected {len(top_documents)} documents.")
            return top_documents

        except Exception as e:
            logger.error(f"Error in simplified evidence selection: {e}")
            return []

    def _generate_drafts_parallel(self, original_query: str, evidence_subsets: List[List[str]]) -> List[Dict[str, Any]]:
        logger.info(f"Generating {len(evidence_subsets)} parallel drafts")
        
        if not evidence_subsets:
            logger.warning("No evidence subsets provided for draft generation")
            return []
        
        drafts = []
        
        for i, evidence_subset in enumerate(evidence_subsets):
            logger.info(f"Generating draft {i + 1}/{len(evidence_subsets)}")
            
            try:
                evidence_text = "\n\n".join([
                    f"Evidence {j+1}: {doc}" 
                    for j, doc in enumerate(evidence_subset)
                ])
                
                prompt = config.DRAFT_PROMPT_TEMPLATE.format(
                    original_query=original_query,
                    evidence_documents=evidence_text
                )
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that provides concise answers with brief rationales based on evidence."},
                    {"role": "user", "content": prompt}
                ]
                
                input_text = self.llm_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = self.llm_tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=config.MAX_INPUT_LENGTH
                ).to(self.llm_model.device)
                
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        **inputs,
                        **self.draft_generation_config
                    )
                
                generated_text = self.llm_tokenizer.decode(
                    outputs[0][len(inputs["input_ids"][0]):], 
                    skip_special_tokens=True
                ).strip()
                
                draft_answer, rationale = self._parse_draft_response(generated_text)
                
                draft = {
                    "draft_answer": draft_answer,
                    "rationale": rationale,
                    "evidence_subset": evidence_subset,
                    "draft_id": i + 1
                }
                
                drafts.append(draft)
                logger.debug(f"Draft {i + 1} generated successfully")
                
            except Exception as e:
                logger.error(f"Error generating draft {i + 1}: {e}")
                error_draft = {
                    "draft_answer": f"Error generating answer: {str(e)}",
                    "rationale": "Generation failed due to technical error",
                    "evidence_subset": evidence_subset,
                    "draft_id": i + 1
                }
                drafts.append(error_draft)
        
        logger.info(f"Successfully generated {len(drafts)} drafts")
        return drafts

    def _parse_draft_response(self, generated_text: str) -> Tuple[str, str]:
        answer_pattern = r'(?:Answer|答案)[:：]\s*(.+?)(?=(?:Rationale|Reasoning|推理|理由)[:：]|$)'
        rationale_pattern = r'(?:Rationale|Reasoning|推理|理由)[:：]\s*(.+)'
        
        answer_match = re.search(answer_pattern, generated_text, re.IGNORECASE | re.DOTALL)
        rationale_match = re.search(rationale_pattern, generated_text, re.IGNORECASE | re.DOTALL)
        
        if answer_match and rationale_match:
            answer = answer_match.group(1).strip()
            rationale = rationale_match.group(1).strip()
            return answer, rationale
        
        paragraphs = [p.strip() for p in generated_text.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            answer = paragraphs[0]
            rationale = ' '.join(paragraphs[1:])
            return answer, rationale
        
        sentences = [s.strip() for s in generated_text.split('.') if s.strip()]
        if len(sentences) >= 2:
            mid_point = len(sentences) // 2
            answer = '. '.join(sentences[:mid_point]) + '.'
            rationale = '. '.join(sentences[mid_point:]) + '.'
            return answer, rationale
        
        return generated_text.strip(), "No specific rationale provided."

    def _verify_drafts(self, original_query: str, cf_queries: List[str], drafts: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"Verifying {len(drafts)} drafts using counterfactual enhancement")
        
        if not drafts:
            logger.warning("No drafts to verify")
            return {"draft_answer": "", "evidence_subset": [], "rationale": ""}
        
        if len(drafts) == 1:
            logger.info("Only one draft available, returning it directly")
            return drafts[0]
        
        draft_scores = []
        
        for i, draft in enumerate(drafts):
            logger.info(f"Evaluating draft {i + 1}/{len(drafts)}")
            
            try:
                evidence_subset = draft["evidence_subset"]
                draft_answer = draft["draft_answer"]
                
                score_causal = self._calculate_causal_score(original_query, cf_queries, evidence_subset)
                
                score_consistency = self._calculate_consistency_score(original_query, draft_answer, evidence_subset)
                
                score_completeness = self._calculate_completeness_score(original_query, draft_answer)
                
                final_score = (0.4 * score_consistency + 
                              0.4 * score_causal + 
                              0.2 * score_completeness)
                
                draft_scores.append({
                    "draft_id": i + 1,
                    "draft": draft,
                    "score_causal": score_causal,
                    "score_consistency": score_consistency,
                    "score_completeness": score_completeness,
                    "final_score": final_score
                })
                
                logger.info(f"Draft {i + 1} scores - Causal: {score_causal:.3f}, "
                           f"Consistency: {score_consistency:.3f}, Completeness: {score_completeness:.3f}, Final: {final_score:.3f}")
                logger.debug(f"Draft {i + 1} answer preview: {draft_answer[:100]}...")
                
            except Exception as e:
                logger.error(f"Error evaluating draft {i + 1}: {e}")
                draft_scores.append({
                    "draft_id": i + 1,
                    "draft": draft,
                    "score_causal": 0.0,
                    "score_consistency": 0.0,
                    "final_score": 0.0
                })
        
        draft_scores.sort(key=lambda x: x["final_score"], reverse=True)
        best_draft_info = draft_scores[0]
        
        logger.info(f"Selected draft {best_draft_info['draft_id']} as the best "
                   f"(Final score: {best_draft_info['final_score']:.3f})")
        
        best_draft = best_draft_info["draft"]
        best_draft["verification_scores"] = {
            "causal_score": best_draft_info["score_causal"],
            "consistency_score": best_draft_info["score_consistency"], 
            "final_score": best_draft_info["final_score"]
        }
        return best_draft

    def _calculate_causal_score(self, original_query: str, cf_queries: List[str], evidence_docs: List[str]) -> float:
        if not evidence_docs:
            return 0.0
        
        try:
            relevance_pairs = [(original_query, doc) for doc in evidence_docs]
            relevance_scores = self.reranker.predict(relevance_pairs)
            avg_relevance = np.mean(relevance_scores)
            
            avg_refutation = 0.0
            if cf_queries:
                cf_pairs = [(cf_query, doc) for doc in evidence_docs for cf_query in cf_queries]
                cf_scores = self.reranker.predict(cf_pairs)
                
                doc_cf_scores = []
                for i in range(len(evidence_docs)):
                    doc_scores = cf_scores[i*len(cf_queries):(i+1)*len(cf_queries)]
                    doc_cf_scores.append(np.mean(doc_scores))
                
                avg_refutation = -np.mean(doc_cf_scores)
            
            causal_score = (config.RERANKER_WEIGHT * avg_relevance + 
                           (1 - config.RERANKER_WEIGHT) * avg_refutation)
            
            return float(causal_score)
            
        except Exception as e:
            logger.error(f"Error calculating causal score: {e}")
            return 0.0

    def _calculate_consistency_score(self, original_query: str, draft_answer: str, evidence_docs: List[str]) -> float:
        if not evidence_docs or not draft_answer:
            return 0.0
        
        if config.ENABLE_MULTI_ASPECT_CONSISTENCY:
            return self._multi_aspect_consistency_evaluation(original_query, draft_answer, evidence_docs)
        else:
            return self._single_aspect_consistency_evaluation(original_query, draft_answer, evidence_docs)

    def _multi_aspect_consistency_evaluation(self, original_query: str, draft_answer: str, evidence_docs: List[str]) -> float:
        evidence_text = "\n".join([f"Evidence {i+1}: {doc}" for i, doc in enumerate(evidence_docs)])
        aspect_scores = []
        
        for aspect in config.CONSISTENCY_ASPECTS:
            try:
                score = self._evaluate_consistency_aspect(original_query, draft_answer, evidence_text, aspect)
                aspect_scores.append(score)
                logger.debug(f"Consistency aspect '{aspect}': {score:.3f}")
            except Exception as e:
                logger.warning(f"Failed to evaluate aspect '{aspect}': {e}")
                aspect_scores.append(0.5)
        
        weights = [0.4, 0.35, 0.25]
        if len(aspect_scores) == len(weights):
            final_score = sum(score * weight for score, weight in zip(aspect_scores, weights))
        else:
            final_score = np.mean(aspect_scores)
        
        logger.info(f"Multi-aspect consistency scores: {aspect_scores}, Final: {final_score:.3f}")
        return final_score

    def _evaluate_consistency_aspect(self, original_query: str, draft_answer: str, evidence_text: str, aspect: str) -> float:
        aspect_prompts = {
            "factual_support": f"""Evaluate the factual support for the answer based on evidence.

Question: {original_query}
Answer: {draft_answer}
Evidence: {evidence_text}

Rate how well the evidence factually supports the answer (Strong/Moderate/Weak/None):""",
            
            "logical_coherence": f"""Evaluate the logical coherence between the answer and evidence.

Question: {original_query}
Answer: {draft_answer}
Evidence: {evidence_text}

Rate the logical consistency and reasoning flow (Strong/Moderate/Weak/None):""",
            
            "completeness": f"""Evaluate how completely the answer addresses the question based on available evidence.

Question: {original_query}
Answer: {draft_answer}
Evidence: {evidence_text}

Rate the completeness of the answer (Strong/Moderate/Weak/None):"""
        }
        
        prompt = aspect_prompts.get(aspect, aspect_prompts["factual_support"])
        
        return self._evaluate_with_llm(prompt)

    def _evaluate_with_llm(self, prompt: str) -> float:
        try:
            messages = [
                {"role": "system", "content": "You are an objective evaluator. Respond with only one word: Strong, Moderate, Weak, or None."},
                {"role": "user", "content": prompt}
            ]
            
            input_text = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.llm_tokenizer(
                input_text, return_tensors="pt", padding=True, 
                truncation=True, max_length=config.MAX_INPUT_LENGTH
            ).to(self.llm_model.device)
            
            with torch.no_grad():
                from transformers import GenerationConfig
                eval_config = GenerationConfig(
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )
                outputs = self.llm_model.generate(**inputs, generation_config=eval_config)
            
            result = self.llm_tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
            ).strip().lower()
            
            if "strong" in result:
                return 1.0
            elif "moderate" in result:
                return 0.7
            elif "weak" in result:
                return 0.4
            elif "none" in result:
                return 0.0
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return 0.5

    def _single_aspect_consistency_evaluation(self, original_query: str, draft_answer: str, evidence_docs: List[str]) -> float:
        evidence_text = "\n".join([f"Evidence {i+1}: {doc}" for i, doc in enumerate(evidence_docs)])
        
        consistency_prompt = f"""You are evaluating the consistency between an answer and evidence. Consider whether the evidence provides reasonable support for the answer, even if not perfect.

Question: {original_query}
Answer: {draft_answer}

Evidence:
{evidence_text}

Rate the support level (respond with only one word):
- "Strong" if evidence strongly supports the answer
- "Moderate" if evidence provides reasonable support  
- "Weak" if evidence provides minimal support
- "None" if evidence contradicts or doesn't support the answer

Support level:"""
        
        return self._evaluate_with_llm(consistency_prompt)
    
    def _calculate_completeness_score(self, original_query: str, draft_answer: str) -> float:
        try:
            answer_length = len(draft_answer.split())
            
            if answer_length < 5:
                length_score = 0.2
            elif answer_length < 15:
                length_score = 0.6
            elif answer_length < 50:
                length_score = 1.0
            elif answer_length < 100:
                length_score = 0.8
            else:
                length_score = 0.6
            
            quality_indicators = [
                "based on", "according to", "evidence", "because", "therefore",
                "however", "although", "specifically", "particularly", "indeed"
            ]
            
            quality_score = 0.0
            answer_lower = draft_answer.lower()
            for indicator in quality_indicators:
                if indicator in answer_lower:
                    quality_score += 0.1
            
            quality_score = min(quality_score, 1.0)
            
            negative_indicators = [
                "could not find", "unable to", "not mentioned", "no information",
                "apologize", "cannot answer", "insufficient"
            ]
            
            penalty = 0.0
            for indicator in negative_indicators:
                if indicator in answer_lower:
                    penalty += 0.3
            
            completeness_score = max(0.0, (0.6 * length_score + 0.4 * quality_score) - penalty)
            
            logger.debug(f"Completeness score: {completeness_score:.3f} (length: {length_score:.3f}, quality: {quality_score:.3f}, penalty: {penalty:.3f})")
            return completeness_score
            
        except Exception as e:
            logger.error(f"Error calculating completeness score: {e}")
            return 0.5

    def _generate_explanatory_answer(self, original_query: str, evidence_docs: List[str]) -> str:
        logger.info("Generating explanatory answer")
        
        evidence_text = "\n\n".join([
            f"Evidence {i+1}: {doc}" 
            for i, doc in enumerate(evidence_docs[:config.MAX_EVIDENCE_DOCS])
        ])
        
        prompt = config.EXPLANATORY_PROMPT_TEMPLATE.format(
            original_query=original_query,
            evidence_documents=evidence_text
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides comprehensive answers with explanations based on evidence."},
            {"role": "user", "content": prompt}
        ]
        
        input_text = self.llm_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.llm_tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=config.MAX_INPUT_LENGTH
        ).to(self.llm_model.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                **self.exp_generation_config
            )
        
        final_answer = self.llm_tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        ).strip()
        
        logger.info("Explanatory answer generated successfully")
        return final_answer
    
    def _generate_explanatory_answer_with_reasoning(self, query: str, best_draft: Dict[str, Any]) -> str:
        logger.info("Generating explanatory answer with reasoning process")
        
        try:
            evidence_subset = best_draft.get("evidence_subset", [])
            draft_answer = best_draft.get("draft_answer", "")
            draft_rationale = best_draft.get("rationale", "")
            verification_scores = best_draft.get("verification_scores", {})
            
            evidence_text = "\n\n".join([
                f"Evidence {i+1}: {doc}" 
                for i, doc in enumerate(evidence_subset)
            ])
            
            prompt = config.EXPLANATORY_PROMPT_TEMPLATE.format(
                original_query=query,
                evidence_documents=evidence_text
            )
            
            enhanced_prompt = f"""{prompt}

**Reference Draft Analysis:**
Previous Analysis: {draft_rationale}
Draft Answer: {draft_answer}
Verification Scores: Causal={verification_scores.get('causal_score', 0):.3f}, Consistency={verification_scores.get('consistency_score', 0):.3f}

Please provide a comprehensive response that improves upon this draft with detailed reasoning and evidence citations."""

            messages = [
                {"role": "system", "content": "You are an expert AI assistant that provides comprehensive, well-reasoned answers with detailed evidence analysis and step-by-step reasoning."},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            input_text = self.llm_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.llm_tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=config.MAX_INPUT_LENGTH
            ).to(self.llm_model.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    **self.exp_generation_config
                )
            
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.llm_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            logger.info("Explanatory answer with reasoning generated successfully")
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error in explanatory answer generation with reasoning: {e}")
            fallback = f"""Based on the evidence analysis:

**Reasoning:** {best_draft.get('rationale', 'No detailed reasoning available.')}

**Answer:** {best_draft.get('draft_answer', 'Unable to provide answer.')}

**Evidence Sources:** {len(best_draft.get('evidence_subset', []))} documents analyzed."""
            return fallback

    def run(self, query: str) -> str:
        logger.info(f"Starting CF-RAG pipeline for query: {query}")
        
        try:
            logger.info("Stage 1: Generating counterfactual queries")
            cf_queries = self._generate_counterfactual_query(query)
            
            if not cf_queries:
                logger.warning("No counterfactual queries generated, continuing with empty list")
                cf_queries = []
            
            logger.info("Stage 2: Executing synergetic retrieval")
            candidate_docs = self._synergetic_retrieval(query, cf_queries)
            
            if not candidate_docs:
                logger.error("No documents retrieved")
                return "I apologize, but I couldn't find relevant information to answer your question."
            
            if config.ENABLE_SIMPLIFIED_MODE:
                logger.info("Stage 3: Simplified mode - Direct causal reranking")
                decisive_evidence = self._simplified_evidence_selection(query, cf_queries, candidate_docs)
                if not decisive_evidence:
                    logger.error("Failed to select evidence in simplified mode")
                    return "I apologize, but I couldn't find decisive evidence."
                
                drafts = self._generate_drafts_parallel(query, [decisive_evidence])
                best_draft = drafts[0] if drafts else {"draft_answer": "", "evidence_subset": decisive_evidence, "rationale": ""}
                logger.info("Simplified mode: Skipping verification stage")
            else:
                logger.info("Stage 3: Evidence clustering and multi-draft generation")
                
                logger.info("Stage 3a: Clustering and sampling evidence")
                evidence_subsets = self._cluster_and_sample_evidence(candidate_docs, query)
                if not evidence_subsets:
                    logger.error("Failed to create evidence subsets")
                    return "I apologize, but I couldn't organize the evidence effectively."
                
                logger.info("Stage 3b: Generating parallel drafts")
                drafts = self._generate_drafts_parallel(query, evidence_subsets)
                if not drafts:
                    logger.error("Failed to generate any answer drafts")
                    return "I apologize, but I couldn't generate answer drafts."
                
                logger.info("Stage 4: Counterfactual-enhanced verification")
                best_draft = self._verify_drafts(query, cf_queries, drafts)
                
                if not best_draft or not best_draft.get('draft_answer'):
                    logger.warning("Verification failed, falling back to the first draft")
                    best_draft = drafts[0]
            
            logger.info("Stage 5: Dialectical generation with verified evidence and reasoning")
            final_answer = self._generate_explanatory_answer_with_reasoning(query, best_draft)
            
            logger.info("CF-RAG pipeline completed successfully")
            return final_answer
            
        except Exception as e:
            logger.error(f"Error in CF-RAG pipeline: {e}")
            return f"An error occurred while processing your question: {str(e)}"

    def get_pipeline_stats(self) -> Dict[str, Any]:
        return {
            "model_info": {
                "llm_model": str(self.llm_model.__class__.__name__),
                "reranker_model": str(self.reranker.__class__.__name__),
                "device": str(self.llm_model.device)
            },
            "config": {
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "counterfactual_temperature": config.COUNTERFACTUAL_TEMPERATURE,
                "explanatory_temperature": config.EXPLANATORY_TEMPERATURE,
                "retrieval_top_k": config.RETRIEVAL_TOP_K,
                "reranker_top_k": config.RERANKER_TOP_K,
                "reranker_weight": config.RERANKER_WEIGHT,
                "relevance_threshold": config.RELEVANCE_THRESHOLD,
                "num_clusters": config.NUM_CLUSTERS,
                "num_drafts": config.NUM_DRAFTS,
                "verification_weight": config.VERIFICATION_WEIGHT
            },
            "pipeline_info": {
                "architecture": "Five-Stage CF-RAG with Multi-Draft Verification",
                "stages": [
                    "Counterfactual Query Generation",
                    "Synergetic Retrieval", 
                    "Evidence Clustering & Multi-Draft Generation",
                    "Counterfactual-Enhanced Verification",
                    "Dialectical Generation"
                ]
            }
        }