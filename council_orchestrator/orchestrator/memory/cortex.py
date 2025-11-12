# council_orchestrator/orchestrator/memory/cortex.py
# Mnemonic cortex vector database functionality

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import re
from ..config.safety import redact_pii, rate_limit_broad_prompt
from .cache import CacheManager

class CortexManager:
    """Manages the Mnemonic Cortex vector database for knowledge retrieval."""

    def __init__(self, project_root: Path, logger):
        self.project_root = project_root
        self.logger = logger
        # Access mnemonic_cortex at project root level
        chroma_db_path = project_root / "mnemonic_cortex/chroma_db"
        
        try:
            self.chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
        except BaseException as e:
            error_msg = str(e)
            if ("panic" in error_msg.lower() or "corrupted" in error_msg.lower() or "sqlite" in error_msg.lower() or 
                "range start index" in error_msg.lower() or "pyo3_runtime.PanicException" in str(type(e))):
                self.logger.critical("CODE RED: ChromaDB corruption detected! Halting all operations per Protocol 115.")
                self.logger.critical(f"Corruption details: {str(e)}")
                import sys
                sys.exit(1)
            else:
                # Re-raise if it's not a corruption error
                raise
        
        self.cortex_collection = self.chroma_client.get_or_create_collection(
            name="sanctuary_cortex",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        # Create CacheManager instance for cache operations
        self.cache_manager = CacheManager(project_root, logger)

    def query_cortex(self, query_text: str, n_results: int = 3) -> str:
        """Query the cortex for relevant knowledge."""
        try:
            results = self.cortex_collection.query(query_texts=[query_text], n_results=n_results)
            context = "CONTEXT_PROVIDED: Here are the top results from the Mnemonic Cortex for your query:\n\n"
            for doc in results['documents'][0]:
                context += f"---\n{doc}\n---\n"
            return context
        except Exception as e:
            error_message = f"CONTEXT_ERROR: Cortex query failed: {e}"
            print(f"[CORTEX] {error_message}")
            return error_message

    def get_latest_documents_by_path(self, path_prefix: str, n_results: int = 15) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent documents from a specific path prefix,
        sorted by the 'entry_id' metadata field.
        """
        self.logger.info(f"Fetching latest {n_results} documents from path prefix: {path_prefix}")
        try:
            # We fetch a larger number to sort them, as ChromaDB's default ordering is by similarity.
            results = self.cortex_collection.get(
                where={"source_file": {"$like": f"{path_prefix}%"}},
                limit=n_results * 2, # Fetch more to ensure we can sort and get the latest
                include=["metadatas"]
            )
            
            if not results or not results['metadatas']:
                self.logger.warning(f"No documents found for path prefix: {path_prefix}")
                return []

            # Sort the results by 'entry_id' (e.g., '281', '280') in descending order.
            # This requires converting the string ID to an integer for correct sorting.
            sorted_metadatas = sorted(
                results['metadatas'],
                key=lambda meta: int(re.search(r'(\d+)', meta.get('entry_id', '0')).group(1)) if re.search(r'(\d+)', meta.get('entry_id', '0')) else 0,
                reverse=True
            )
            
            # Return the top n_results as a list of dicts
            latest_docs = []
            for meta in sorted_metadatas[:n_results]:
                latest_docs.append({
                    "title": meta.get('title', '(untitled)'),
                    "path": meta.get('source_file', 'N/A'),
                    "updated_at": meta.get('timestamp', 'N/A')
                })

            self.logger.info(f"Successfully retrieved {len(latest_docs)} latest documents for {path_prefix}")
            return latest_docs

        except Exception as e:
            self.logger.error(f"Failed to get latest documents for {path_prefix}: {e}")
            return []

    def ingest_document(self, document: str, metadata: dict = None) -> bool:
        """Ingest a document into the cortex."""
        try:
            doc_id = f"doc_{hash(document) % 1000000}"
            self.cortex_collection.add(
                documents=[document],
                ids=[doc_id],
                metadatas=[metadata or {}]
            )

            # Phase 3: Refresh cache for updated files
            if metadata and 'path' in metadata:
                CacheManager.prefill_guardian_delta([metadata['path']])

            return True
        except Exception as e:
            print(f"[CORTEX] Failed to ingest document: {e}")
            return False

    def search_parent_docs(self, must=None, should=None, filters=None, k=6):
        """
        Phase 2: Search for parent documents using structured query.
        Returns list of dicts with doc_id, path, score, snippet, sha256.
        """
        try:
            # Build query from must/should terms
            query_terms = []
            if must:
                query_terms.extend(must)
            if should:
                query_terms.extend(should[:3])  # Limit should terms
            query_text = " ".join(query_terms) if query_terms else "general knowledge"

            # Execute search
            results = self.cortex_collection.query(
                query_texts=[query_text],
                n_results=k,
                where=filters if filters else None
            )

            hits = []
            for i, doc in enumerate(results['documents'][0]):
                # SAFETY: Redact PII from retrieved snippets
                safe_snippet = redact_pii(doc[:500]) if doc else ""
                hit = {
                    "doc_id": results['ids'][0][i] if results['ids'] else f"doc_{i}",
                    "path": results['metadatas'][0][i].get('path', '') if results['metadatas'] else '',
                    "score": float(results['distances'][0][i]) if results['distances'] else 0.0,
                    "snippet": safe_snippet,
                    "sha256": results['metadatas'][0][i].get('sha256', '') if results['metadatas'] else ''
                }
                hits.append(hit)

            # DEDUPLICATE near-duplicate hits before returning
            hits = self._deduplicate_parent_docs(hits)

            return hits

        except Exception as e:
            print(f"[CORTEX] Parent doc search failed: {e}")
            return []

# --- Phase 2: Self-Querying Retriever (skeleton) ---

@dataclass
class StructuredQuery:
    intent: str                    # e.g., "retrieve_parent_docs"
    must_terms: List[str] = field(default_factory=list)
    should_terms: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)  # {"path_prefix": "docs/", "file_types": ["md"]}
    k: int = 6

@dataclass
class ParentDocHit:
    doc_id: str
    path: str
    score: float
    snippet: Optional[str] = None
    sha256: Optional[str] = None

@dataclass
class NoveltySignal:
    is_novel: bool
    signal: str  # "none"|"low"|"medium"|"high"
    basis: Dict[str, Any] = field(default_factory=dict)  # { "overlap_ratio": 0.18, ... }

@dataclass
class ConflictSignal:
    conflicts_with: List[str] = field(default_factory=list)  # list of cache keys / doc ids
    basis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryDirective:
    tier: str                     # "fast" | "medium" | "slow"
    justification: str

@dataclass
class RetrievalSignals:
    structured_query: StructuredQuery
    parent_docs: List[ParentDocHit]
    retrieval_latency_ms: int

@dataclass
class RoundSignals:
    retrieval: RetrievalSignals
    novelty: NoveltySignal
    conflict: ConflictSignal
    memory_directive: MemoryDirective

class SelfQueryingRetriever:
    """
    Phase 2: Plans a structured retrieval, executes parent-doc lookup,
    computes novelty/conflict, and proposes a memory placement directive.
    """

    def __init__(self, cortex_idx, cache, prompt_hasher):
        """
        cortex_idx: your vector/parent-doc index adapter (read-only)
        cache: your CAG adapter (Phase 3 ready; can return hit/miss, streaks)
        prompt_hasher: callable[str]->str used to derive stable cache keys
        """
        self.cortex_idx = cortex_idx
        self.cache = cache
        self.hash_prompt = prompt_hasher

    # --- 1) Query Planning ----------------------------------------------------
    def plan_query(self, user_prompt: str, council_role: str) -> StructuredQuery:
        # SAFETY: Rate limit broad prompts to prevent index carpet-bombing
        rate_limit_check = rate_limit_broad_prompt(user_prompt)
        if not rate_limit_check["allow"]:
            # Return minimal query for broad prompts
            return StructuredQuery(
                intent="rate_limited_broad_prompt",
                must_terms=["general"],  # Minimal terms
                should_terms=[],
                filters={"file_types": ["md"]},
                k=3  # Limit results
            )

        # Extremely conservative first pass. Refine later with role heuristics.
        must, should = self._extract_terms(user_prompt)
        return StructuredQuery(
            intent="retrieve_parent_docs",
            must_terms=must,
            should_terms=should,
            filters={"file_types": ["md", "py", "txt"], "path_prefix": ""},
            k=6,
        )

    def _extract_terms(self, text: str) -> Tuple[List[str], List[str]]:
        # TODO: replace with lightweight keyword extractor; start with naive split
        toks = [t.strip(",.()[]{}:\"'").lower() for t in text.split()]
        toks = [t for t in toks if len(t) > 2]
        return toks[:5], toks[5:12]

    # --- 2) Parent-Doc Retrieval ----------------------------------------------
    def run_parent_doc_retrieval(self, q: StructuredQuery) -> RetrievalSignals:
        t0 = time.time()
        hits = self.cortex_idx.search_parent_docs(
            must=q.must_terms, should=q.should_terms, filters=q.filters, k=q.k
        )
        parent_docs = [
            ParentDocHit(
                doc_id=h["doc_id"],
                path=h.get("path",""),
                score=float(h.get("score", 0.0)),
                snippet=h.get("snippet"),
                sha256=h.get("sha256"),
            )
            for h in (hits or [])
        ]
        latency = int((time.time() - t0) * 1000)
        return RetrievalSignals(structured_query=q, parent_docs=parent_docs, retrieval_latency_ms=latency)

    # --- 3) Novelty & Conflict -------------------------------------------------
    def assess_novelty(self, prompt: str, parent_docs: List[ParentDocHit]) -> NoveltySignal:
        """
        Enhanced novelty assessment with raw overlap metrics (token/Jaccard/ROUGE).
        Logs comprehensive metrics for future tuning.
        """
        # Calculate multiple overlap metrics
        token_overlap = self._estimate_overlap(prompt, parent_docs)
        jaccard_similarity = self._calculate_jaccard(prompt, parent_docs)
        rouge1_metrics = self._calculate_rouge1(prompt, parent_docs)

        # Determine novelty signal based on combined metrics
        combined_score = (token_overlap + jaccard_similarity + rouge1_metrics.get("f1", 0)) / 3

        if combined_score < 0.25:
            signal = "high"
            is_novel = True
        elif combined_score < 0.55:
            signal = "medium"
            is_novel = True
        else:
            signal = "low"
            is_novel = False

        return NoveltySignal(
            is_novel=is_novel,
            signal=signal,
            basis={
                "token_overlap_ratio": token_overlap,
                "jaccard_similarity": jaccard_similarity,
                "rouge1_precision": rouge1_metrics.get("precision", 0),
                "rouge1_recall": rouge1_metrics.get("recall", 0),
                "rouge1_f1": rouge1_metrics.get("f1", 0),
                "combined_score": combined_score,
                "parent_docs_count": len(parent_docs)
            }
        )

    def _estimate_overlap(self, prompt: str, parent_docs: List[ParentDocHit]) -> float:
        # TODO: improve — quick token overlap proxy
        terms = set(self._extract_terms(prompt)[0] + self._extract_terms(prompt)[1])
        in_snips = " ".join([pd.snippet or "" for pd in parent_docs]).lower()
        covered = sum(1 for t in terms if t in in_snips)
        return 0.0 if not terms else covered / len(terms)

    def _calculate_jaccard(self, prompt: str, parent_docs: List[ParentDocHit]) -> float:
        """Calculate Jaccard similarity between prompt and retrieved docs."""
        prompt_tokens = set(self._extract_terms(prompt)[0] + self._extract_terms(prompt)[1])
        doc_tokens = set()
        for pd in parent_docs:
            doc_tokens.update(self._extract_terms(pd.snippet or "")[0] + self._extract_terms(pd.snippet or "")[1])

        if not prompt_tokens and not doc_tokens:
            return 1.0  # Both empty = identical
        if not prompt_tokens or not doc_tokens:
            return 0.0  # One empty = no similarity

        intersection = len(prompt_tokens.intersection(doc_tokens))
        union = len(prompt_tokens.union(doc_tokens))
        return intersection / union if union > 0 else 0.0

    def _calculate_rouge1(self, prompt: str, parent_docs: List[ParentDocHit]) -> Dict[str, float]:
        """Calculate ROUGE-1 metrics (unigram overlap)."""
        prompt_unigrams = set(prompt.lower().split())
        doc_unigrams = set()
        for pd in parent_docs:
            doc_unigrams.update((pd.snippet or "").lower().split())

        if not prompt_unigrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        intersection = len(prompt_unigrams.intersection(doc_unigrams))
        precision = intersection / len(prompt_unigrams) if prompt_unigrams else 0.0
        recall = intersection / len(doc_unigrams) if doc_unigrams else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def detect_conflict(self, prompt: str) -> ConflictSignal:
        """
        Enhanced conflict detection with human-readable reasons.
        Checks for conflicts with stable cached answers.
        """
        key = self.hash_prompt(prompt)
        entry = self.cache.peek(key)  # non-mutating look

        if entry and entry.get("stable") is True:
            cached_answer = entry.get("answer", "")
            cached_docs = entry.get("evidence_docs", [])
            cached_confidence = entry.get("confidence", 0.0)

            return ConflictSignal(
                conflicts_with=[f"cached_answer_{key[:8]}"],  # Short hash for readability
                basis={
                    "reason": "stable_cached_answer_exists",
                    "cached_answer_hash": self.hash_prompt(cached_answer)[:16],
                    "cached_evidence_docs": [doc.get("doc_id", "") for doc in cached_docs],
                    "cached_confidence": cached_confidence,
                    "conflict_type": "cached_vs_current_prompt"
                }
            )

        return ConflictSignal()

    # --- 4) Memory Placement ---------------------------------------------------
    def propose_memory_directive(
        self,
        confidence: float,
        citations: List[str],
        novelty: NoveltySignal,
        conflict: ConflictSignal,
        cache_hit_streak: int = 0,
        parent_docs: List[ParentDocHit] = None,  # Add for guardrail
    ) -> MemoryDirective:
        """
        Rules (with no-evidence guardrail):
        - NO-EVIDENCE GUARDRAIL: if parent_docs=[] or citations=[], downgrade to "fast" and cap confidence
        - conflict present => "fast" (needs arbitration)
        - high confidence + citations >=2 + cache_hit_streak>=3 => "slow"
        - else if citations>=1 or novelty is False => "medium"
        - else => "fast"
        """
        parent_docs = parent_docs or []

        # NO-EVIDENCE GUARDRAIL: Force fast tier if no evidence
        if not parent_docs or not citations:
            return MemoryDirective(
                tier="fast",
                justification="No-evidence guardrail: empty parent_docs or citations; force fast tier."
            )

        # MIN-EVIDENCE QUALITY CHECK: Validate evidence quality before allowing medium/slow
        evidence_quality = self._assess_evidence_quality(citations, parent_docs)
        if not evidence_quality["meets_threshold"]:
            return MemoryDirective(
                tier="fast",
                justification=f"Evidence quality below threshold: {evidence_quality['reason']}; force fast tier."
            )

        if conflict.conflicts_with:
            return MemoryDirective(tier="fast", justification="Conflict detected with stable cache entry; hold in fast memory for arbitration.")
        if confidence >= 0.8 and len(citations) >= 2 and cache_hit_streak >= 3:
            return MemoryDirective(tier="slow", justification="High confidence, strong evidence, recurring access; promote to Slow.")
        if (len(citations) >= 1) or (novelty.is_novel is False):
            return MemoryDirective(tier="medium", justification="Evidence present or not novel; store operationally.")
        return MemoryDirective(tier="fast", justification="Ephemeral or weakly supported; keep session-local.")

    def _validate_citation_overlap(self, citations: List[str], parent_docs: List[ParentDocHit], min_overlap_tokens: int = 3) -> bool:
        """
        Enforce must-have overlap: at least one citation must overlap ≥ N tokens with retrieved spans.
        Returns True if validation passes, False if citations are invalid (hallucinated).
        """
        if not citations or not parent_docs:
            return False

        # Combine all retrieved text
        retrieved_text = " ".join([pd.snippet or "" for pd in parent_docs]).lower()

        # Check each citation for overlap
        for citation in citations:
            citation_tokens = set(self._extract_terms(citation)[0] + self._extract_terms(citation)[1])
            overlap_count = sum(1 for token in citation_tokens if token in retrieved_text)
            if overlap_count >= min_overlap_tokens:
                return True

        return False  # No citation had sufficient overlap

    def _assess_evidence_quality(self, citations: List[str], parent_docs: List[ParentDocHit]) -> Dict[str, Any]:
        """
        Assess evidence quality using BM25 overlap and ROUGE-1 metrics.
        Returns dict with meets_threshold boolean and reason.
        """
        if not citations or not parent_docs:
            return {"meets_threshold": False, "reason": "missing_citations_or_docs"}

        # Combine all retrieved text for analysis
        retrieved_text = " ".join([pd.snippet or "" for pd in parent_docs])

        # Simple BM25-style overlap check (token frequency in retrieved docs)
        retrieved_tokens = set(retrieved_text.lower().split())
        citation_tokens = set()
        for citation in citations:
            citation_tokens.update(self._extract_terms(citation)[0] + self._extract_terms(citation)[1])

        # Calculate overlap ratio
        overlap_tokens = citation_tokens.intersection(retrieved_tokens)
        overlap_ratio = len(overlap_tokens) / len(citation_tokens) if citation_tokens else 0

        # BM25 threshold: >= 30% of citation tokens found in retrieved docs
        bm25_threshold = 0.3
        if overlap_ratio < bm25_threshold:
            return {
                "meets_threshold": False,
                "reason": f"BM25_overlap_{overlap_ratio:.2f}_below_{bm25_threshold}",
                "overlap_ratio": overlap_ratio
            }

        # Simple ROUGE-1 check (unigram overlap)
        citation_unigrams = set(" ".join(citations).lower().split())
        retrieved_unigrams = set(retrieved_text.lower().split())
        rouge1_precision = len(citation_unigrams.intersection(retrieved_unigrams)) / len(citation_unigrams) if citation_unigrams else 0

        rouge1_threshold = 0.2
        if rouge1_precision < rouge1_threshold:
            return {
                "meets_threshold": False,
                "reason": f"ROUGE1_{rouge1_precision:.2f}_below_{rouge1_threshold}",
                "rouge1_precision": rouge1_precision
            }

        return {
            "meets_threshold": True,
            "reason": "evidence_quality_passed",
            "overlap_ratio": overlap_ratio,
            "rouge1_precision": rouge1_precision
        }

    def _deduplicate_parent_docs(self, hits: List[dict], jaccard_threshold: float = 0.8) -> List[dict]:
        """
        Collapse near-duplicate RAG hits using Jaccard similarity on token sets.
        Returns deduplicated list, keeping highest-scoring representative of each cluster.
        """
        if not hits:
            return hits

        deduplicated = []

        for hit in sorted(hits, key=lambda x: x.get("score", 0), reverse=True):
            # Check if this hit is too similar to any already selected
            is_duplicate = False
            hit_tokens = set((hit.get("snippet") or "").lower().split())

            for selected in deduplicated:
                selected_tokens = set((selected.get("snippet") or "").lower().split())
                if hit_tokens and selected_tokens:
                    intersection = len(hit_tokens & selected_tokens)
                    union = len(hit_tokens | selected_tokens)
                    jaccard = intersection / union if union > 0 else 0
                    if jaccard >= jaccard_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated.append(hit)

        return deduplicated

    # --- 5) End-to-end convenience --------------------------------------------
    def run(
        self,*, prompt:str, council_role:str, confidence:float, citations:List[str]
    ) -> RoundSignals:
        # Stage 1: Plan query
        t0 = time.time()
        q = self.plan_query(prompt, council_role)
        plan_latency = int((time.time() - t0) * 1000)

        # Stage 2: Parent-doc retrieval
        t1 = time.time()
        retrieval = self.run_parent_doc_retrieval(q)
        retrieval_latency = int((time.time() - t1) * 1000)

        # Stage 3: Analyze (novelty + conflict)
        t2 = time.time()
        novelty = self.assess_novelty(prompt, retrieval.parent_docs)
        conflict = self.detect_conflict(prompt)

        # CITATION OVERLAP ENFORCEMENT: Validate citations against retrieved docs
        citations_valid = self._validate_citation_overlap(citations, retrieval.parent_docs)
        if citations and not citations_valid:
            # Citations exist but don't overlap - cap confidence and mark as potentially hallucinated
            confidence = min(confidence, 0.3)  # Cap at low confidence
            novelty = NoveltySignal(  # Override novelty to reflect citation issues
                is_novel=True,
                signal="high",
                basis={"citation_overlap_failure": True, "original_overlap": novelty.basis.get("overlap_ratio", 0)}
            )

        cache_key = self.hash_prompt(prompt)
        cache_hit_streak = int(self.cache.hit_streak(cache_key) or 0)
        directive = self.propose_memory_directive(
            confidence=confidence,
            citations=citations,
            novelty=novelty,
            conflict=conflict,
            cache_hit_streak=cache_hit_streak,
            parent_docs=retrieval.parent_docs,  # Pass for guardrail
        )
        analyze_latency = int((time.time() - t2) * 1000)

        # Stage 4: Emit (packet creation)
        t3 = time.time()
        signals = RoundSignals(
            retrieval=retrieval,
            novelty=novelty,
            conflict=conflict,
            memory_directive=directive,
        )
        emit_latency = int((time.time() - t3) * 1000)

        # Update retrieval signals with stage timings
        retrieval.plan_latency_ms = plan_latency
        retrieval.analyze_latency_ms = analyze_latency
        retrieval.emit_latency_ms = emit_latency

        return signals