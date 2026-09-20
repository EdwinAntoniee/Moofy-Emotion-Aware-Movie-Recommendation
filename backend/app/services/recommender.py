import gc
import json
import pickle
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import chromadb

from app.core.config import settings, MODELS_DIR, CHROMA_DB_DIR, DATA_DIR, ENRICHED_METADATA_FILE, MOVIES_CSV_FILE
from app.schemas.recommend import RecommendResponse, MovieCard, EmotionScore

# Comprehensive adult, erotic, NSFW and explicit content keywords to strictly filter out
ADULT_KEYWORDS = [
    r'\bxxx\b', r'erotic', r'\bporn', r'\bhentai\b', r'\bsoftcore\b', r'\bhardcore\b',
    r'\bhooker\b', r'\bstripper\b', r'\bstriptease\b', r'\bplayboy\b', r'\bvixen\b', r'\bprostitute\b',
    r'\bsex\b', r'\bsexual', r'\bseduction\b', r'\billicit\b', r'\bsensual',
    r'\btaboo\b', r'\bnude\b', r'\bnudity\b', r'\bbrothel\b', r'\bescort\b', r'\borgy\b',
    r'\blust\b', r'\bpeepshow\b', r'\bbondage\b', r'\bfetish\b'
]
ADULT_REGEX = re.compile('|'.join(ADULT_KEYWORDS), re.IGNORECASE)

# Restrict PyTorch background threads to conserve memory in constrained environments
try:
    torch.set_num_threads(1)
except Exception:
    pass

class RecommenderService:
    _instance = None

    def __init__(self):
        self.device = torch.device("cpu")
        print(f"[RecommenderService] Initializing on device: {self.device}")

        # 1. Load DistilBERT Emotion Classifier if weights exist on disk
        safetensors_path = MODELS_DIR / "model.safetensors"
        bin_path = MODELS_DIR / "pytorch_model.bin"
        
        self.distilbert_model = None
        self.tokenizer = None
        self.emotion_classes = ["Anger", "Fear", "Joy", "Love", "Sadness", "Surprise"]

        if safetensors_path.exists() or bin_path.exists():
            try:
                print(f"[RecommenderService] Loading DistilBERT weights from {MODELS_DIR}")
                self.tokenizer = DistilBertTokenizer.from_pretrained(str(MODELS_DIR))
                self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(str(MODELS_DIR))
                self.distilbert_model.to(self.device)
                self.distilbert_model.eval()
                print("[RecommenderService] DistilBERT model loaded successfully.")
            except Exception as e:
                print(f"[Warning] Failed to load local DistilBERT weights ({e}). Using semantic emotion engine.")
                self.distilbert_model = None
        else:
            print("[RecommenderService] Model weights not packaged in cloud image. Using high-precision SBERT semantic emotion engine.")

        gc.collect()

        # 2. Load Sentence-BERT on CPU
        print("[RecommenderService] Loading SentenceTransformer all-MiniLM-L6-v2")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        
        # Precompute emotion anchor embeddings once at startup
        self.emotion_anchors = [
            "anger rage fury aggressive violent mad revenge wrath hate hostility",
            "fear terror horror scary suspense anxiety dread panic spooky thriller danger nightmare",
            "joy happiness cheerful uplifting hilarious fun comedy adventure exciting celebration",
            "love romance romantic affection passion sweet heart couple intimacy crush tender",
            "sadness crying grief heartbreak depression sorrow tearful lonely mourn tragic despair",
            "surprise shocked unexpected plot twist mystery mind blowing astonishing revelation discovery"
        ]
        self.anchor_embeddings = self.sbert_model.encode(self.emotion_anchors)
        gc.collect()

        # 3. Connect to ChromaDB
        print(f"[RecommenderService] Connecting to ChromaDB at {CHROMA_DB_DIR}")
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.collection = self.chroma_client.get_or_create_collection(name="movie_synopses")
        
        # 4. Load Precomputed Vector Embeddings & Enriched Metadata
        self.movie_embeddings = None
        self.movie_ids = []
        embeddings_file = DATA_DIR / "movie_embeddings.npy"
        ids_file = DATA_DIR / "movie_ids.json"
        if embeddings_file.exists() and ids_file.exists():
            try:
                self.movie_embeddings = np.load(embeddings_file)
                with open(ids_file, "r", encoding="utf-8") as f:
                    self.movie_ids = [str(mid) for mid in json.load(f)]
                print(f"[RecommenderService] Loaded {len(self.movie_ids)} precomputed vector embeddings.")
            except Exception as e:
                print(f"[Warning] Failed to load precomputed embeddings: {e}")

        self.enriched_metadata: Dict[str, dict] = {}
        if ENRICHED_METADATA_FILE.exists():
            with open(ENRICHED_METADATA_FILE, "r", encoding="utf-8") as f:
                self.enriched_metadata = json.load(f)
            print(f"[RecommenderService] Loaded {len(self.enriched_metadata)} enriched movie records.")
        elif MOVIES_CSV_FILE.exists():
            df = pd.read_csv(MOVIES_CSV_FILE)
            for _, row in df.iterrows():
                m_id = str(row["movie_id"])
                self.enriched_metadata[m_id] = {
                    "movie_id": int(row["movie_id"]),
                    "title": str(row["title"]),
                    "overview": str(row.get("overview", "")),
                    "poster_path": None,
                    "backdrop_path": None,
                    "release_date": "",
                    "vote_average": 7.0,
                    "vote_count": 100,
                    "genres": [],
                    "emotion_label": str(row.get("emotion_label", "Joy"))
                }

        # 5. Auto-seed ChromaDB if collection is empty using precomputed embeddings
        if self.collection.count() == 0:
            print("[RecommenderService] ChromaDB collection is empty. Auto-seeding from precomputed vectors...")
            self._seed_chroma_collection()

        print(f"[RecommenderService] ChromaDB collection ready. Item count: {self.collection.count()}")
        gc.collect()

    def _seed_chroma_collection(self):
        """Auto-seed ChromaDB without expensive runtime encoding."""
        try:
            if self.movie_embeddings is not None and len(self.movie_ids) > 0:
                print(f"[RecommenderService] Seeding ChromaDB from precomputed {len(self.movie_ids)} embeddings...")
                batch_size = 200
                for i in range(0, len(self.movie_ids), batch_size):
                    batch_ids = self.movie_ids[i:i+batch_size]
                    batch_embs = self.movie_embeddings[i:i+batch_size].tolist()
                    batch_docs = [self.enriched_metadata.get(mid, {}).get("overview", "") for mid in batch_ids]
                    batch_metas = [{
                        "title": self.enriched_metadata.get(mid, {}).get("title", ""),
                        "emotion_label": self.enriched_metadata.get(mid, {}).get("emotion_label", "Joy"),
                        "vote_average": float(self.enriched_metadata.get(mid, {}).get("vote_average", 7.0))
                    } for mid in batch_ids]

                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embs,
                        documents=batch_docs,
                        metadatas=batch_metas
                    )
                print(f"[RecommenderService] Successfully indexed {len(self.movie_ids)} movies into ChromaDB.")
                gc.collect()
                return
        except Exception as e:
            print(f"[Error] Failed to auto-seed ChromaDB: {e}")

    @classmethod
    def get_instance(cls) -> "RecommenderService":
        if cls._instance is None:
            cls._instance = RecommenderService()
        return cls._instance

    def is_safe_and_high_quality(self, title: str, overview: str, vote_avg: float, vote_count: int) -> bool:
        """Filter out adult/erotic content, unvetted placeholder releases, and very low quality films."""
        # 1. Quality threshold: Must have rating >= 5.8 and at least 15 votes
        if vote_avg < 5.8 or vote_count < 15:
            return False

        # 2. Adult / Explicit content filter
        combined_text = f"{title} {overview}"
        if ADULT_REGEX.search(combined_text):
            return False

        return True

    def predict_emotion(self, text: str) -> Tuple[str, Dict[str, float], List[EmotionScore]]:
        """Predict emotion probabilities for an emotion prompt with robust multi-layer fallback."""
        if self.distilbert_model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                ).to(self.device)

                with torch.inference_mode():
                    outputs = self.distilbert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

                emotion_dict = {
                    self.emotion_classes[i]: float(probs[i])
                    for i in range(len(self.emotion_classes))
                }
                primary_emotion = self.emotion_classes[int(np.argmax(probs))]
            except Exception as e:
                print(f"[Warning] DistilBERT inference failed ({e}). Using semantic emotion anchor mapping.")
                primary_emotion, emotion_dict = self._semantic_emotion_predict(text)
        else:
            primary_emotion, emotion_dict = self._semantic_emotion_predict(text)

        breakdown = [
            EmotionScore(
                emotion=emo,
                score=score,
                percentage=int(round(score * 100))
            )
            for emo, score in sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
        ]

        return primary_emotion, emotion_dict, breakdown

    def _semantic_emotion_predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Instant semantic cosine similarity against precomputed emotion anchors."""
        query_emb = self.sbert_model.encode(text)
        norm_q = np.linalg.norm(query_emb)
        norm_a = np.linalg.norm(self.anchor_embeddings, axis=1)
        sims = np.dot(self.anchor_embeddings, query_emb) / (norm_a * norm_q + 1e-9)
        exp_sims = np.exp(sims * 5)
        probs = exp_sims / np.sum(exp_sims)
        emotion_dict = {
            self.emotion_classes[i]: float(probs[i])
            for i in range(len(self.emotion_classes))
        }
        primary_emotion = self.emotion_classes[int(np.argmax(probs))]
        return primary_emotion, emotion_dict

    def recommend(
        self,
        prompt: str,
        alpha: float = 0.5,
        top_k: int = 12,
        filter_emotion: Optional[str] = None
    ) -> RecommendResponse:
        """Hybrid Recommendation Pipeline."""
        # 1. Predict Emotion Distribution
        primary_emotion, emotion_dict, breakdown = self.predict_emotion(prompt)

        # 2. Semantic Search with ChromaDB
        # 2. Semantic Search (Ultra-fast precomputed matrix, fallback to ChromaDB)
        prompt_emb_np = self.sbert_model.encode(prompt)
        prompt_embedding = prompt_emb_np.tolist()
        
        candidate_ids = []
        distances = []

        if self.movie_embeddings is not None and len(self.movie_ids) > 0:
            try:
                norm_q = np.linalg.norm(prompt_emb_np)
                norm_m = np.linalg.norm(self.movie_embeddings, axis=1)
                sims = np.dot(self.movie_embeddings, prompt_emb_np) / (norm_m * norm_q + 1e-9)
                top_idx = np.argsort(sims)[::-1][:60]
                candidate_ids = [self.movie_ids[i] for i in top_idx]
                distances = [float(max(0.0, 1.0 - sims[i])) for i in top_idx]
            except Exception as e:
                print(f"[Warning] Precomputed vector search failed: {e}")

        if not candidate_ids and self.collection.count() > 0:
            try:
                n_res = min(60, self.collection.count())
                query_res = self.collection.query(
                    query_embeddings=[prompt_embedding],
                    n_results=n_res,
                    include=["documents", "metadatas", "distances"]
                )
                candidate_ids = query_res["ids"][0] if query_res.get("ids") else []
                distances = query_res["distances"][0] if query_res.get("distances") else []
            except Exception as e:
                print(f"[Warning] ChromaDB query failed: {e}")

        # Fallback if both empty or failed
        if not candidate_ids and self.enriched_metadata:
            candidate_ids = list(self.enriched_metadata.keys())[:60]
            distances = [1.0] * len(candidate_ids)

        scored_candidates = []

        for m_id, dist in zip(candidate_ids, distances):
            meta = self.enriched_metadata.get(str(m_id), {})
            movie_title = meta.get("title", f"Movie #{m_id}")
            overview = meta.get("overview", "")
            vote_avg = float(meta.get("vote_average", 7.0))
            vote_count = int(meta.get("vote_count", 100))
            movie_emotion = meta.get("emotion_label", primary_emotion)

            # Strict Adult / Low-Quality Filter
            if not self.is_safe_and_high_quality(movie_title, overview, vote_avg, vote_count):
                continue

            if filter_emotion and filter_emotion != "ALL" and movie_emotion != filter_emotion:
                continue

            # Semantic similarity score
            sim_score = max(0.0, 1.0 - (dist / 2.0))

            # Emotion resonance score
            emo_score = emotion_dict.get(movie_emotion, 0.0)

            # Adaptive Hybrid Score Formula
            hybrid_score = (alpha * sim_score) + ((1.0 - alpha) * emo_score)

            card = MovieCard(
                movie_id=int(m_id),
                title=movie_title,
                overview=overview,
                poster_path=meta.get("poster_path"),
                backdrop_path=meta.get("backdrop_path"),
                release_date=meta.get("release_date", ""),
                vote_average=vote_avg,
                vote_count=vote_count,
                genres=meta.get("genres", []),
                emotion_label=movie_emotion,
                semantic_similarity=round(float(sim_score), 4),
                emotion_resonance=round(float(emo_score), 4),
                hybrid_score=round(float(hybrid_score), 4)
            )

            scored_candidates.append(card)

        # Sort candidate movies by hybrid score descending
        scored_candidates.sort(key=lambda x: x.hybrid_score, reverse=True)
        final_movies = scored_candidates[:top_k]

        return RecommendResponse(
            prompt=prompt,
            primary_emotion=primary_emotion,
            emotion_breakdown=breakdown,
            alpha=alpha,
            total_results=len(final_movies),
            movies=final_movies
        )
