"""
rachel_memory.py — Autobiographical memory system for Rachel VTuber AI
══════════════════════════════════════════════════════════════════════
Rachel writes her own memories in first person, tagged with emotion
and intensity from the <|channel>emotion output. Memories are embedded
and stored in ChromaDB for semantic recall.

Three stages:
  1. ENCODE  — After each interaction, Rachel generates a 1-2 sentence
               memory. The emotion/intensity from the channel output are
               stored as metadata alongside the vector embedding.
  2. RECALL  — At inference time, query memories by semantic similarity
               with emotion-congruent boosting (current mood biases
               recall toward matching emotional memories).
  3. DREAM   — Offline post-stream consolidation. Merges near-duplicate
               memories, decays low-salience entries, strengthens
               emotionally intense ones. Mimics memory consolidation.

Install:
    pip install chromadb requests numpy

Embedding server (separate llama-server instance):
    llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --embedding --host 0.0.0.0 --port 8081 -c 2048 --pooling mean

Usage:
    from rachel_memory import RachelMemory
    mem = RachelMemory()
    mem.encode("Ben clutched that 1v4 and I screamed", emotion="joy", intensity="high")
    results = mem.recall("remember that clutch play?", current_emotion="joy")
    mem.dream()  # run post-stream
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

try:
    import chromadb
except ImportError:
    raise ImportError("pip install chromadb")

from config import EMBED_URL, EMBED_MODEL, MEMORY_DB_PATH

log = logging.getLogger("rachel-memory")

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

EMBED_TIMEOUT   = 30
PERSIST_DIR     = MEMORY_DB_PATH
COLLECTION_NAME = "rachel_autobio"

# Emotion vocabulary (matches Rachel V5 training set)
EMOTIONS = [
    "joy", "sadness", "anger", "fear", "surprise",
    "disgust", "love", "neutral", "sarcasm",
]
INTENSITIES = ["low", "medium", "high"]

# Emotion similarity matrix — emotions that feel "close" to each other.
# Used for mood-congruent recall boosting.
# Higher = more similar. 1.0 = identical emotion.
EMOTION_AFFINITY = {
    "joy":      {"joy": 1.0, "love": 0.7, "surprise": 0.5, "sarcasm": 0.3},
    "love":     {"love": 1.0, "joy": 0.7, "surprise": 0.4},
    "sadness":  {"sadness": 1.0, "fear": 0.5, "anger": 0.3, "disgust": 0.3},
    "anger":    {"anger": 1.0, "disgust": 0.6, "sadness": 0.3, "sarcasm": 0.4},
    "fear":     {"fear": 1.0, "sadness": 0.5, "surprise": 0.4},
    "surprise": {"surprise": 1.0, "joy": 0.5, "fear": 0.4},
    "disgust":  {"disgust": 1.0, "anger": 0.6, "sarcasm": 0.4},
    "sarcasm":  {"sarcasm": 1.0, "anger": 0.4, "joy": 0.3},
    "neutral":  {"neutral": 1.0},
}

# Intensity multiplier for salience scoring
INTENSITY_WEIGHT = {"low": 0.5, "medium": 1.0, "high": 1.5}

# Dream stage parameters
DREAM_SIMILARITY_THRESHOLD = 0.92  # memories above this similarity get merged
DREAM_DECAY_DAYS           = 30    # neutral/low memories older than this decay
DREAM_DECAY_RATE           = 0.1   # subtracted from salience per dream cycle
DREAM_MIN_SALIENCE         = 0.1   # below this, memory is pruned


# ═══════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════

def get_embedding(text: str) -> list[float]:
    """Get embedding vector from LM Studio (OpenAI-compatible /v1/embeddings)."""
    try:
        resp = requests.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "input": text},
            timeout=EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI format: {"data": [{"embedding": [...]}]}
        # llama-server legacy format: {"embedding": [...]}
        if "data" in data and isinstance(data["data"], list):
            return data["data"][0]["embedding"]
        elif "embedding" in data:
            return data["embedding"]
        else:
            raise ValueError(f"Unexpected embedding response: {list(data.keys())}")
    except Exception as e:
        log.error(f"Embedding error: {e}")
        raise


# ═══════════════════════════════════════════════════════════
# MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════

class RachelMemory:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"Memory loaded: {self.collection.count()} memories in {persist_dir}")

    # ─── ENCODE ────────────────────────────────────────────

    def encode(
        self,
        memory_text: str,
        emotion: str = "neutral",
        intensity: str = "medium",
        context: str = "",
        memory_type: str = "stream_moment",
    ) -> str:
        """
        Store a new memory. Rachel writes these herself — first person,
        past tense, capturing what happened and how she felt.

        Args:
            memory_text: Rachel's first-person memory, e.g.
                         "Ben clutched that 1v4 and I screamed so loud lol"
            emotion:     From channel output (joy, anger, etc.)
            intensity:   low / medium / high
            context:     Optional extra context (game, scene, etc.)
            memory_type: stream_moment | conversation | fact | dream_merged

        Returns:
            Memory ID (hash)
        """
        emotion = emotion.lower() if emotion.lower() in EMOTIONS else "neutral"
        intensity = intensity.lower() if intensity.lower() in INTENSITIES else "medium"

        # Salience = intensity weight × recency (starts at 1.0)
        salience = INTENSITY_WEIGHT.get(intensity, 1.0)

        now = datetime.now()
        memory_id = hashlib.md5(
            f"{memory_text}:{now.isoformat()}".encode()
        ).hexdigest()

        embedding = get_embedding(memory_text)

        metadata = {
            "emotion":     emotion,
            "intensity":   intensity,
            "salience":    salience,
            "memory_type": memory_type,
            "context":     context,
            "created_at":  now.isoformat(),
            "last_recalled": "",
            "recall_count": 0,
            "dream_count":  0,
        }

        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[memory_text],
            metadatas=[metadata],
        )

        log.info(
            f"Encoded: [{emotion}/{intensity}] {memory_text[:60]}... "
            f"(salience={salience:.1f})"
        )
        return memory_id

    def encode_from_response(
        self,
        rachel_channel_text: str,
        rachel_spoken_text: str,
        user_input: str,
        emotion: str,
        intensity: str,
        llm_url: str = None,
    ) -> str | None:
        """
        After Rachel responds, ask her to write a memory of what just happened.

        If llm_url is provided, sends a prompt to the LLM to generate the
        memory text in Rachel's voice. Otherwise, constructs a simple memory
        from the channel + spoken text.

        Args:
            rachel_channel_text: Raw emotion channel content
            rachel_spoken_text:  What Rachel said out loud
            user_input:          What triggered the response
            emotion:             Parsed emotion
            intensity:           Parsed intensity
            llm_url:             Optional LLM endpoint for memory generation

        Returns:
            Memory ID or None if the memory was too boring to store
        """
        # Skip encoding for trivial/neutral interactions
        if emotion == "neutral" and intensity == "low":
            # Only store 1 in 5 neutral/low memories to avoid noise
            if hash(rachel_spoken_text) % 5 != 0:
                return None

        if llm_url:
            # Ask Rachel to write her own memory
            memory_text = self._generate_memory_text(
                llm_url, rachel_channel_text, rachel_spoken_text,
                user_input, emotion, intensity,
            )
        else:
            # Fallback: construct from response
            memory_text = self._construct_memory(
                rachel_spoken_text, user_input, emotion,
            )

        if not memory_text or len(memory_text) < 10:
            return None

        return self.encode(
            memory_text=memory_text,
            emotion=emotion,
            intensity=intensity,
            context=user_input[:100],
        )

    def _generate_memory_text(
        self, llm_url, channel, spoken, user_input, emotion, intensity,
    ) -> str:
        """Ask the LLM to write a memory in Rachel's voice."""
        prompt = (
            "Write a 1-2 sentence memory of what just happened, in first person, "
            "past tense, as Rachel would remember it. Be specific about what "
            "happened, not generic. Include how you felt.\n\n"
            f"What Ben said/did: {user_input}\n"
            f"My internal reaction: {channel}\n"
            f"What I said: {spoken}\n"
            f"My emotion: {emotion}/{intensity}\n\n"
            "My memory (1-2 sentences, first person):"
        )
        try:
            resp = requests.post(llm_url, json={
                "model": "rachel",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": False,
            }, timeout=30)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # Clean up — remove quotes, prefixes
            text = text.strip('"').strip("'")
            if text.startswith("Memory:"):
                text = text[7:].strip()
            return text
        except Exception as e:
            log.warning(f"Memory generation failed: {e}")
            return self._construct_memory(spoken, user_input, emotion)

    def _construct_memory(self, spoken, user_input, emotion) -> str:
        """Simple fallback memory construction without LLM."""
        # Trim to essential content
        spoken_short = spoken[:120].strip()
        input_short = user_input[:80].strip()
        return f"Ben said '{input_short}' and I responded '{spoken_short}'"

    # ─── RECALL ────────────────────────────────────────────

    def recall(
        self,
        query: str,
        n_results: int = 5,
        current_emotion: str = None,
        current_intensity: str = None,
        memory_type: str = None,
        max_age_days: int = None,
    ) -> list[dict]:
        """
        Recall memories by semantic similarity with mood-congruent boosting.

        When Rachel is currently feeling joy/high, joyful memories get a
        boost in the ranking. This mimics mood-congruent memory bias in
        humans — you remember happy things more easily when you're happy.

        Args:
            query:             What to search for
            n_results:         Max memories to return
            current_emotion:   Rachel's current emotion (for mood boost)
            current_intensity: Rachel's current intensity
            memory_type:       Filter by type (stream_moment, conversation, etc.)
            max_age_days:      Only return memories within this many days

        Returns:
            List of {text, emotion, intensity, salience, age_days, score}
        """
        # Fetch more than needed so we can re-rank after mood boosting
        fetch_n = min(n_results * 3, self.collection.count() or 1)

        where_filter = {}
        if memory_type:
            where_filter["memory_type"] = memory_type

        try:
            embedding = get_embedding(query)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=fetch_n,
                where=where_filter if where_filter else None,
            )
        except Exception as e:
            log.error(f"Recall error: {e}")
            return []

        if not results["documents"] or not results["documents"][0]:
            return []

        # Build scored results
        scored = []
        now = datetime.now()

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Base score = cosine similarity (1 - distance for cosine)
            similarity = 1.0 - dist

            # Mood-congruent boost
            mood_boost = 0.0
            if current_emotion and meta.get("emotion"):
                affinity_map = EMOTION_AFFINITY.get(current_emotion, {})
                mood_boost = affinity_map.get(meta["emotion"], 0.0) * 0.15

            # Intensity boost (high intensity memories are more memorable)
            intensity_boost = INTENSITY_WEIGHT.get(
                meta.get("intensity", "medium"), 1.0
            ) * 0.05

            # Salience factor
            salience = meta.get("salience", 1.0)
            salience_boost = salience * 0.1

            # Recency factor (slight boost for recent memories)
            created = meta.get("created_at", "")
            age_days = 0
            if created:
                try:
                    created_dt = datetime.fromisoformat(created)
                    age_days = (now - created_dt).days
                except Exception:
                    pass

            if max_age_days and age_days > max_age_days:
                continue

            recency_boost = max(0, 0.05 - (age_days * 0.001))

            # Combined score
            final_score = (
                similarity + mood_boost + intensity_boost +
                salience_boost + recency_boost
            )

            scored.append({
                "text":      doc,
                "emotion":   meta.get("emotion", "neutral"),
                "intensity": meta.get("intensity", "medium"),
                "salience":  salience,
                "age_days":  age_days,
                "score":     round(final_score, 4),
                "similarity": round(similarity, 4),
                "mood_boost": round(mood_boost, 4),
                "id":        results["ids"][0][len(scored)] if results.get("ids") else "",
            })

        # Sort by final score, return top N
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Update recall metadata for returned memories
        for m in scored[:n_results]:
            if m.get("id"):
                try:
                    self.collection.update(
                        ids=[m["id"]],
                        metadatas=[{
                            "last_recalled": now.isoformat(),
                            "recall_count": meta.get("recall_count", 0) + 1,
                        }],
                    )
                except Exception:
                    pass

        return scored[:n_results]

    def recall_for_prompt(
        self,
        query: str,
        current_emotion: str = None,
        n_results: int = 3,
    ) -> str:
        """
        Returns a formatted memory block ready for prompt injection.

        Example output:
            [RACHEL'S MEMORIES]
            - (2 days ago, joy/high) Ben clutched that 1v4 and I screamed
            - (5 days ago, surprise/high) We hit 10k viewers for the first time
            - (12 days ago, joy/medium) Chat did the wave and it was adorable
        """
        results = self.recall(
            query,
            n_results=n_results,
            current_emotion=current_emotion,
        )

        if not results:
            return ""

        lines = ["[RACHEL'S MEMORIES]"]
        for r in results:
            age = r["age_days"]
            if age == 0:
                age_str = "today"
            elif age == 1:
                age_str = "yesterday"
            else:
                age_str = f"{age} days ago"

            lines.append(
                f"- ({age_str}, {r['emotion']}/{r['intensity']}) "
                f"{r['text'][:150]}"
            )

        return "\n".join(lines)

    # ─── DREAM (Consolidation) ─────────────────────────────

    def dream(self, verbose: bool = True) -> dict:
        """
        Post-stream memory consolidation. Run this when the stream ends
        or on a schedule (e.g. daily).

        Three operations:
        1. MERGE  — Find near-duplicate memories (>0.92 cosine similarity)
                    and merge them into a single stronger memory.
        2. DECAY  — Reduce salience of old neutral/low memories.
        3. PRUNE  — Remove memories below minimum salience threshold.

        Returns:
            Stats dict: {merged, decayed, pruned, total_before, total_after}
        """
        stats = {
            "merged": 0, "decayed": 0, "pruned": 0,
            "total_before": self.collection.count(),
        }

        if verbose:
            print(f"🌙 Dream stage starting... ({stats['total_before']} memories)")

        # ── Step 1: Merge duplicates ──
        stats["merged"] = self._dream_merge(verbose)

        # ── Step 2: Decay old low-salience memories ──
        stats["decayed"] = self._dream_decay(verbose)

        # ── Step 3: Prune dead memories ──
        stats["pruned"] = self._dream_prune(verbose)

        stats["total_after"] = self.collection.count()

        if verbose:
            print(f"🌙 Dream complete:")
            print(f"   Merged:  {stats['merged']} duplicate pairs")
            print(f"   Decayed: {stats['decayed']} memories")
            print(f"   Pruned:  {stats['pruned']} memories")
            print(f"   Total:   {stats['total_before']} → {stats['total_after']}")

        return stats

    def _dream_merge(self, verbose: bool) -> int:
        """Find and merge near-duplicate memories."""
        merged_count = 0
        all_data = self.collection.get(include=["documents", "metadatas", "embeddings"])

        if not all_data["ids"]:
            return 0

        ids = all_data["ids"]
        docs = all_data["documents"]
        metas = all_data["metadatas"]
        embeds = np.array(all_data["embeddings"])

        # Track which IDs have been merged (to skip)
        merged_ids = set()

        for i in range(len(ids)):
            if ids[i] in merged_ids:
                continue

            for j in range(i + 1, len(ids)):
                if ids[j] in merged_ids:
                    continue

                # Cosine similarity
                sim = np.dot(embeds[i], embeds[j]) / (
                    np.linalg.norm(embeds[i]) * np.linalg.norm(embeds[j]) + 1e-8
                )

                if sim >= DREAM_SIMILARITY_THRESHOLD:
                    # Merge: keep the one with higher salience, boost it
                    sal_i = metas[i].get("salience", 1.0)
                    sal_j = metas[j].get("salience", 1.0)

                    if sal_i >= sal_j:
                        keep_idx, drop_idx = i, j
                    else:
                        keep_idx, drop_idx = j, i

                    keep_id = ids[keep_idx]
                    drop_id = ids[drop_idx]
                    keep_meta = metas[keep_idx]

                    # Boost salience of survivor (memories that repeat matter)
                    new_salience = min(
                        keep_meta.get("salience", 1.0) + 0.2,
                        3.0,  # cap
                    )
                    keep_meta["salience"] = new_salience
                    keep_meta["dream_count"] = keep_meta.get("dream_count", 0) + 1
                    keep_meta["memory_type"] = "dream_merged"

                    self.collection.update(ids=[keep_id], metadatas=[keep_meta])
                    self.collection.delete(ids=[drop_id])
                    merged_ids.add(drop_id)
                    merged_count += 1

                    if verbose:
                        print(
                            f"   🔀 Merged: '{docs[drop_idx][:40]}...' "
                            f"into '{docs[keep_idx][:40]}...' "
                            f"(sim={sim:.3f}, salience→{new_salience:.1f})"
                        )

        return merged_count

    def _dream_decay(self, verbose: bool) -> int:
        """Reduce salience of old, low-intensity memories."""
        decayed_count = 0
        cutoff = (datetime.now() - timedelta(days=DREAM_DECAY_DAYS)).isoformat()

        # Get all memories
        all_data = self.collection.get(include=["metadatas"])
        if not all_data["ids"]:
            return 0

        for mem_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            created = meta.get("created_at", "")
            intensity = meta.get("intensity", "medium")
            salience = meta.get("salience", 1.0)

            # Only decay neutral/low intensity memories older than cutoff
            if intensity in ["low"] and created and created < cutoff:
                new_salience = max(salience - DREAM_DECAY_RATE, 0.0)
                if new_salience != salience:
                    meta["salience"] = new_salience
                    meta["dream_count"] = meta.get("dream_count", 0) + 1
                    self.collection.update(ids=[mem_id], metadatas=[meta])
                    decayed_count += 1

            # Also decay very old medium memories (60+ days)
            old_cutoff = (
                datetime.now() - timedelta(days=DREAM_DECAY_DAYS * 2)
            ).isoformat()
            if (intensity == "medium" and created and created < old_cutoff
                    and meta.get("recall_count", 0) == 0):
                new_salience = max(salience - DREAM_DECAY_RATE * 0.5, 0.0)
                if new_salience != salience:
                    meta["salience"] = new_salience
                    meta["dream_count"] = meta.get("dream_count", 0) + 1
                    self.collection.update(ids=[mem_id], metadatas=[meta])
                    decayed_count += 1

        return decayed_count

    def _dream_prune(self, verbose: bool) -> int:
        """Remove memories below minimum salience."""
        pruned_count = 0
        all_data = self.collection.get(include=["metadatas", "documents"])
        if not all_data["ids"]:
            return 0

        to_delete = []
        for mem_id, meta, doc in zip(
            all_data["ids"], all_data["metadatas"], all_data["documents"]
        ):
            salience = meta.get("salience", 1.0)
            if salience < DREAM_MIN_SALIENCE:
                to_delete.append(mem_id)
                if verbose:
                    print(f"   🗑️  Pruned: '{doc[:50]}...' (salience={salience:.2f})")

        if to_delete:
            self.collection.delete(ids=to_delete)
            pruned_count = len(to_delete)

        return pruned_count

    # ─── UTILITIES ─────────────────────────────────────────

    def stats(self) -> dict:
        """Return memory statistics."""
        all_data = self.collection.get(include=["metadatas"])
        if not all_data["ids"]:
            return {"total": 0}

        emotion_counts = {}
        intensity_counts = {}
        type_counts = {}
        total_salience = 0.0

        for meta in all_data["metadatas"]:
            emo = meta.get("emotion", "unknown")
            intensity = meta.get("intensity", "unknown")
            mtype = meta.get("memory_type", "unknown")
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
            type_counts[mtype] = type_counts.get(mtype, 0) + 1
            total_salience += meta.get("salience", 1.0)

        total = len(all_data["ids"])
        return {
            "total":       total,
            "by_emotion":  emotion_counts,
            "by_intensity": intensity_counts,
            "by_type":     type_counts,
            "avg_salience": round(total_salience / total, 2) if total else 0,
        }

    def dump(self, limit: int = 20) -> list[dict]:
        """Dump recent memories for debugging."""
        all_data = self.collection.get(
            include=["metadatas", "documents"],
            limit=limit,
        )
        results = []
        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            results.append({
                "text":      doc,
                "emotion":   meta.get("emotion"),
                "intensity": meta.get("intensity"),
                "salience":  meta.get("salience"),
                "created":   meta.get("created_at"),
                "recalls":   meta.get("recall_count", 0),
            })
        return results


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    mem = RachelMemory()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rachel_memory.py encode 'memory text' joy high")
        print("  python rachel_memory.py recall 'query text'")
        print("  python rachel_memory.py dream")
        print("  python rachel_memory.py stats")
        print("  python rachel_memory.py dump")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "encode":
        text = sys.argv[2] if len(sys.argv) > 2 else "Test memory"
        emotion = sys.argv[3] if len(sys.argv) > 3 else "neutral"
        intensity = sys.argv[4] if len(sys.argv) > 4 else "medium"
        mid = mem.encode(text, emotion=emotion, intensity=intensity)
        print(f"Encoded: {mid}")

    elif cmd == "recall":
        query = sys.argv[2] if len(sys.argv) > 2 else "what happened?"
        emotion = sys.argv[3] if len(sys.argv) > 3 else None
        results = mem.recall(query, current_emotion=emotion)
        for r in results:
            print(
                f"  [{r['emotion']}/{r['intensity']}] "
                f"score={r['score']:.3f} sim={r['similarity']:.3f} "
                f"mood+={r['mood_boost']:.3f} | {r['text'][:80]}"
            )

    elif cmd == "dream":
        stats = mem.dream(verbose=True)

    elif cmd == "stats":
        s = mem.stats()
        print(json.dumps(s, indent=2))

    elif cmd == "dump":
        for m in mem.dump():
            print(
                f"  [{m['emotion']}/{m['intensity']}] "
                f"sal={m['salience']:.1f} recalls={m['recalls']} | "
                f"{m['text'][:80]}"
            )