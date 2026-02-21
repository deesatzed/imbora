"""LLM-guided per-column semantic analysis and treatment planning.

Enriches ColumnProfile objects with structured per-column decisions:
- semantic_meaning: what real-world concept this column represents
- semantic_dtype: higher-level type (currency, zipcode, ICD_code, etc.)
- imputation_strategy: per-column imputation method
- encoding_strategy: per-column encoding method
- outlier_strategy: per-column outlier treatment
- text_processing_strategy: for text columns
- interaction_candidates: columns to create interactions with
- data_quality_flags: issues (constant, leakage_suspect, ID_column, etc.)
- importance_prior: expected importance (high, medium, low, drop)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.datasci.models import ColumnProfile
from src.llm.client import LLMMessage

logger = logging.getLogger("associate.datasci.column_ai_enricher")

ENRICHMENT_SYSTEM_PROMPT = """You are an expert data scientist performing semantic column analysis.
Given column metadata (name, dtype, statistics, sample values), determine the optimal
per-column treatment strategy.

Output a JSON array where each element has these fields:
- column: string (column name)
- semantic_meaning: string (what real-world concept, e.g. "patient age in years")
- semantic_dtype: string (e.g. "age", "currency", "zipcode", "ICD_code", "email", "text_review", "count", "ratio", "binary_flag", "identifier")
- imputation_strategy: string (one of: "median", "mean", "mode", "knn", "mice", "forward_fill", "domain_zero", "fill_empty_string", "drop_row")
- encoding_strategy: string (one of: "keep_numeric", "onehot", "ordinal", "target_encode", "frequency", "hash", "sentence_embeddings", "tfidf")
- outlier_strategy: string (one of: "none", "clip_to_iqr", "winsorize", "log_transform", "remove", "keep")
- text_processing_strategy: string (one of: "none", "tfidf", "embed_sentence_transformer", "llm_extract_entities", "regex_extract")
- interaction_candidates: list of strings (other column names to create interactions with)
- data_quality_flags: list of strings (any of: "constant", "id_column", "leakage_suspect", "near_duplicate", "high_missing")
- importance_prior: string (one of: "high", "medium", "low", "drop")

Rules:
1. ID columns (names like "id", "index", "row_num") should get importance_prior="drop"
2. Columns with >50% missing should get data_quality_flags=["high_missing"]
3. Constant columns should get data_quality_flags=["constant"], importance_prior="drop"
4. Text columns should get encoding_strategy="tfidf" or "sentence_embeddings"
5. Categorical with cardinality < 10: "onehot". 10-50: "target_encode". 50+: "hash" or "frequency"
6. Numeric columns should generally get imputation_strategy="median" unless domain suggests otherwise
7. Only output valid JSON. No explanation text outside the array.

Domain-aware heuristics (apply these FIRST, they override generic rules):
8. Person name columns (Name, passenger_name, full_name, etc.): set text_processing_strategy="regex_extract", encoding_strategy="frequency". Names contain titles (Mr, Mrs, Dr, Master, Miss, Rev, etc.) that encode age, gender, and social status — regex extraction is critical.
9. Ticket/reference number columns (Ticket, ticket_no, booking_ref, etc.): set text_processing_strategy="regex_extract", encoding_strategy="frequency". Shared ticket numbers indicate traveling groups — regex_extract captures group size and numeric prefix signal.
10. Cabin/room/seat columns with high missing rate: missing values ARE informative (cabin recorded = higher class). Set imputation_strategy="fill_empty_string", text_processing_strategy="regex_extract" to extract deck letter and cabin number. Flag data_quality_flags=["high_missing"] but importance_prior="high" (NOT drop).
11. For survival/mortality/outcome prediction: sex/gender columns are importance_prior="high" with interaction_candidates including class/status columns. Age columns get interaction_candidates with class/fare/sex. Fare/price/cost columns get outlier_strategy="log_transform".
12. Family/group size indicators: columns like SibSp, Parch, siblings, parents, children — set interaction_candidates to EACH OTHER so FamilySize = SibSp+Parch+1 can be derived. importance_prior="high".
13. Embarked/port/origin/destination: low-cardinality categorical — set encoding_strategy="onehot", imputation_strategy="mode"."""

BATCH_SIZE = 15  # Max columns per LLM call to stay within token limits


class ColumnAIEnricher:
    """LLM-guided per-column semantic analysis and treatment planning."""

    def __init__(self, llm_client: Any, model_router: Any):
        self.llm_client = llm_client
        self.model_router = model_router

    def enrich_profiles(
        self,
        profiles: list[ColumnProfile],
        target_column: str,
        problem_type: str,
        sample_rows: list[dict[str, Any]],
    ) -> list[ColumnProfile]:
        """Enrich column profiles with LLM semantic analysis.

        Args:
            profiles: List of ColumnProfile objects from the profiler.
            target_column: Name of the target column.
            problem_type: 'classification' or 'regression'.
            sample_rows: 5-10 sample rows as list of dicts.

        Returns:
            Updated list of ColumnProfile objects with enriched fields.
        """
        # Filter out target column
        non_target_profiles = [p for p in profiles if not p.is_target and p.name != target_column]

        if not non_target_profiles:
            return profiles

        # Process in batches
        enrichments: dict[str, dict[str, Any]] = {}

        for i in range(0, len(non_target_profiles), BATCH_SIZE):
            batch = non_target_profiles[i:i + BATCH_SIZE]
            batch_enrichments = self._enrich_batch(
                batch, target_column, problem_type, sample_rows,
            )
            enrichments.update(batch_enrichments)

        # Apply enrichments to profiles
        for profile in profiles:
            if profile.name in enrichments:
                enrichment = enrichments[profile.name]
                profile.semantic_meaning = enrichment.get("semantic_meaning", "")
                profile.semantic_dtype = enrichment.get("semantic_dtype", "")
                profile.imputation_strategy = enrichment.get("imputation_strategy", "")
                profile.encoding_strategy = enrichment.get("encoding_strategy", "")
                profile.outlier_strategy = enrichment.get("outlier_strategy", "")
                profile.text_processing_strategy = enrichment.get("text_processing_strategy", "")
                profile.interaction_candidates = enrichment.get("interaction_candidates", [])
                profile.data_quality_flags = enrichment.get("data_quality_flags", [])
                profile.importance_prior = enrichment.get("importance_prior", "")

        return profiles

    def _enrich_batch(
        self,
        batch: list[ColumnProfile],
        target_column: str,
        problem_type: str,
        sample_rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Enrich a batch of columns via a single LLM call.

        Returns:
            Dict mapping column name -> enrichment dict.
        """
        # Build column descriptions for prompt
        col_descriptions = []
        for profile in batch:
            desc = (
                f"- {profile.name}: dtype={profile.dtype}, "
                f"cardinality={profile.cardinality}, "
                f"missing_pct={profile.missing_pct:.1f}%"
            )
            if profile.distribution_summary:
                stats = profile.distribution_summary
                if "mean" in stats:
                    desc += f", mean={stats['mean']:.3f}"
                if "std" in stats:
                    desc += f", std={stats['std']:.3f}"
                if "min" in stats:
                    desc += f", min={stats['min']}"
                if "max" in stats:
                    desc += f", max={stats['max']}"
            if profile.text_detected:
                desc += ", TEXT_DETECTED"
            col_descriptions.append(desc)

        # Sample values for context
        sample_text = ""
        if sample_rows:
            batch_col_names = {p.name for p in batch}
            for i, row in enumerate(sample_rows[:3]):
                filtered_row = {
                    k: v for k, v in row.items() if k in batch_col_names
                }
                # Truncate long values
                display_row = {}
                for k, v in filtered_row.items():
                    v_str = str(v)
                    display_row[k] = v_str[:100] if len(v_str) > 100 else v_str
                sample_text += f"  Row {i}: {display_row}\n"

        prompt = (
            f"Dataset: {problem_type} problem, target='{target_column}'\n\n"
            f"Columns to analyze:\n"
            f"{chr(10).join(col_descriptions)}\n\n"
        )
        if sample_text:
            prompt += f"Sample values:\n{sample_text}\n"
        prompt += "Output the JSON array with per-column analysis."

        try:
            models = self.model_router.get_model_chain("ds_analyst")
            messages = [
                LLMMessage(role="system", content=ENRICHMENT_SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ]

            response = self.llm_client.complete_with_fallback(
                messages=messages,
                models=models,
                max_tokens=4096,
                temperature=0.0,
            )

            return self._parse_enrichments(response.content, batch)

        except Exception as e:
            logger.warning(
                "LLM column enrichment failed for batch of %d columns: %s. "
                "Falling back to heuristic enrichment.",
                len(batch), e,
            )
            return self._heuristic_enrichment(batch)

    def _parse_enrichments(
        self,
        llm_output: str,
        batch: list[ColumnProfile],
    ) -> dict[str, dict[str, Any]]:
        """Parse LLM JSON output into enrichment dicts.

        Falls back to heuristic enrichment on parse failure.
        """
        # Extract JSON from LLM output (may have markdown fences)
        json_text = llm_output.strip()
        if "```" in json_text:
            # Extract content between code fences
            parts = json_text.split("```")
            for part in parts[1:]:
                # Skip the language identifier line
                lines = part.strip().split("\n")
                if lines[0].strip().lower() in ("json", ""):
                    json_text = "\n".join(lines[1:])
                else:
                    json_text = part
                break

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            # Try to find JSON array in the text
            start = json_text.find("[")
            end = json_text.rfind("]")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(json_text[start:end + 1])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM enrichment JSON")
                    return self._heuristic_enrichment(batch)
            else:
                logger.warning("No JSON array found in LLM enrichment output")
                return self._heuristic_enrichment(batch)

        if not isinstance(parsed, list):
            logger.warning("LLM enrichment output is not a list")
            return self._heuristic_enrichment(batch)

        # Build result dict
        batch_names = {p.name for p in batch}
        result: dict[str, dict[str, Any]] = {}

        valid_imputation = {
            "median", "mean", "mode", "knn", "mice",
            "forward_fill", "domain_zero", "fill_empty_string", "drop_row",
        }
        valid_encoding = {
            "keep_numeric", "onehot", "ordinal", "target_encode",
            "frequency", "hash", "sentence_embeddings", "tfidf",
        }
        valid_outlier = {
            "none", "clip_to_iqr", "winsorize", "log_transform", "remove", "keep",
        }
        valid_text = {
            "none", "tfidf", "embed_sentence_transformer",
            "llm_extract_entities", "regex_extract",
        }
        valid_importance = {"high", "medium", "low", "drop"}

        for item in parsed:
            if not isinstance(item, dict):
                continue
            col_name = item.get("column", "")
            if col_name not in batch_names:
                continue

            enrichment: dict[str, Any] = {
                "semantic_meaning": str(item.get("semantic_meaning", "")),
                "semantic_dtype": str(item.get("semantic_dtype", "")),
            }

            # Validate each field against allowed values
            imp = str(item.get("imputation_strategy", ""))
            enrichment["imputation_strategy"] = imp if imp in valid_imputation else "median"

            enc = str(item.get("encoding_strategy", ""))
            enrichment["encoding_strategy"] = enc if enc in valid_encoding else "keep_numeric"

            out = str(item.get("outlier_strategy", ""))
            enrichment["outlier_strategy"] = out if out in valid_outlier else "none"

            txt = str(item.get("text_processing_strategy", ""))
            enrichment["text_processing_strategy"] = txt if txt in valid_text else "none"

            imp_prior = str(item.get("importance_prior", ""))
            enrichment["importance_prior"] = imp_prior if imp_prior in valid_importance else "medium"

            # Lists
            interactions = item.get("interaction_candidates", [])
            enrichment["interaction_candidates"] = (
                [str(i) for i in interactions] if isinstance(interactions, list) else []
            )

            flags = item.get("data_quality_flags", [])
            enrichment["data_quality_flags"] = (
                [str(f) for f in flags] if isinstance(flags, list) else []
            )

            result[col_name] = enrichment

        # Fill any missing columns with heuristic
        for profile in batch:
            if profile.name not in result:
                heuristic = self._heuristic_enrichment([profile])
                result.update(heuristic)

        return result

    def _heuristic_enrichment(
        self,
        batch: list[ColumnProfile],
    ) -> dict[str, dict[str, Any]]:
        """Provide heuristic-based enrichment as fallback.

        Mirrors the logic from column_profiler._recommend_treatment()
        but produces structured per-column decisions.
        """
        result: dict[str, dict[str, Any]] = {}

        for profile in batch:
            enrichment: dict[str, Any] = {
                "semantic_meaning": "",
                "semantic_dtype": profile.dtype,
                "interaction_candidates": [],
                "data_quality_flags": [],
                "importance_prior": "medium",
            }

            # Data quality flags
            if profile.missing_pct > 50:
                enrichment["data_quality_flags"].append("high_missing")
            if profile.cardinality <= 1:
                enrichment["data_quality_flags"].append("constant")
                enrichment["importance_prior"] = "drop"

            # Check for ID-like columns
            name_lower = profile.name.lower()
            if any(id_word in name_lower for id_word in ("_id", "index", "row_num", "unnamed")):
                enrichment["data_quality_flags"].append("id_column")
                enrichment["importance_prior"] = "drop"

            # Per-dtype strategies
            if profile.dtype == "numeric":
                enrichment["imputation_strategy"] = "median"
                enrichment["encoding_strategy"] = "keep_numeric"
                enrichment["outlier_strategy"] = "clip_to_iqr"
                enrichment["text_processing_strategy"] = "none"
            elif profile.dtype == "categorical":
                enrichment["imputation_strategy"] = "mode"
                enrichment["outlier_strategy"] = "none"
                enrichment["text_processing_strategy"] = "none"
                if profile.cardinality < 10:
                    enrichment["encoding_strategy"] = "onehot"
                elif profile.cardinality < 50:
                    enrichment["encoding_strategy"] = "target_encode"
                else:
                    enrichment["encoding_strategy"] = "frequency"
            elif profile.dtype == "text" or profile.text_detected:
                enrichment["imputation_strategy"] = "fill_empty_string"
                enrichment["outlier_strategy"] = "none"
                # Domain-aware text processing
                if any(ind in name_lower for ind in ("name", "passenger", "person", "customer")):
                    enrichment["encoding_strategy"] = "frequency"
                    enrichment["text_processing_strategy"] = "regex_extract"
                    enrichment["importance_prior"] = "high"
                elif any(ind in name_lower for ind in ("cabin", "room", "berth", "seat")):
                    enrichment["encoding_strategy"] = "hash"
                    enrichment["text_processing_strategy"] = "regex_extract"
                    enrichment["importance_prior"] = "high"
                elif any(ind in name_lower for ind in ("ticket", "booking", "ref", "reservation")):
                    enrichment["encoding_strategy"] = "frequency"
                    enrichment["text_processing_strategy"] = "regex_extract"
                elif any(ind in name_lower for ind in ("email", "url", "phone", "address")):
                    enrichment["encoding_strategy"] = "frequency"
                    enrichment["text_processing_strategy"] = "regex_extract"
                else:
                    enrichment["encoding_strategy"] = "tfidf"
                    enrichment["text_processing_strategy"] = "tfidf"
            elif profile.dtype == "datetime":
                enrichment["imputation_strategy"] = "forward_fill"
                enrichment["encoding_strategy"] = "keep_numeric"
                enrichment["outlier_strategy"] = "none"
                enrichment["text_processing_strategy"] = "none"
            elif profile.dtype == "boolean":
                enrichment["imputation_strategy"] = "mode"
                enrichment["encoding_strategy"] = "keep_numeric"
                enrichment["outlier_strategy"] = "none"
                enrichment["text_processing_strategy"] = "none"
            else:
                enrichment["imputation_strategy"] = "median"
                enrichment["encoding_strategy"] = "keep_numeric"
                enrichment["outlier_strategy"] = "none"
                enrichment["text_processing_strategy"] = "none"

            result[profile.name] = enrichment

        return result
