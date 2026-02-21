"""Per-column treatment execution engine.

Executes structured per-column imputation, encoding, outlier treatment,
and text processing strategies based on enriched ColumnProfile decisions.

Replaces the previous one-size-fits-all approach (median imputation,
select_dtypes(include=["number"])) with per-column strategies determined
by LLM semantic analysis.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.datasci.models import ColumnProfile

logger = logging.getLogger("associate.datasci.column_treatment_executor")


class ColumnTreatmentExecutor:
    """Execute per-column imputation, encoding, outlier treatment, and text processing."""

    def execute(
        self,
        df: pd.DataFrame,
        profiles: list[ColumnProfile],
        target_column: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Apply all per-column treatments.

        Args:
            df: Input DataFrame.
            profiles: Enriched ColumnProfile list (with imputation_strategy, etc.).
            target_column: Target column name (not treated).

        Returns:
            Tuple of (transformed DataFrame, list of new feature names).
        """
        df = df.copy()
        new_features: list[str] = []

        # Build lookup by column name
        profile_map: dict[str, ColumnProfile] = {p.name: p for p in profiles}

        for col in df.columns:
            if col == target_column:
                continue

            profile = profile_map.get(col)
            if profile is None:
                continue

            # Skip columns marked for dropping
            if profile.importance_prior == "drop":
                logger.info("Dropping column '%s' (importance_prior=drop)", col)
                df = df.drop(columns=[col])
                continue

            # Apply imputation
            if profile.imputation_strategy and col in df.columns:
                df = self._apply_imputation(df, col, profile.imputation_strategy)

            # Apply outlier treatment (before encoding, on numeric)
            if profile.outlier_strategy and profile.outlier_strategy != "none" and col in df.columns:
                df = self._apply_outlier_treatment(df, col, profile.outlier_strategy)

            # Apply text processing BEFORE encoding — text processing extracts
            # features from the original column (titles from names, deck from cabin)
            # and may drop the original. Encoding runs on whatever remains.
            if (
                profile.text_processing_strategy
                and profile.text_processing_strategy != "none"
                and col in df.columns
            ):
                df, text_features = self._apply_text_processing(
                    df, col, profile.text_processing_strategy,
                )
                new_features.extend(text_features)

            # Apply encoding (only if column still exists after text processing)
            if profile.encoding_strategy and col in df.columns:
                df, encoding_features = self._apply_encoding(
                    df, col, profile.encoding_strategy, target_column,
                )
                new_features.extend(encoding_features)

        # Apply interaction features
        interaction_features = self._apply_interactions(df, profiles, target_column)
        if interaction_features:
            for feat_name, feat_series in interaction_features.items():
                df[feat_name] = feat_series
                new_features.append(feat_name)

        return df, new_features

    def _apply_imputation(
        self, df: pd.DataFrame, col: str, strategy: str,
    ) -> pd.DataFrame:
        """Apply per-column imputation strategy."""
        series = df[col]

        if not series.isna().any():
            return df  # No missing values, skip

        try:
            if strategy == "median":
                if pd.api.types.is_numeric_dtype(series):
                    df[col] = series.fillna(series.median())
                else:
                    df[col] = series.fillna(series.mode().iloc[0] if len(series.mode()) > 0 else "")
            elif strategy == "mean":
                if pd.api.types.is_numeric_dtype(series):
                    df[col] = series.fillna(series.mean())
                else:
                    df[col] = series.fillna(series.mode().iloc[0] if len(series.mode()) > 0 else "")
            elif strategy == "mode":
                mode_vals = series.mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else 0
                df[col] = series.fillna(fill_val)
            elif strategy == "knn":
                df = self._impute_knn(df, col)
            elif strategy == "mice":
                df = self._impute_mice(df, col)
            elif strategy == "forward_fill":
                df[col] = series.ffill()
                # Fill any remaining NaN at the start with backfill
                df[col] = df[col].bfill()
            elif strategy == "domain_zero":
                df[col] = series.fillna(0)
            elif strategy == "fill_empty_string":
                df[col] = series.fillna("")
            elif strategy == "drop_row":
                # Don't actually drop rows here — just fill with median
                # Dropping rows could misalign with target vector
                if pd.api.types.is_numeric_dtype(series):
                    df[col] = series.fillna(series.median())
                else:
                    df[col] = series.fillna("")
            else:
                # Default: median for numeric, mode for others
                if pd.api.types.is_numeric_dtype(series):
                    df[col] = series.fillna(series.median())
                else:
                    mode_vals = series.mode()
                    fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else ""
                    df[col] = series.fillna(fill_val)

        except Exception as e:
            logger.warning("Imputation failed for '%s' with strategy '%s': %s", col, strategy, e)
            # Fallback: fill with 0 for numeric, empty string for others
            if pd.api.types.is_numeric_dtype(series):
                df[col] = series.fillna(0)
            else:
                df[col] = series.fillna("")

        return df

    def _impute_knn(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Apply KNN imputation for a column."""
        try:
            from sklearn.impute import KNNImputer

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if col not in numeric_cols:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "")
                return df

            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        except ImportError:
            logger.info("sklearn KNNImputer not available; falling back to median")
            df[col] = df[col].fillna(df[col].median())
        except Exception as e:
            logger.warning("KNN imputation failed for '%s': %s", col, e)
            df[col] = df[col].fillna(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0)

        return df

    def _impute_mice(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Apply MICE (IterativeImputer) for a column."""
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if col not in numeric_cols:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "")
                return df

            imputer = IterativeImputer(max_iter=10, random_state=42)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        except ImportError:
            logger.info("IterativeImputer not available; falling back to median")
            df[col] = df[col].fillna(df[col].median())
        except Exception as e:
            logger.warning("MICE imputation failed for '%s': %s", col, e)
            df[col] = df[col].fillna(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0)

        return df

    def _apply_outlier_treatment(
        self, df: pd.DataFrame, col: str, strategy: str,
    ) -> pd.DataFrame:
        """Apply per-column outlier treatment."""
        if not pd.api.types.is_numeric_dtype(df[col]):
            return df

        series = df[col].dropna()
        if len(series) < 4:
            return df

        try:
            if strategy == "clip_to_iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower=lower, upper=upper)

            elif strategy == "winsorize":
                try:
                    from scipy.stats.mstats import winsorize as scipy_winsorize
                    # Winsorize at 5th/95th percentile
                    winsorized = scipy_winsorize(df[col].fillna(0).values, limits=[0.05, 0.05])
                    df[col] = pd.Series(winsorized, index=df.index)
                except ImportError:
                    # Fallback: manual clip at 5th/95th
                    lower = series.quantile(0.05)
                    upper = series.quantile(0.95)
                    df[col] = df[col].clip(lower=lower, upper=upper)

            elif strategy == "log_transform":
                # Apply log1p for right-skewed data (preserves zeros)
                min_val = df[col].min()
                if min_val >= 0:
                    df[col] = np.log1p(df[col])
                else:
                    # Shift to positive before log
                    df[col] = np.log1p(df[col] - min_val)

            elif strategy == "remove":
                # Set outlier values to NaN (will be imputed later)
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (df[col] < lower) | (df[col] > upper)
                df.loc[mask, col] = np.nan
                # Re-impute with median
                df[col] = df[col].fillna(series.median())

            # 'keep' and 'none': do nothing

        except Exception as e:
            logger.warning("Outlier treatment failed for '%s': %s", col, e)

        return df

    def _apply_encoding(
        self,
        df: pd.DataFrame,
        col: str,
        strategy: str,
        target_column: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Apply per-column encoding strategy.

        Returns:
            Tuple of (modified DataFrame, list of new feature names).
        """
        new_features: list[str] = []

        if strategy == "keep_numeric":
            # Already numeric, no encoding needed
            return df, new_features

        try:
            if strategy == "onehot":
                dummies = pd.get_dummies(df[[col]], columns=[col], drop_first=True, dtype=float)
                new_cols = [c for c in dummies.columns if c not in df.columns]
                for c in new_cols:
                    df[c] = dummies[c]
                    new_features.append(c)
                df = df.drop(columns=[col])

            elif strategy == "ordinal":
                try:
                    from sklearn.preprocessing import OrdinalEncoder
                    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    df[col] = encoder.fit_transform(df[[col]].astype(str))
                except ImportError:
                    # Fallback: factorize
                    df[col] = pd.factorize(df[col])[0].astype(float)

            elif strategy == "target_encode":
                try:
                    from sklearn.preprocessing import TargetEncoder
                    if target_column in df.columns:
                        encoder = TargetEncoder(random_state=42)
                        target = df[target_column]
                        df[col] = encoder.fit_transform(
                            df[[col]].astype(str), target,
                        )
                    else:
                        # Fallback to frequency encoding
                        freq = df[col].value_counts(normalize=True)
                        df[col] = df[col].map(freq).fillna(0)
                except ImportError:
                    # Fallback to frequency encoding
                    freq = df[col].value_counts(normalize=True)
                    df[col] = df[col].map(freq).fillna(0)

            elif strategy == "frequency":
                freq = df[col].value_counts(normalize=True)
                new_col = f"{col}_freq"
                df[new_col] = df[col].map(freq).fillna(0)
                new_features.append(new_col)
                df = df.drop(columns=[col])

            elif strategy == "hash":
                # Hashing trick for very high cardinality
                n_features = 32
                new_col = f"{col}_hash"
                df[new_col] = df[col].apply(
                    lambda x: hash(str(x)) % n_features,
                ).astype(float)
                new_features.append(new_col)
                df = df.drop(columns=[col])

            elif strategy in ("sentence_embeddings", "tfidf"):
                # Text encoding handled by _apply_text_processing
                pass

        except Exception as e:
            logger.warning("Encoding failed for '%s' with strategy '%s': %s", col, strategy, e)

        return df, new_features

    def _apply_text_processing(
        self,
        df: pd.DataFrame,
        col: str,
        strategy: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Apply per-column text processing strategy.

        Returns:
            Tuple of (modified DataFrame, list of new feature names).
        """
        new_features: list[str] = []

        if strategy == "none":
            return df, new_features

        text_series = df[col].fillna("").astype(str)

        try:
            if strategy == "tfidf":
                from sklearn.feature_extraction.text import TfidfVectorizer

                max_features = 50  # Cap to prevent feature explosion
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words="english",
                    max_df=0.95,
                    min_df=2,
                )

                # Need at least 2 unique documents
                if text_series.nunique() < 2:
                    return df, new_features

                tfidf_matrix = vectorizer.fit_transform(text_series)
                feature_names = vectorizer.get_feature_names_out()

                for i, fname in enumerate(feature_names):
                    new_col = f"{col}_tfidf_{fname}"
                    df[new_col] = tfidf_matrix[:, i].toarray().ravel()
                    new_features.append(new_col)

                # Drop original text column after TF-IDF
                df = df.drop(columns=[col])

            elif strategy == "embed_sentence_transformer":
                try:
                    from sentence_transformers import SentenceTransformer

                    # Truncate text to 512 chars for embedding
                    truncated = text_series.str[:512].tolist()

                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    embeddings = model.encode(
                        truncated, show_progress_bar=False, batch_size=100,
                    )

                    for i in range(embeddings.shape[1]):
                        new_col = f"{col}_emb_{i}"
                        df[new_col] = embeddings[:, i]
                        new_features.append(new_col)

                    df = df.drop(columns=[col])

                except ImportError:
                    logger.info(
                        "sentence-transformers not available; "
                        "falling back to TF-IDF for '%s'", col,
                    )
                    return self._apply_text_processing(df, col, "tfidf")

            elif strategy == "regex_extract":
                col_lower = col.lower()

                # Detect name-like columns → extract titles
                name_indicators = ("name", "passenger", "person", "full_name", "customer_name")
                if any(ind in col_lower for ind in name_indicators):
                    # Title extraction: "Braund, Mr. Owen Harris" → "Mr"
                    title_col = f"{col}_title"
                    df[title_col] = text_series.str.extract(
                        r",\s*([A-Za-z]+)\.", expand=False,
                    ).fillna("Unknown")
                    # Map rare titles to common groups
                    title_map = {
                        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs",
                        "Master": "Master", "Dr": "Rare", "Rev": "Rare",
                        "Col": "Rare", "Major": "Rare", "Mlle": "Miss",
                        "Mme": "Mrs", "Ms": "Miss", "Lady": "Rare",
                        "Sir": "Rare", "Capt": "Rare", "Don": "Rare",
                        "Dona": "Rare", "Countess": "Rare", "Jonkheer": "Rare",
                    }
                    df[title_col] = df[title_col].map(title_map).fillna("Rare")
                    # Frequency-encode the title
                    title_freq = df[title_col].value_counts(normalize=True)
                    title_freq_col = f"{col}_title_freq"
                    df[title_freq_col] = df[title_col].map(title_freq).fillna(0)
                    new_features.append(title_freq_col)
                    # Also one-hot for the major categories
                    for t in ["Mr", "Miss", "Mrs", "Master"]:
                        ohe_col = f"{col}_is_{t}"
                        df[ohe_col] = (df[title_col] == t).astype(float)
                        new_features.append(ohe_col)
                    df = df.drop(columns=[title_col])

                # Detect cabin-like columns → extract deck letter + has_cabin
                elif any(ind in col_lower for ind in ("cabin", "room", "berth", "seat")):
                    deck_col = f"{col}_deck"
                    df[deck_col] = text_series.apply(
                        lambda x: x[0].upper() if x and len(x) > 0 and x[0].isalpha() else "U",
                    )
                    deck_freq = df[deck_col].value_counts(normalize=True)
                    deck_freq_col = f"{col}_deck_freq"
                    df[deck_freq_col] = df[deck_col].map(deck_freq).fillna(0)
                    new_features.append(deck_freq_col)

                    has_col = f"{col}_has"
                    df[has_col] = text_series.apply(
                        lambda x: 0.0 if (not x or x.strip() == "" or x == "nan") else 1.0,
                    )
                    new_features.append(has_col)
                    df = df.drop(columns=[deck_col])

                # Detect ticket-like columns → group size
                elif any(ind in col_lower for ind in ("ticket", "booking", "ref", "reservation")):
                    group_col = f"{col}_group_size"
                    ticket_counts = text_series.value_counts()
                    df[group_col] = text_series.map(ticket_counts).fillna(1).astype(float)
                    new_features.append(group_col)

                    # Also extract numeric prefix if present
                    num_col = f"{col}_num_prefix"
                    df[num_col] = text_series.str.extract(r"(\d+)", expand=False).fillna("0").astype(float)
                    new_features.append(num_col)

                else:
                    # Generic regex extraction for other text columns
                    num_col = f"{col}_num_count"
                    df[num_col] = text_series.str.count(r"\d+")
                    new_features.append(num_col)

                    len_col = f"{col}_char_len"
                    df[len_col] = text_series.str.len()
                    new_features.append(len_col)

                    word_col = f"{col}_word_count"
                    df[word_col] = text_series.str.split().str.len().fillna(0)
                    new_features.append(word_col)

                df = df.drop(columns=[col])

            elif strategy == "llm_extract_entities":
                # Fall back to regex_extract (LLM entity extraction
                # would need the OpenRouter client which is not available here)
                return self._apply_text_processing(df, col, "regex_extract")

        except Exception as e:
            logger.warning(
                "Text processing failed for '%s' with strategy '%s': %s",
                col, strategy, e,
            )
            # Fallback: drop the text column
            if col in df.columns:
                df = df.drop(columns=[col])

        return df, new_features

    def _apply_interactions(
        self,
        df: pd.DataFrame,
        profiles: list[ColumnProfile],
        target_column: str,
    ) -> dict[str, pd.Series]:
        """Create pairwise interaction features based on profile hints.

        Returns:
            Dict mapping new feature name -> Series.
        """
        interaction_features: dict[str, pd.Series] = {}

        for profile in profiles:
            if (
                profile.name == target_column
                or not profile.interaction_candidates
                or profile.importance_prior == "drop"
                or profile.name not in df.columns
            ):
                continue

            for partner in profile.interaction_candidates:
                if partner not in df.columns or partner == target_column:
                    continue

                # Only create interaction if both columns are numeric
                if (
                    pd.api.types.is_numeric_dtype(df[profile.name])
                    and pd.api.types.is_numeric_dtype(df[partner])
                ):
                    # Avoid duplicates (A*B == B*A)
                    pair_key = tuple(sorted([profile.name, partner]))
                    feat_name = f"{pair_key[0]}_x_{pair_key[1]}"

                    if feat_name not in interaction_features:
                        interaction_features[feat_name] = (
                            df[profile.name].fillna(0) * df[partner].fillna(0)
                        )

        return interaction_features
