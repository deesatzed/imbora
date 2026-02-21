# Data Audit Skill Procedure

## Purpose
Perform comprehensive data quality assessment on a tabular dataset before any modeling.

## Steps
1. Load the dataset (CSV, TSV, or Parquet)
2. Profile every column: detect dtype, cardinality, missing rate, distribution stats
3. Identify text columns for downstream embedding/clustering
4. Run label quality assessment (Cleanlab) on the target column
5. Compute dataset fingerprint for reproducibility tracking
6. Generate LLM semantic analysis: domain understanding, potential issues, feature relationships
7. Compile recommended actions (imputation strategies, encoding suggestions, drop candidates)

## Quality Gates
- All columns must be profiled with a detected dtype
- Missing percentage must be computed for every column
- Dataset fingerprint must be non-empty
- LLM semantic analysis must reference at least 3 specific columns by name
- Overall quality score must be in [0, 1]

## Output
DataAuditReport with: column_profiles, overall_quality_score, label_issues_count,
llm_semantic_analysis, recommended_actions, dataset_fingerprint
