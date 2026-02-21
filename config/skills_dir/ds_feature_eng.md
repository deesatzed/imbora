# Feature Engineering Skill Procedure

## Purpose
Apply LLM-guided evolutionary feature engineering (LLM-FE) plus deterministic per-column transforms.

## Steps
1. Compute baseline CV score on original features
2. Apply deterministic per-column transforms:
   - Numeric: log transform for positive columns
   - Categorical: frequency encoding
   - Datetime: extract year, month, day-of-week
3. Run LLM-FE evolutionary loop for configured generations:
   - LLM proposes transforms as Python functions
   - Execute transforms safely on data copy
   - CV-evaluate each transform
   - Keep winners, mutate/recombine for next generation
4. Rank all features by importance (gradient boosting feature_importances_)
5. Identify and drop near-zero importance features
6. Compile transformation code for reproducibility

## Quality Gates
- Baseline score computed successfully
- At least 1 generation of LLM-FE completed
- Feature ranking covers all numeric columns
- Best generation score >= baseline score

## Output
FeatureEngineeringReport with: original_feature_count, new_features, feature_ranking,
llm_fe_generations_run, best_generation_score, transformation_code, dropped_features
