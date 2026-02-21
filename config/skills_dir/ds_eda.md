# EDA Skill Procedure

## Purpose
Perform LLM-guided exploratory data analysis to uncover patterns, relationships, and anomalies.

## Steps
1. Generate analytical questions using LLM (QUIS pattern)
2. Run statistical analysis: target distribution, feature skewness, class imbalance
3. Compute feature-target correlations for all numeric columns
4. Detect outliers using IQR method
5. Analyze text columns (length, word count, uniqueness)
6. Synthesize all findings via LLM into actionable narrative
7. Generate recommendations for feature engineering

## Quality Gates
- At least 3 questions generated
- Statistical findings include target distribution analysis
- Correlation analysis covers all numeric features
- LLM synthesis references specific columns and values

## Output
EDAReport with: questions_generated, findings, correlations, outlier_summary,
text_column_analysis, llm_synthesis, recommendations
