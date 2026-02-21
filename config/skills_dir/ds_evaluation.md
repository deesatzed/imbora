# Evaluation Skill Procedure

## Purpose
Multi-dimensional model evaluation across predictive performance, robustness, fairness, interpretability, and baseline comparison with LLM narrative synthesis.

## Steps
1. Load dataset and prepare numeric feature matrix X and target vector y
2. Identify the best model candidate from the training report
3. Train the best model on full data for evaluation; fall back to GradientBoosting if the original model package is unavailable
4. Compute predictive scores:
   - Classification: accuracy, F1 (macro), AUC (via predict_proba if available)
   - Regression: R2, MAE, RMSE
5. Compute robustness scores:
   - Add Gaussian noise with std = 0.1 * per-feature std
   - Measure score drop (absolute and relative percentage)
   - Use accuracy for classification, R2 for regression as primary metric
6. Compute fairness scores (if sensitive_columns provided):
   - Per-group accuracy/R2 using pandas groupby on each sensitive column
   - Compute fairness gap (max group score - min group score) per column
   - Skip groups with fewer than 5 samples
7. Compute interpretability scores:
   - Extract feature importances (feature_importances_, coef_, or GradientBoosting fallback)
   - Calculate Gini coefficient of importances (concentration measure)
   - Count features needed for 80% cumulative importance
8. Compute feature importance ranking using GradientBoosting feature_importances_
9. Compute baseline comparison:
   - Classification: vs majority class predictor
   - Regression: vs mean value predictor
   - Calculate lift over baseline (absolute and relative percentage)
10. Call LLM with ds_evaluator role for evaluation narrative covering all dimensions
11. Compute overall letter grade (A/B/C/D/F) from weighted dimension scores:
    - Predictive: 40%, Robustness: 25%, Fairness: 15%, Interpretability: 20%
    - Fairness weight redistributed to predictive if no sensitive columns
12. Generate actionable recommendations based on evaluation gaps

## Quality Gates
- Best model trained successfully on full data
- All predictive metrics computed with real sklearn metrics
- Robustness evaluation uses actual noise injection on real features
- Fairness evaluation covers all specified sensitive columns
- Feature importance computed for all features
- Baseline comparison computed against appropriate naive predictor
- LLM narrative references specific evaluation scores
- Overall grade assigned based on weighted dimension scoring
- At least one recommendation generated

## Output
EvaluationReport with: predictive_scores, robustness_scores, fairness_scores,
interpretability_scores, feature_importance, baseline_comparison,
llm_evaluation_narrative, overall_grade, recommendations
