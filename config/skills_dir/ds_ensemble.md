# Ensemble Skill Procedure

## Purpose
Build a stacked generalization ensemble from trained model candidates with uncertainty routing and Pareto-optimal configuration selection.

## Steps
1. Load dataset and prepare numeric feature matrix X and target vector y
2. Reconstruct ModelTrainingReport from input; verify at least 2 candidates available
3. Build Level-0 out-of-fold predictions for each base model via sklearn cross_val_predict
4. Stack Level-0 predictions into a meta-feature matrix
5. Train Level-1 meta-learner on stacked features:
   - Classification: LogisticRegression (max_iter=1000, solver=lbfgs)
   - Regression: Ridge (alpha=1.0)
6. Evaluate meta-learner with cross_val_score on stacked features
7. Compute prediction disagreement across Level-0 base models
8. Route samples by uncertainty threshold into confident and uncertain paths
9. Run Pareto optimization across configured objectives (accuracy, interpretability, speed)
   - Add each base model and the stacked ensemble as solutions
   - Compute Pareto front and select knee point
10. Call LLM to analyze ensemble for overfitting risk and diversity

## Quality Gates
- At least 2 base models produced valid OOF predictions
- Meta-learner CV score computed successfully
- Prediction disagreement computed for all samples
- Pareto front contains at least 1 solution
- LLM analysis references meta-learner score and routing percentages

## Output
EnsembleReport with: stacking_architecture, routing_rules, pareto_front,
selected_configuration, llm_ensemble_analysis
