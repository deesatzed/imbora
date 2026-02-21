# Deployment Skill Procedure

## Purpose
Generate a complete deployment package with serialized model, API scaffold, monitoring configuration, data contract, and LLM completeness review.

## Steps
1. Load dataset and prepare numeric feature matrix X and target vector y
2. Identify the best model candidate from the training report
3. Train the final model on full data (all rows, no cross-validation hold-out):
   - Use _build_sklearn_estimator pattern matching model_training.py
   - Fall back to GradientBoosting if the original model package is unavailable
4. Serialize the trained model with joblib:
   - Save to artifacts_dir/project_id/model.joblib
   - Create parent directories as needed
5. Generate FastAPI scaffold code (as a string, not written to disk):
   - Include /predict POST endpoint with Pydantic input validation
   - Include /health GET endpoint for liveness check
   - Include model loading at startup
   - List all input feature names as typed fields
6. Compute per-feature training statistics (mean, std, min, max, median, Q25, Q75)
7. Generate monitoring configuration:
   - Per-feature drift thresholds: mean shift > 2*std, range expansion > 50%
   - Performance thresholds: alert at 10% drop, critical at 20% drop
   - Include check interval, window size, drift test method
8. Generate data contract:
   - List all input features with pandas dtype mapped to contract type
   - Include required/nullable constraints
   - Specify minimum features required
9. Call LLM with ds_evaluator role for completeness review:
   - Summarize model artifact, API scaffold, monitoring config, data contract
   - Include evaluation grade and predictive scores
   - Request completeness score and missing element identification

## Quality Gates
- Final model trained on full data without errors
- Model artifact saved as valid joblib file at expected path
- API scaffold includes all required endpoints (/predict, /health)
- Monitoring config has drift thresholds for all numeric features
- Data contract documents all input features with correct types
- LLM completeness review references specific package components
- All artifacts manifest recorded in experiment metadata

## Output
DeploymentPackage with: model_artifact_path, feature_pipeline_path,
api_scaffold_code, monitoring_config, data_contract, llm_completeness_review
