# Model Training Skill Procedure

## Purpose
Train scale-appropriate model candidates with cross-validation, conformal calibration, and LLM behavioral analysis.

## Steps
1. Load dataset and prepare numeric feature matrix X and target vector y
2. Determine scale tier (small/medium/large) via row count thresholds
3. Select model candidates using scale router (TabPFN, TabICL, XGBoost, CatBoost, LightGBM, NAM)
4. For each candidate, run sklearn cross_val_score with configured CV folds
5. Measure per-candidate training time and single-sample prediction latency
6. Fit each candidate on full data, then run conformal prediction calibration:
   - Generate out-of-fold predictions via cross_val_predict
   - Calibrate ConformalPredictor on residuals
   - Evaluate empirical coverage against target (1 - alpha)
7. Extract NAM shape functions if NAM was trained
8. Call LLM for behavioral analysis of all candidate results
9. Identify best candidate by mean CV score

## Quality Gates
- At least one model candidate trained successfully
- All CV scores arrays have length equal to configured cv_folds
- Conformal coverage evaluated for each candidate with a fitted estimator
- LLM behavioral analysis references specific model names and scores
- Best candidate identified

## Output
ModelTrainingReport with: scale_tier, candidates (list of ModelCandidate), best_candidate,
nam_shape_functions, llm_evaluation_narrative
