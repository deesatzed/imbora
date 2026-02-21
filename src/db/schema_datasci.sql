-- The Associate — Data Science Module Schema Extension
-- Idempotent: safe to run multiple times (IF NOT EXISTS guards)

-- Add task_type column to tasks table for DS task routing
ALTER TABLE tasks ADD COLUMN IF NOT EXISTS task_type TEXT NOT NULL DEFAULT 'general';

-- ds_experiments: track model configs + metrics across DS pipeline phases
CREATE TABLE IF NOT EXISTS ds_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    task_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    run_id UUID REFERENCES sotappr_runs(id) ON DELETE SET NULL,
    experiment_phase TEXT NOT NULL,
    experiment_config JSONB NOT NULL DEFAULT '{}',
    metrics JSONB NOT NULL DEFAULT '{}',
    artifacts_manifest JSONB NOT NULL DEFAULT '{}',
    dataset_fingerprint TEXT,
    parent_experiment_id UUID REFERENCES ds_experiments(id),
    status TEXT NOT NULL DEFAULT 'RUNNING'
        CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
    created_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ds_experiments_project
    ON ds_experiments(project_id);
CREATE INDEX IF NOT EXISTS idx_ds_experiments_task
    ON ds_experiments(task_id);
CREATE INDEX IF NOT EXISTS idx_ds_experiments_phase
    ON ds_experiments(experiment_phase);

-- ds_artifacts: store model files, feature pipelines, reports
CREATE TABLE IF NOT EXISTS ds_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES ds_experiments(id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    artifact_metadata JSONB NOT NULL DEFAULT '{}',
    size_bytes BIGINT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ds_artifacts_experiment
    ON ds_artifacts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_ds_artifacts_type
    ON ds_artifacts(artifact_type);
