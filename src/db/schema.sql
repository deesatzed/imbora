-- The Associate — Database Schema
-- PostgreSQL with pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;

-- 1. PROJECTS
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    tech_stack JSONB NOT NULL DEFAULT '{}',
    project_rules TEXT,
    banned_dependencies TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 2. TASKS (Work Queue)
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING','RESEARCHING','CODING','REVIEWING','STUCK','DONE')),
    priority INTEGER NOT NULL DEFAULT 0,
    context_snapshot_id UUID,
    attempt_count INTEGER DEFAULT 0,
    council_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(project_id, priority DESC);

-- 3. HYPOTHESIS_LOG (Trial & Error Memory)
CREATE TABLE IF NOT EXISTS hypothesis_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    approach_summary TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'FAILURE'
        CHECK (outcome IN ('SUCCESS','FAILURE')),
    error_signature TEXT,
    error_full TEXT,
    files_changed TEXT[],
    duration_seconds REAL,
    model_used TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(task_id, attempt_number)
);
CREATE INDEX IF NOT EXISTS idx_hyp_task ON hypothesis_log(task_id);
CREATE INDEX IF NOT EXISTS idx_hyp_error_sig ON hypothesis_log(error_signature);

-- 4. METHODOLOGIES (Long-Term Memory / RAG)
CREATE TABLE IF NOT EXISTS methodologies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_description TEXT NOT NULL,
    problem_embedding vector(384),
    solution_code TEXT NOT NULL,
    methodology_notes TEXT,
    source_task_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    tags TEXT[] DEFAULT '{}',
    language TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Item 2: Cross-project methodology scope
ALTER TABLE methodologies ADD COLUMN IF NOT EXISTS scope TEXT NOT NULL DEFAULT 'project';
-- Item 3: Typed methodology taxonomy
ALTER TABLE methodologies ADD COLUMN IF NOT EXISTS methodology_type TEXT;
-- Item 4: Files affected by this methodology's solution
ALTER TABLE methodologies ADD COLUMN IF NOT EXISTS files_affected TEXT[] DEFAULT '{}';

-- Full-text search via generated tsvector column
ALTER TABLE methodologies ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', problem_description || ' ' || COALESCE(methodology_notes, ''))
    ) STORED;
CREATE INDEX IF NOT EXISTS idx_meth_fts ON methodologies USING gin(search_vector);
CREATE INDEX IF NOT EXISTS idx_meth_scope ON methodologies(scope);

-- Vector similarity index (IVFFlat) — created after initial data load for best performance
-- For small datasets (<1000 rows), sequential scan is faster anyway.
-- Uncomment when the table has enough rows:
-- CREATE INDEX IF NOT EXISTS idx_meth_embedding ON methodologies
--     USING ivfflat (problem_embedding vector_cosine_ops) WITH (lists = 100);

-- 5. PEER_REVIEWS (Council Diagnoses)
CREATE TABLE IF NOT EXISTS peer_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    model_used TEXT NOT NULL,
    diagnosis TEXT NOT NULL,
    recommended_approach TEXT,
    reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_peer_task ON peer_reviews(task_id);

-- 6. CONTEXT_SNAPSHOTS (Checkpoint/Rewind State)
CREATE TABLE IF NOT EXISTS context_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    git_ref TEXT NOT NULL,
    file_manifest JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_snap_task ON context_snapshots(task_id);

-- 7. SOTAPPR RUNS (Phase artifacts + execution lifecycle)
CREATE TABLE IF NOT EXISTS sotappr_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    mode TEXT NOT NULL DEFAULT 'execute'
        CHECK (mode IN ('build','dry-run','execute','resume')),
    status TEXT NOT NULL DEFAULT 'planned'
        CHECK (status IN ('planned','seeded','running','paused','completed','failed')),
    governance_pack TEXT NOT NULL DEFAULT 'balanced',
    spec_json JSONB NOT NULL,
    report_json JSONB NOT NULL,
    report_path TEXT,
    repo_path TEXT NOT NULL,
    tasks_seeded INTEGER NOT NULL DEFAULT 0,
    tasks_processed INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    stop_reason TEXT,
    estimated_cost_usd DOUBLE PRECISION,
    elapsed_hours DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);
ALTER TABLE sotappr_runs ADD COLUMN IF NOT EXISTS stop_reason TEXT;
ALTER TABLE sotappr_runs ADD COLUMN IF NOT EXISTS estimated_cost_usd DOUBLE PRECISION;
ALTER TABLE sotappr_runs ADD COLUMN IF NOT EXISTS elapsed_hours DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS idx_sotappr_runs_project ON sotappr_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_sotappr_runs_status ON sotappr_runs(status);
CREATE INDEX IF NOT EXISTS idx_sotappr_runs_created ON sotappr_runs(created_at DESC);

-- 8. SOTAPPR ARTIFACTS (Phase/event/replay payloads)
CREATE TABLE IF NOT EXISTS sotappr_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES sotappr_runs(id) ON DELETE CASCADE,
    phase INTEGER NOT NULL,
    artifact_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sotappr_artifacts_run ON sotappr_artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_sotappr_artifacts_phase ON sotappr_artifacts(phase);

-- 8b. AGENT PACKETS (APC/1.0 first-class persistence)
CREATE TABLE IF NOT EXISTS agent_packets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES sotappr_runs(id) ON DELETE CASCADE,
    task_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    attempt_number INTEGER NOT NULL DEFAULT 0,
    packet_id UUID NOT NULL UNIQUE,
    root_packet_id UUID,
    parent_packet_id UUID,
    packet_type TEXT NOT NULL,
    channel TEXT NOT NULL,
    run_phase TEXT NOT NULL,
    sender_agent_id TEXT,
    sender_role TEXT,
    recipients_json JSONB NOT NULL DEFAULT '[]',
    packet_json JSONB NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}',
    packet_hash TEXT,
    confidence DOUBLE PRECISION,
    trace_id TEXT,
    lifecycle_state TEXT,
    lifecycle_history JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_agent_packets_run ON agent_packets(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_packets_task ON agent_packets(task_id);
CREATE INDEX IF NOT EXISTS idx_agent_packets_type ON agent_packets(packet_type);
CREATE INDEX IF NOT EXISTS idx_agent_packets_phase ON agent_packets(run_phase);
CREATE INDEX IF NOT EXISTS idx_agent_packets_trace ON agent_packets(trace_id);
CREATE INDEX IF NOT EXISTS idx_agent_packets_created ON agent_packets(created_at DESC);

-- 8c. PACKET EVENTS (state machine transitions)
CREATE TABLE IF NOT EXISTS packet_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES sotappr_runs(id) ON DELETE CASCADE,
    task_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    packet_row_id UUID REFERENCES agent_packets(id) ON DELETE CASCADE,
    packet_id UUID NOT NULL,
    event TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    occurred_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_packet_events_packet_id ON packet_events(packet_id);
CREATE INDEX IF NOT EXISTS idx_packet_events_run ON packet_events(run_id);
CREATE INDEX IF NOT EXISTS idx_packet_events_task ON packet_events(task_id);
CREATE INDEX IF NOT EXISTS idx_packet_events_occurred_at ON packet_events(occurred_at DESC);

-- 8d. PACKET LINEAGE (transfer/protocol exchange metadata)
CREATE TABLE IF NOT EXISTS packet_lineage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    packet_row_id UUID NOT NULL REFERENCES agent_packets(id) ON DELETE CASCADE,
    packet_id UUID NOT NULL UNIQUE,
    protocol_id TEXT,
    parent_protocol_ids TEXT[] DEFAULT '{}',
    ancestor_swarms TEXT[] DEFAULT '{}',
    cross_use_case BOOLEAN NOT NULL DEFAULT FALSE,
    transfer_mode TEXT,
    lineage_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_packet_lineage_protocol ON packet_lineage(protocol_id);
CREATE INDEX IF NOT EXISTS idx_packet_lineage_packet ON packet_lineage(packet_id);

-- 9. TOKEN_COSTS (Item 1 — per-call LLM cost tracking)
CREATE TABLE IF NOT EXISTS token_costs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE SET NULL,
    run_id UUID REFERENCES sotappr_runs(id) ON DELETE SET NULL,
    agent_role TEXT NOT NULL DEFAULT '',
    model_used TEXT NOT NULL DEFAULT '',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_token_costs_task ON token_costs(task_id);
CREATE INDEX IF NOT EXISTS idx_token_costs_run ON token_costs(run_id);
CREATE INDEX IF NOT EXISTS idx_token_costs_created ON token_costs(created_at DESC);
