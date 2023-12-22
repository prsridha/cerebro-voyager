-- Seed
CREATE TABLE IF NOT EXISTS seed (
    id INTEGER PRIMARY KEY,
    seed_val INTEGER DEFAULT 0
);

-- Error
CREATE TABLE IF NOT EXISTS debug_errors (
    id INTEGER PRIMARY KEY,
    error_message TEXT DEFAULT ""
);

-- Restarts
CREATE TABLE IF NOT EXISTS restarts (
    worker_id INTEGER PRIMARY KEY,
    restart_count INTEGER DEFAULT 0
);

-- Dataset Locators
CREATE TABLE IF NOT EXISTS dataset_locators (
    id INTEGER PRIMARY KEY,
    params TEXT DEFAULT "{}"
);

-- ETL Task
CREATE TABLE IF NOT EXISTS etl_task (
    id INTEGER PRIMARY KEY,
    task TEXT DEFAULT "",
    mode TEXT DEFAULT ""
);

-- ETL Spec
CREATE TABLE IF NOT EXISTS etl_spec (
    id INTEGER PRIMARY KEY,
    spec_str TEXT DEFAULT ""
);

-- ETL Worker Status
CREATE TABLE IF NOT EXISTS etl_worker_status (
    worker_id INTEGER PRIMARY KEY,
    worker_status TEXT DEFAULT ""
);

-- ETL Worker Progress
CREATE TABLE IF NOT EXISTS etl_worker_progress (
    worker_id INTEGER PRIMARY KEY,
    worker_progress FLOAT DEFAULT 0.0
);

-- MOP Spec
CREATE TABLE IF NOT EXISTS etl_worker_progress (
    worker_id INTEGER PRIMARY KEY,
    worker_progress FLOAT DEFAULT 0.0
);

-- MOP Task
CREATE TABLE IF NOT EXISTS mop_task (
    worker_id INTEGER PRIMARY KEY,
    task_id TEXT DEFAULT ""
    task TEXT DEFAULT ""
);

-- MOP Spec
CREATE TABLE IF NOT EXISTS mop_spec (
    id INTEGER PRIMARY KEY,
    spec_str TEXT DEFAULT ""
);

-- MOP Sample Time
CREATE TABLE IF NOT EXISTS mop_sample_time (
    model_id INTEGER,
    parallelism TEXT,
    time_taken FLOAT,
    PRIMARY KEY (model_id, parallelism),
    UNIQUE (model_id, parallelism)
);

-- MOP Worker Status
CREATE TABLE IF NOT EXISTS mop_worker_status (
    worker_id INTEGER PRIMARY KEY,
    worker_status TEXT DEFAULT ""
);

-- MOP Model Mapping
CREATE TABLE IF NOT EXISTS mop_model_mapping (
    model_id INTEGER PRIMARY KEY,
    model_config TEXT DEFAULT ""
);

-- MOP Model on Worker
CREATE TABLE IF NOT EXISTS mop_model_on_worker (
    worker_id INTEGER PRIMARY KEY,
    epoch INTEGER,
    model_id INTEGER,
    is_last_worker BOOLEAN
);

-- MOP Parallelism Mapping
CREATE TABLE IF NOT EXISTS mop_parallelism_mapping (
    model_id INTEGER PRIMARY KEY,
    parallelism TEXT
);

-- MOP Parallelism Mapping
CREATE TABLE IF NOT EXISTS mop_model_parallelism_on_worker (
    worker_id INTEGER PRIMARY KEY,
    model_id INTEGER,
    parallelism TEXT
);
