-- =============================================================
-- ZeMo Unified Schema (Strict Core)
-- 设计目标：
-- 1) 单一数据源：日志/账单/任务状态通过查询聚合视图拼接
-- 2) 关系驱动：edges.relation_type + rank 驱动父子/引用/画布连线
-- 3) 零历史兼容字段：只保留当前业务真实使用字段
-- =============================================================

PRAGMA foreign_keys = ON;

-- =============================================================
-- 1) 核心节点
-- =============================================================
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT,
    content TEXT,
    properties TEXT,
    due_at INTEGER,
    completed_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    is_deleted INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_nodes_type_deleted_created ON nodes(type, is_deleted, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_updated_at ON nodes(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_due_at ON nodes(due_at);

-- =============================================================
-- 2) 核心关系边（Edge-Driven）
-- =============================================================
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    context_key TEXT,
    status TEXT NOT NULL DEFAULT 'todo',
    secondary_category TEXT,
    rank TEXT NOT NULL DEFAULT '500',
    properties TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL DEFAULT 0,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY(target_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique_relation ON edges(source_id, target_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_edges_source_relation_deleted ON edges(source_id, relation_type, is_deleted);
CREATE INDEX IF NOT EXISTS idx_edges_relation_status_category_deleted ON edges(relation_type, status, secondary_category, is_deleted);
CREATE INDEX IF NOT EXISTS idx_edges_source_relation_rank ON edges(source_id, relation_type, is_deleted, rank);
CREATE INDEX IF NOT EXISTS idx_edges_target_relation ON edges(target_id, relation_type, is_deleted);

-- =============================================================
-- 3) 用户主档（主表）
-- =============================================================
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    avatar TEXT,
    xp INTEGER NOT NULL DEFAULT 0,
    level INTEGER NOT NULL DEFAULT 1,
    coins INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- =============================================================
-- 4) 资产账户（从表）
-- =============================================================
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL DEFAULT 1,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    balance REAL NOT NULL DEFAULT 0.0,
    currency TEXT NOT NULL DEFAULT 'CNY',
    category TEXT NOT NULL DEFAULT 'cash',
    card_no TEXT,
    properties TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_accounts_user ON accounts(user_id, is_deleted);
CREATE INDEX IF NOT EXISTS idx_accounts_name ON accounts(name);

-- =============================================================
-- 5) 资金流水（单一数据源）
-- =============================================================
CREATE TABLE IF NOT EXISTS transactions (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    amount REAL NOT NULL,
    type TEXT NOT NULL,
    category TEXT,
    description TEXT,
    location TEXT,
    transaction_time INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL DEFAULT 0,
    balance_snapshot REAL DEFAULT 0.0,
    source_node_id TEXT,
    item_node_id TEXT,
    location_node_id TEXT,
    properties TEXT,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(account_id) REFERENCES accounts(id)
);

CREATE INDEX IF NOT EXISTS idx_transactions_account_time ON transactions(account_id, transaction_time DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions(transaction_time DESC);

-- =============================================================
-- 6) 日志中心
-- =============================================================
CREATE TABLE IF NOT EXISTS logs (
    id TEXT PRIMARY KEY,
    owner_type TEXT NOT NULL DEFAULT 'node',
    owner_id TEXT NOT NULL,
    action TEXT NOT NULL,
    text TEXT,
    parent_log_id TEXT,
    sync_block_id TEXT,
    source_log_id TEXT,
    timestamp INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL DEFAULT 0,
    properties TEXT,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(parent_log_id) REFERENCES logs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_logs_owner_created ON logs(owner_type, owner_id, created_at DESC, is_deleted);
CREATE INDEX IF NOT EXISTS idx_logs_time ON logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_sync_block_id ON logs(sync_block_id);
CREATE INDEX IF NOT EXISTS idx_logs_source_log_id ON logs(source_log_id);

-- =============================================================
-- 7) 附件
-- node_id 绑定主节点（Node）
-- =============================================================
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    file_name TEXT,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_hash TEXT,
    file_size INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL DEFAULT 0,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attachments_node_created ON attachments(node_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_attachments_hash ON attachments(file_hash);

-- =============================================================
-- 8) 用户偏好
-- =============================================================
CREATE TABLE IF NOT EXISTS user_kv (
    user_id INTEGER NOT NULL DEFAULT 0,
    k TEXT NOT NULL,
    v TEXT,
    updated_at INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, k)
);

CREATE INDEX IF NOT EXISTS idx_user_kv_updated ON user_kv(updated_at DESC);

-- =============================================================
-- 9) 时间线聚合视图（单一数据源，不做冗余写入）
-- =============================================================
CREATE VIEW IF NOT EXISTS vw_timeline_feed AS
SELECT
    'log' AS source_type,
    l.id AS source_id,
    l.owner_id AS entity_id,
    COALESCE(l.text, '') AS content,
    COALESCE(NULLIF(l.timestamp, 0), l.created_at) AS created_at,
    NULL AS amount,
    NULL AS txn_type
FROM logs l
WHERE l.is_deleted = 0
UNION ALL
SELECT
    'transaction' AS source_type,
    t.id AS source_id,
    t.account_id AS entity_id,
    COALESCE(t.description, t.category, '') AS content,
    COALESCE(t.transaction_time, t.created_at) AS created_at,
    t.amount AS amount,
    t.type AS txn_type
FROM transactions t
WHERE t.is_deleted = 0
UNION ALL
SELECT
    'task_done' AS source_type,
    n.id AS source_id,
    n.id AS entity_id,
    COALESCE(n.content, '') AS content,
    COALESCE(n.completed_at, n.updated_at, n.created_at) AS created_at,
    NULL AS amount,
    NULL AS txn_type
FROM nodes n
WHERE n.is_deleted = 0
  AND (
      lower(COALESCE(n.status, '')) IN ('done', 'completed')
      OR n.properties LIKE '%"status":"done"%'
      OR n.properties LIKE '%"status":"completed"%'
  );

