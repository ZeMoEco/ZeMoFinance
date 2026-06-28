# ZeMo Biz 业务编排层核心接口规范（Schema 对齐重写版）

更新时间：2026-04-11  
适用版本：当前 strict core schema（以 utils/DB/sql/create_tables.sql 与 utils/DB/DB.uts 为准）

---

## 1. 设计基准（对齐当前真实表）

1. 单一数据源（Single Source of Truth）  
- Timeline 不做跨表冗余写入。首页/日历流由日志、交易、任务状态通过查询聚合。
- 任务完成与账单记录优先写各自主表（nodes / transactions），不再复制一份到 logs。

2. 关系驱动（Edge-Driven）  
- 所有父子、引用、画布连接、排序都走 edges.relation_type + edges.rank。
- 实体数据（内容/状态）与关系拓扑解耦。

3. 本地高性能（Local-First）  
- 高频 UI 操作（拖拽/连续编辑）先内存更新，后防抖批量落盘。
- 所有复合写入场景必须事务化（DB.beginTransaction / setTransactionSuccessful / endTransaction）。

---

## 2. 数据基线（当前真实表能力）

## 2.1 核心表与用途

- nodes：统一实体（任务/笔记/画布节点等）
- edges：统一关系（父子、引用、视觉连线、排序）
- logs：操作/事件流（支持 parent_log_id 嵌套）
- transactions：财务流水
- accounts：账户资产
- users：用户主档
- user_kv：用户偏好 KV
- attachments：文件元数据（外键只允许挂到 nodes.id）
- vw_timeline_feed：时间线聚合视图（logs + transactions + task_done）

## 2.2 关键约束

- attachments.node_id 外键指向 nodes.id。  
  结论：附件不能直接挂 logs.id 或 transactions.id，文件必须直连某个真实业务节点（task/note/canvas node 等）。

- edges(source_id, target_id, relation_type) 唯一。  
  结论：同一对 source-target 同一种 relation_type 只能存在 1 条边；多重语义需拆 relation_type 或写 properties。

- nodes/status 与 nodes/properties 并存。  
  结论：重构时以 nodes.status 为主字段，properties 用于扩展信息，避免状态分裂。

## 2.3 统一 relation_type 建议（保留现有关系名）

- CONTAINS_TASK_TREE：任务树包含
- CONTAINS_NOTE_BLOCK：笔记块顺序包含
- CONTAINS_CANVAS：画布容器包含
- CONTAINS：通用包含（仅在无法归类时使用）
- REFERENCES：语义引用或弱关联（知识引用、视觉关联）
- DEPENDS_ON：有先后约束的依赖关系

说明：目标方案不再使用状态面具和挂载面具（HAS_*）。状态统一落 nodes.status，文件统一走 attachments 直连节点。

不建议用 properties.scene 做语义分流。建议把语义直接编码进 relation_type（CONTAINS_*），
这样查询可直接命中 edges(source_id, relation_type, is_deleted, rank) 复合索引，减少 JSON 解析过滤开销。

## 2.4 vw_timeline_feed 说明（它是视图，不是物理表）

- vw_timeline_feed 是 SQLite View，只保存查询定义，不保存数据副本。
- 它不会改变现有表结构，不会新增业务字段，也不会额外放大主数据存储。
- 查询视图时会实时执行底层 UNION ALL，性能取决于底层表索引与过滤条件。
- 当前 schema 已有关键索引（logs.created_at、transactions.transaction_time、nodes.completed_at/updated_at 等相关字段可用），通常能支撑分页时间线。
- 优化建议：始终使用 LIMIT/OFFSET + 时间窗口过滤；若后续数据量非常大，再考虑增量快照表（物化缓存）而非一开始就冗余写入。

---

## 3. 接口规范（功能描述 + 表实现）

## 3.1 UserBiz（用户与配置）

### API: initOrUpdateAccount(data)

功能描述：初始化或更新用户主档（username/avatar/xp/level/coins）。

建议签名：

- 输入：{ username?: string, avatar?: string, xp?: number, level?: number, coins?: number }
- 输出：Promise<number>（返回 user_id）

表实现：

- 读 users 首行（单用户模式）
- 若不存在则 INSERT
- 存在则 UPDATE 指定字段

SQL 参考：

```sql
SELECT id FROM users ORDER BY id ASC LIMIT 1;

INSERT INTO users (username, avatar, xp, level, coins, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?);

UPDATE users
SET username = COALESCE(?, username),
    avatar   = COALESCE(?, avatar),
    xp       = COALESCE(?, xp),
    level    = COALESCE(?, level),
    coins    = COALESCE(?, coins),
    updated_at = ?
WHERE id = ?;
```

### API: setPreference(k, v) / getPreference(k)

功能描述：存储主题、默认视图、排版偏好等个性化配置。

表实现：

- user_kv（主键 user_id + k）
- set 使用 INSERT OR REPLACE
- get 使用单行查询

SQL 参考：

```sql
INSERT OR REPLACE INTO user_kv (user_id, k, v, updated_at)
VALUES (?, ?, ?, ?);

SELECT v FROM user_kv WHERE user_id = ? AND k = ? LIMIT 1;
```

### API: updateWallet(accountId, delta)

功能描述：账户余额原子增减。

表实现：

- accounts.balance 原子更新
- 需要事务包裹（与交易插入同事务更安全）

SQL 参考：

```sql
UPDATE accounts
SET balance = balance + ?,
    updated_at = ?
WHERE id = ? AND is_deleted = 0;
```

---

## 3.2 TimelineBiz（时间线中枢）

### API: recordLog(entityId, action, text, files?)

功能描述：记录行为日志；可附带图片/文件。

建议签名：

- 输入：entityId: string, action: string, text: string, files?: string[]
- 输出：Promise<string>（logId）

表实现：

1) 写 logs  
2) 若有附件：
- attachments 直接写到 entityId 对应的业务节点（attachments.node_id = entityId）
- file_type 可标记为 image/document/video/log_proof 等

说明：attachments 不能直接 node_id = log.id（因外键只认 nodes.id），因此这里采用“文件直连业务节点”。

SQL 参考：

```sql
INSERT INTO logs (id, entity_id, action, text, created_at, updated_at, properties, is_deleted)
VALUES (?, ?, ?, ?, ?, ?, ?, 0);
```

### API: appendLog(parentLogId, text, files?)

功能描述：日志回复/补充说明。

表实现：

- logs.entity_id = parentLogId
- logs.parent_log_id = parentLogId
- 附件直连被回复日志的宿主节点：先查 parentLogId 对应日志的 entity_id，再写 attachments.node_id = entity_id

SQL 参考：

```sql
INSERT INTO logs (id, entity_id, action, text, parent_log_id, created_at, updated_at, properties, is_deleted)
VALUES (?, ?, 'COMMENT', ?, ?, ?, ?, ?, 0);
```

### API: getTimeline(offset, limit)

功能描述：统一主页时光流。

表实现：

- 直接读 vw_timeline_feed（logs + transactions + task_done）
- 按 created_at DESC 分页

SQL 参考：

```sql
SELECT source_type, source_id, entity_id, content, created_at, amount, txn_type
FROM vw_timeline_feed
WHERE created_at IS NOT NULL
ORDER BY created_at DESC
LIMIT ? OFFSET ?;
```

重构约束：

- 记账与任务完成不额外写 logs，避免重复事件。
- 仅“解释性变更”写 logs（如备注、评论、人工记录）。

---

## 3.3 FinanceBiz（记账与流水）

### API: recordTransaction(accId, amount, type, sourceId, receipt?)

功能描述：记账、更新账户余额、挂来源对象、可带小票。

建议签名：

- 输入：accId: string, amount: number, type: 'income'|'expense'|'transfer', sourceId?: string, receipt?: string[]
- 输出：Promise<string>（transactionId）

表实现：

1) 事务开始  
2) UPDATE accounts.balance = balance +/- amount  
3) INSERT transactions（properties 记录 source_node_id）  
4) receipt 若存在：附件直连 sourceId 对应业务节点（attachments.node_id = sourceId，file_type='receipt'）  
5) 提交事务

SQL 参考：

```sql
INSERT INTO transactions (
  id, account_id, amount, type, category, description, location,
  transaction_time, created_at, updated_at, balance_snapshot, properties, is_deleted
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0);
```

properties 建议：

```json
{
  "source_node_id": "node_xxx"
}
```

约束：若上传 receipt，则 sourceId 必须是有效 node_id（因为 attachments 只能直连 nodes.id）。

### API: getNodeCost(nodeId)

功能描述：统计某任务/食谱/项目累计花费。

表实现：

- transactions.properties 存 source_node_id
- 聚合 SUM(amount)

SQL 参考：

```sql
SELECT COALESCE(SUM(amount), 0) AS total
FROM transactions
WHERE is_deleted = 0
  AND type = 'expense'
  AND json_extract(properties, '$.source_node_id') = ?;
```

---

## 3.4 NoteBiz（模块化笔记）

### API: insertBlock(noteId, text, rank, files?)

功能描述：块级笔记插入，按 rank 控制顺序。

表实现：

1) INSERT nodes（type='NOTE_BLOCK'）  
2) INSERT edges（source_id=noteId, target_id=blockId, relation_type='CONTAINS_NOTE_BLOCK', rank）  
3) files 写 attachments（node_id=blockId）

SQL 参考：

```sql
INSERT INTO nodes (id, type, content, properties, created_at, updated_at, is_deleted)
VALUES (?, 'NOTE_BLOCK', ?, ?, ?, ?, 0);

INSERT INTO edges (id, source_id, target_id, relation_type, rank, properties, created_at, updated_at, is_deleted)
VALUES (?, ?, ?, 'CONTAINS_NOTE_BLOCK', ?, NULL, ?, ?, 0);
```

### API: addComment(targetId, targetType, text, files?)

功能描述：统一评论入口（笔记评论、任务补充、日志回复）。

表实现：

- 评论主体写 logs（action='COMMENT'）
- targetType in ('note','task')：entity_id=targetId
- targetType='log'：entity_id=targetId 且 parent_log_id=targetId
- 附件直连目标业务节点：
- note/task 评论：attachments.node_id = targetId
- log 回复：attachments.node_id = parent log 的 entity_id

properties 建议：

```json
{
  "target_type": "task|note|log"
}
```

---

## 3.5 TaskBiz（任务引擎）

### API: createTask(content, parentId?)

功能描述：创建任务，支持父子层级。

表实现：

1) INSERT nodes（type='TASK', status='todo'）  
2) 若 parentId 存在，INSERT edges（relation_type='CONTAINS_TASK_TREE'，rank 默认 '500'）

SQL 参考：

```sql
INSERT INTO nodes (id, type, status, content, properties, created_at, updated_at, is_deleted)
VALUES (?, 'TASK', 'todo', ?, ?, ?, ?, 0);

INSERT INTO edges (id, source_id, target_id, relation_type, rank, properties, created_at, updated_at, is_deleted)
VALUES (?, ?, ?, 'CONTAINS_TASK_TREE', '500', NULL, ?, ?, 0);
```

兼容说明：建议在迁移期保留旧 CONTAINS 读取兼容，新增写入统一到 CONTAINS_TASK_TREE。

### API: toggleStatus(nodeId, newStatus, proofFiles?)

功能描述：任务状态切换，支持完成证明。

表实现：

- UPDATE nodes.status
- 完成态写 completed_at
- 可选同步 JSON_SET(properties, '$.status', newStatus) 兼容旧查询
- proofFiles 通过 TimelineBiz.recordLog 写解释性轨迹并挂附件

SQL 参考：

```sql
UPDATE nodes
SET status = ?,
    completed_at = CASE WHEN ? IN ('done','completed') THEN ? ELSE completed_at END,
    properties = json_set(COALESCE(properties, '{}'), '$.status', ?),
    updated_at = ?
WHERE id = ? AND is_deleted = 0;
```

去重策略：

- newStatus 为 done/completed 时，不再额外写“完成日志事件”；时间线直接从 vw_timeline_feed 的 task_done 分支获取。

---

## 3.6 CanvasBiz（空间化画布）

### API: getCanvasCards(canvasId)

功能描述：获取画布卡片及其最近动态。

表实现：

- edges：找出 canvas 的子节点（CONTAINS_CANVAS）
- logs：按 entity_id 聚合最近一条
- attachments：拉卡片节点封面（直连节点附件）

SQL 参考：

```sql
WITH cards AS (
  SELECT n.*, e.rank
  FROM edges e
  JOIN nodes n ON n.id = e.target_id
  WHERE e.source_id = ?
    AND e.relation_type = 'CONTAINS_CANVAS'
    AND e.is_deleted = 0
    AND n.is_deleted = 0
), latest_log AS (
  SELECT l1.*
  FROM logs l1
  JOIN (
    SELECT entity_id, MAX(created_at) AS mx
    FROM logs
    WHERE is_deleted = 0
    GROUP BY entity_id
  ) t ON t.entity_id = l1.entity_id AND t.mx = l1.created_at
)
SELECT c.*, ll.id AS latest_log_id, ll.text AS latest_log_text, ll.created_at AS latest_log_at
FROM cards c
LEFT JOIN latest_log ll ON ll.entity_id = c.id
ORDER BY c.rank ASC;
```

### API: addCanvasNode(canvasId, content, x, y)

功能描述：在画布中新增节点并设置坐标。

表实现：

- INSERT nodes（properties 保存 x,y）
- INSERT edges（canvas -> node, relation_type='CONTAINS_CANVAS'）

### API: linkNodes(sourceId, targetId, relationType, styleJSON)

功能描述：建立节点关系或视觉连线。

表实现：

- INSERT OR REPLACE edges（受唯一索引约束）
- 样式写 edges.properties

### API: moveNode(nodeId, dx, dy)

功能描述：移动节点坐标，持久化位置。

表实现：

- 从 nodes.properties 读取当前 x/y
- 写回 JSON_SET(properties, '$.x', newX, '$.y', newY)

SQL 参考：

```sql
UPDATE nodes
SET properties = json_set(COALESCE(properties, '{}'), '$.x', ?, '$.y', ?),
    updated_at = ?
WHERE id = ?;
```

性能约束：

- touchmove 不落盘
- touchend 或 500ms 防抖后写入
- 可批量 UPDATE 降低锁竞争

---

## 3.7 CalendarBiz（日历视口）

### API: getCalendar(startDate, endDate)

功能描述：返回时间窗口内的多源事件集合，并按天分组供 UI 渲染。

表实现：

- 主源：vw_timeline_feed（日志、交易、完成任务）
- 可选补充：nodes.due_at 范围内的未完成任务
- 前端按 YYYY-MM-DD 分组

SQL 参考：

```sql
SELECT source_type, source_id, entity_id, content, created_at, amount, txn_type
FROM vw_timeline_feed
WHERE created_at BETWEEN ? AND ?

UNION ALL

SELECT 'task_due' AS source_type,
       id AS source_id,
       id AS entity_id,
       COALESCE(content, '') AS content,
       due_at AS created_at,
       NULL AS amount,
       NULL AS txn_type
FROM nodes
WHERE is_deleted = 0
  AND type = 'TASK'
  AND due_at BETWEEN ? AND ?
  AND COALESCE(status, 'todo') NOT IN ('done', 'completed', 'archived')
ORDER BY created_at DESC;
```

---

## 4. 事务与一致性规范

1. 需要事务的接口
- recordTransaction（余额 + 流水 + 附件）
- createTask（节点 + 父子边）
- insertBlock（块节点 + 顺序边 + 附件）
- recordLog/appendLog（日志 + 附件直连节点）

2. 软删优先
- 业务删除优先更新 is_deleted=1
- 物理删除仅在归档/清理流程执行

3. 时间戳统一
- 全部使用毫秒时间戳（Date.now()）

4. 幂等与唯一
- 依赖 edges 唯一索引时，关系写入使用 INSERT OR REPLACE / 先查后改

---

## 5. 当前代码与目标规范偏差（重构任务清单）

1. Timeline 仍有 nodes.type='timeline' 写入路径（TimelineBiz.list/add）
- 目标：统一迁到 logs + vw_timeline_feed 聚合

2. Task 状态当前主要放在 edges 的 HAS_STATUS 面具
- 目标：去面具化，统一改为 nodes.status（必要时同步 properties.status 做兼容）

3. NoteBiz.getNotesByAttribute 仍查询 tasks 表（现 schema 无该表）
- 目标：改为 nodes + edges + properties 方案

4. CalendarBiz 仍在查询 nodes.type='log' / 'timeline'
- 目标：改用 logs 与 vw_timeline_feed

5. FinanceBiz 类型声明中 account_id/id 仍按 number 使用
- 目标：统一改为 string（与 schema TEXT 对齐）

6. AttachmentService 插入 attachments 时未显式写 file_type
- 目标：强制写 file_type（image/document/video/...）

7. 日志/交易附件仍存在“载体节点”方案残留
- 目标：全部改为 attachments 直连业务节点，移除 LOG_ASSET / TXN_ASSET 设计

8. relation_type 语义子类型尚未统一（CONTAINS 仍混用）
- 目标：统一为 CONTAINS_TASK_TREE / CONTAINS_NOTE_BLOCK / CONTAINS_CANVAS，避免 scene JSON 过滤

---

## 6. 重构顺序建议（可直接拆迭代）

阶段 A（基础一致性）
- 统一 ID 类型为 string
- 修复 attachments.file_type 必填
- 去掉 HAS_STATUS / HAS_ATTACHMENT / HAS_LOG 等面具依赖
- 将日志/交易附件改为直连业务节点
- 移除对不存在表 tasks 的依赖

阶段 B（时间线收敛）
- TimelineBiz 改读 vw_timeline_feed
- Task 完成/Finance 记账取消冗余 logs 写入

阶段 C（关系字典统一）
- 统一 CONTAINS_* 子类型（TASK_TREE / NOTE_BLOCK / CANVAS）
- Task 父子、Canvas 容器、Note 块顺序按 relation_type 直接过滤

阶段 D（性能与体验）
- Canvas moveNode 防抖批量落盘
- Calendar 聚合 SQL 与前端分组缓存优化

阶段 E（主页导入与同步块）
- 主页时间线导入任务/笔记从“文本复制”升级为“同步块引用”
- 建立 source log -> sync block -> task/note 消费关系，实现单源多引用与同步更新

---

## 7. 新需求实施方案：主页时间线日志导入任务/笔记（同步块）

## 7.1 需求目标

1. 主页时间线长按菜单“导入为任务新记录 / 导入到最近笔记”改为导入同步块，而不是复制文本。
2. 同一条时间线日志只维护一个源数据块，被多个任务/笔记引用时保持内容同步。
3. 不新增物理表，复用 nodes / edges / logs / attachments。

## 7.2 当前问题（现状）

1. 当前导入任务走 TaskBiz.addTaskLog，导入笔记走 NoteBiz.updateNoteContent，本质是字符串拷贝。
2. 源日志后续修改后，导入目标不会同步更新，数据会分叉。
3. 附件缺少“按日志精确归属”的引用元数据，后续同步只能粗粒度回退到 entity 级附件。

## 7.3 数据模型设计（小幅扩展表结构）

0. 约束与取舍（关键）
- 不移除 edges.source_id/target_id -> nodes.id 外键约束（保证关系完整性与级联清理安全）。
- 为避免把同步块主键写入 properties，新增 logs 热字段列：
- logs.sync_block_id TEXT（引用的同步块节点 ID）
- logs.source_log_id TEXT（来源日志 ID）
- 建议索引：

```sql
CREATE INDEX IF NOT EXISTS idx_logs_sync_block_id ON logs(sync_block_id);
CREATE INDEX IF NOT EXISTS idx_logs_source_log_id ON logs(source_log_id);
```

说明：标题保持“复用核心表”，这里是对 logs 的增量列扩展，不新增业务实体表。

1. 同步块节点（nodes）
- type: 'SYNC_BLOCK'（建议新增到 NodeType；迁移期也可临时用 type='NOTE' + properties.block_kind='sync_block'）
- content: 同步块正文（默认镜像源日志 text）
- properties 建议字段：

```json
{
  "source_log_id": "log_xxx",
  "source_entity_id": "node_xxx",
  "source_type": "log",
  "sync_mode": "mirror",
  "version": 1,
  "last_synced_at": 1710000000000,
  "attachment_refs": ["att_x", "att_y"]
}
```

2. 任务侧引用（logs）
- 向目标任务新增一条日志：
- action='SYNC_BLOCK_REF'
- entity_id=taskId
- text 可保留源日志摘要（作为降级展示）
- 直接写 logs.sync_block_id / logs.source_log_id（热字段列，避免 JSON 解析）
- 推荐同时写一条 taskId -> syncBlockId 的 REFERENCES 边（edges）用于去重与反查

3. 笔记侧引用（edges）
- 在 noteId -> syncBlockId 写一条 CONTAINS_NOTE_BLOCK 边（rank 控制顺序）
- edges.properties 建议增加 block_kind='sync_ref'
- 结论：笔记侧必须依赖 edges；任务侧建议“logs + edges 双写”，其中 logs 管时序展示，edges 管关系检索

4. 源日志反向绑定（logs.sync_block_id）
- 源日志行写 logs.sync_block_id，保证同一 source_log_id 导入多次时复用同一个同步块（幂等）
- 说明：不能仅用 edges 替代该绑定，因为 edges.source_id/target_id 有 nodes 外键约束，不能直接指向 logs.id
- 可选增强：同步块节点 properties 冗余 source_log_id 作为冷备元数据

## 7.4 Biz API 设计

### TimelineBiz

1. ensureSyncBlockFromLog(logId: string): Promise<string | null>
- 若 logs.sync_block_id 已存在且节点有效，直接返回
- 否则创建 SYNC_BLOCK 节点并回写源日志 logs.sync_block_id

2. importTimelineLogToTask(logId: string, taskId: string, parentLogId?: string | null): Promise<string | null>
- 调 ensureSyncBlockFromLog
- 写 task 日志（action='SYNC_BLOCK_REF'，并写 sync_block_id/source_log_id 列）
- 可选：写 task -> syncBlock 的 REFERENCES 边用于反查

3. importTimelineLogToNote(logId: string, noteId: string, rank?: string): Promise<boolean>
- 调 ensureSyncBlockFromLog
- 写 note -> syncBlock 的 CONTAINS_NOTE_BLOCK

4. syncBlockFromSourceLog(logId: string): Promise<boolean>
- 在 updateTimelineEntry / deleteTimelineEntry 后调用
- 将源日志 text / 删除态同步到对应 SYNC_BLOCK

### TaskBiz / NoteBiz

1. TaskBiz.getTaskLogs 需把 logs.sync_block_id / logs.source_log_id 暴露给 UI
2. NoteBiz 新增 listNoteBlocks(noteId) 与 insertSyncBlock(noteId, syncBlockId, rank)

## 7.5 UI 与渲染改造

1. 主页时间线（theme-feed）
- 长按导入操作改为传递 logId（不是 content 文本）
- 分别调用 TimelineBiz.importTimelineLogToTask / importTimelineLogToNote

2. 任务日志列表（task-log-list）
- 识别 action='SYNC_BLOCK_REF' 或 logs.sync_block_id
- 渲染 SyncBlockCard（只读）并显示“源日志时间/来源”

3. 笔记页（note-edit / note-render）
- 在块列表中支持 sync_ref 类型
- 渲染时读取 SYNC_BLOCK 最新内容（而非静态快照）

## 7.6 一致性与事务规范

1. 导入任务/笔记必须使用事务：
- ensureSyncBlock + 写目标引用（logs 或 edges）必须同事务提交

2. 幂等约束：
- 同一 source_log_id 必须复用同一 sync_block_id
- 对同一 noteId + syncBlockId + CONTAINS_NOTE_BLOCK 可做去重（先查后写或 INSERT OR REPLACE）

3. 删除策略：
- 删除源日志时，不直接物理删已导入目标
- SYNC_BLOCK 标记 source_deleted=1，并在 UI 提示“源已删除”

## 7.7 迭代拆分（建议）

E1（数据层）
- logs 增加 sync_block_id/source_log_id 列与索引
- TimelineBiz 新增 ensure/import/sync API
- TaskBiz.getTaskLogs 补充 sync_block_id/source_log_id 透传

E2（任务链路）
- 主页导入任务改走 sync block
- task-log-list 增加同步块卡片渲染

E3（笔记链路）
- NoteBiz 块接口补齐
- 笔记页面支持 sync_ref 渲染与排序

E4（附件精确同步）
- 日志写入时补全 properties.attachment_refs
- 同步块优先按 attachment_refs 取附件，缺失时回退 entity 级附件

## 7.8 验收标准

1. 同一条主页时间线可导入多个任务/笔记且只生成 1 个 SYNC_BLOCK。
2. 修改源日志内容后，所有导入位置显示内容同步变化。
3. 删除源日志后，导入位置不崩溃，显示“源已删除”状态。
4. 导入过程失败时不产生半写入脏数据（事务回滚可验证）。

---

本文件作为 Biz 层重构的目标契约，后续代码改造以本规范与 create_tables.sql 为唯一真相来源。
