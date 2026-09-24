# M3-Agent 教育错题记忆系统

这是基于 ByteDance Seed [M3-Agent](https://github.com/ByteDance-Seed/m3-agent) 思想实现的单学习者原型。上游源码固定在 `vendor/m3-agent`，本项目不修改上游的视频、人脸和音频管线，而是实现独立的手写数学作答导入、情景记忆、语义聚合、精确检索和证据导出。

当前版本完成：

- 从本地 FERMAT parquet 导入全部记录；
- 按 `grade` 生成可复现的 `Time`；
- 将原始记录与 benchmark truth 分表保存；
- 可选地用 FERMAT 参考标注生成“仅供评测”的情景记忆；
- 按领域、错误状态和模拟时间精确查询；
- 聚合知识点和错误类型，生成 HTML 错题回访画廊；
- 从 parquet 提取原图，生成参考订正 PNG；
- 写入教育记忆图节点和边，为后续 M3 Control 工具化检索做准备。
- 接入OpenAI兼容数学VLM，保存逐步转写、知识点、错误类型、首错步骤和bbox；
- 将每次VLM诊断按模型和prompt版本独立保存，并与benchmark truth隔离评测；
- 保存结构化订正，并验证数值等式、代数恒等式、导数和原函数；
- 自动生成带首错红框和图内正确结果的可审计 PNG；
- 从最新 VLM 诊断聚合可解释的知识点薄弱状态及证据 IDs。
- 将中文自然语言请求转成结构化检索计划，并输出“错因总结、原图回访、可审计订正”三部分 HTML 报告。

## 安装

```powershell
cd "C:\Users\wangf\Desktop\AI agent\edu agent\m3-education-memory"
python -m pip install -e .
```

如果使用当前工作区已经安装的依赖：

```powershell
$env:PYTHONPATH="vendor\python;..\dataset\.python-libs;src"
```

## 导入

以下命令将 FERMAT 导入 SQLite。`--bootstrap-reference-memory` 会使用数据集的 `pert_reasoning` 和 `has_error` 生成评测用记忆；这些内容明确标记为 `benchmark_reference`，不能作为视觉模型能力成绩。

```powershell
python -m m3_edu_memory.cli import-fermat `
  --dataset "..\dataset\FERMAT" `
  --db "data\education_memory.db" `
  --simulation-anchor "2026-09-23" `
  --bootstrap-reference-memory
```

已有数据库可在不重新读取 parquet 的情况下校准时间：

```powershell
python -m m3_edu_memory.cli rebase-times `
  --db "data\education_memory.db" `
  --anchor "2026-09-23"
```

## 查询微积分错题

```powershell
python -m m3_edu_memory.cli query `
  --db "data\education_memory.db" `
  --domain clc `
  --errors-only `
  --limit 10
```

查看数据库、七个月份和记忆图统计：

```powershell
python -m m3_edu_memory.cli stats --db "data\education_memory.db"
```

提炼当前单学习者的微积分薄弱点：

```powershell
python -m m3_edu_memory.cli weaknesses `
  --db "data\education_memory.db" `
  --domain clc
```

## 导出错题画廊和订正图片

```powershell
python -m m3_edu_memory.cli export `
  --db "data\education_memory.db" `
  --domain clc `
  --errors-only `
  --limit 10 `
  --output "outputs\calculus-errors-10"
```

输出包括 `report.json`、`gallery.html`、`images/` 和 `corrections/`。

## 数据边界

当前全部记录视为同一位学习者，不保存 `student_id`。`Time` 是从 grade 生成的模拟时间：七个 grade 顺序映射到以配置锚点结束的最近七个自然月。锚点为 2026-09-23 时，`c07` 到 `c00` 依次对应 2026 年 3–9 月，当前月不会生成晚于锚点日期的时间。它只用于演示时间排序和知识点状态更新，不能解释为真实学习日期。

FERMAT 的 `orig_a`、`pert_a`、`pert_reasoning` 和 `has_error` 保存在独立的 `benchmark_truth` 表。后续接入 VLM 时，模型输入层不得读取该表；该表只用于离线评估。

数学VLM配置、批处理和评测命令见 [docs/VLM_INTEGRATION.md](docs/VLM_INTEGRATION.md)。

## 自然语言记忆查询

```powershell
python -m m3_edu_memory.cli answer `
  --db "data\education_memory.db" `
  --model "qwen3.7-plus" `
  --request-file "config\demo_memory_request.txt" `
  --output "outputs\calculus-vlm-memory"
```

输出目录包含 `report.json`、`gallery.html`、`images/` 与 `audits/`。这条执行路径只使用 VLM 诊断及其派生记忆，不连接 `benchmark_truth`。
在旧版 Windows PowerShell 中建议使用 UTF-8 的 `--request-file`，避免命令行中文被系统代码页转换。

## M3 Control 工具与 HTTP API

控制层提供 `expand_concepts`、`filter_attempts`、`search_memories`、`get_evidence` 和 `generate_correction`。当前使用可复现的确定性规划器，以后可以只替换规划策略为 M3-Agent-Control 或其他 LLM。

```powershell
python -m m3_edu_memory.server `
  --db "data\education_memory.db" `
  --model "qwen3.7-plus" `
  --host 127.0.0.1 `
  --port 8765
```

接口包括：

- `GET /health`
- `POST /v1/memory/query`，JSON 请求示例：`{"request":"帮我整理代数错题","limit":10}`
- `GET /v1/memory/weaknesses?domain=alg`
- `GET /v1/attempts/{attempt_id}`
- `POST /v1/errors/{attempt_id}/review`，支持 `confirmed`、`rejected`、`modified`

服务只依赖 Python 标准库；查询响应中的 `tool_trace` 记录实际使用的控制工具，所有结论保留 evidence attempt IDs。

### 学生错题记忆工作台

启动服务后访问 `http://127.0.0.1:8765/student`（根路径 `/` 也会打开同一页面）。页面支持自然语言查询、常用问题快捷入口、薄弱知识点、错因分布、手写原图、审计订正图、订正验证状态和教师修订入口。

```powershell
$env:PYTHONPATH="vendor\python;src"
python -m m3_edu_memory.server `
  --db "data\education_memory.db" `
  --model "qwen3.7-plus" `
  --host 127.0.0.1 `
  --port 8765
```

“过去三月”“最近3个月”等请求会相对于当前模型记忆中的最新记录计算时间窗口。时间来源细节保留在接口元数据中，不在学生界面展示。

审核记录作为新版本写入 `diagnosis_reviews`，不会覆盖模型原始输出。检索使用 `has_error_effective` 和 `error_type_effective`，而离线模型评测仍使用原始预测，避免人工结果污染模型指标。

## 混合检索索引

为最新模型记忆建立 SQLite FTS5 与本地确定性向量基线：

```powershell
python -m m3_edu_memory.cli index-memory `
  --db "data\education_memory.db" `
  --model "qwen3.7-plus"

python -m m3_edu_memory.cli search-memory `
  --db "data\education_memory.db" `
  --model "qwen3.7-plus" `
  --query "多项式展开错误"
```

本地 `local-hash-embedding-v1` 只用于离线工程基线，不代表语义 embedding 质量。`EmbeddingClient` 协议允许后续切换云端 embedding、开源向量模型或 pgvector，而不改变 Control 工具接口。索引存在时 `search_memories` 自动使用混合排序，否则回退到词法检索。
新的 VLM 诊断或人工审核写入后会自动增量刷新对应 attempt 的文档、全文索引和向量；索引失败不会回滚已经持久化的诊断，并会在结果中返回 `index_error` 供修复任务处理。

火山方舟多模态向量模型已通过 `config/embedding_providers.json` 接入。配置只保存环境变量名，不保存 Key。建立语义索引及启动学生 UI 服务时使用同一档案：

```powershell
python -m m3_edu_memory.cli index-memory `
  --db "data\seed_memory.db" `
  --model "seed-education-router-v1" `
  --embedding-config "config\embedding_providers.json" `
  --embedding-profile "volcengine-doubao-embedding-vision"

python -m m3_edu_memory.server `
  --db "data\seed_memory.db" `
  --model "seed-education-router-v1" `
  --embedding-config "config\embedding_providers.json" `
  --embedding-profile "volcengine-doubao-embedding-vision" `
  --vector-weight 1.0 `
  --host 127.0.0.1 --port 8785
```

该档案使用 `doubao-embedding-vision-251215`、1024 维稠密向量，并区分记忆文档与用户查询的 instructions。固定评测中纯向量排序取得更高 MRR 与 NDCG，因此学生 UI 当前推荐 `--vector-weight 1.0`；结构化领域、时间和错误状态仍在向量排序前精确过滤。账号需在方舟控制台单独开通该向量模型；Seed 2.1 VLM 权限不自动包含它。

固定检索评测使用 14 条中英文领域查询，并以 `memory_documents.domain_code` 作为弱 ground truth：

```powershell
python -m m3_edu_memory.cli evaluate-retrieval `
  --db "data\seed_memory.db" `
  --model "seed-education-router-v1" `
  --queries "config\retrieval_eval_queries.json" `
  --k 10 `
  --output "outputs\retrieval-evaluation\seed-semantic.json"
```

评测输出 Precision@K、Recall@K、Hit@K、MRR 和 NDCG@K。学生查询先取得全部符合领域、时间和错误状态的候选，再在候选集内排序并返回请求数量，避免 `top-k` 提前截断造成漏题。

新增接口：

- `GET /v1/memory/index`
- `GET /v1/analysis/metrics?model=...`：VLM 成功率、图片压缩率、平均/P95 延迟、请求次数和 token usage
- `POST /v1/memory/search`

## 上传与异步诊断

`POST /v1/attempts` 接收 JSON：

```json
{
  "image_base64": "data:image/png;base64,...",
  "question_text": "题目文本",
  "domain_code": "alg",
  "subdomain_code": "epr",
  "grade": "c12",
  "attempted_at": "2026-09-22T10:00:00+08:00",
  "idempotency_key": "client-generated-stable-key",
  "model": "qwen3.7-plus"
}
```

接口立即返回 `202 Accepted`、`attempt_id` 与 `job_id`。图片以 SHA-256 内容地址写入 `data/objects/sha256/`，相同内容不会重复占用空间；20 MiB 为单图上限。重复的 `idempotency_key` 返回原任务。

查询处理状态：

- `GET /v1/jobs/{job_id}`
- `GET /v1/attempts/{attempt_id}`：未完成时返回 202，完成后返回诊断与证据
- `GET /v1/memory/mastery-history?model=...`：查看薄弱知识点状态的不可变历史快照

运行后台 worker：

```powershell
python -m m3_edu_memory.cli process-jobs `
  --db "data\education_memory.db" `
  --base-url "https://your-vlm-endpoint/v1" `
  --model "your-model" `
  --api-key-env VLM_API_KEY `
  --limit 10
```

worker 从数据库原子领取任务，并将状态更新为 `processing`、`completed` 或 `failed`。上传接口和 worker 解耦，因此更换 VLM 只影响 worker 配置。

任务可靠性策略：

- worker 领取任务时写入 `lease_owner` 和 `lease_expires_at`；默认租约 600 秒；
- 崩溃 worker 的过期租约会在下一次领取任务前自动回收；
- 调用失败按指数退避重新排队，默认最多尝试 3 次；
- 重试耗尽后写入 `dead_lettered_at`，保留最后错误；
- `POST /v1/jobs/{job_id}/retry` 可以人工重置失败任务；
- `GET /v1/jobs/metrics` 返回状态计数、重试数量、死信数量和过期租约数量。

worker 可使用 `--worker-id`、`--lease-seconds` 和 `--retry-base-seconds` 调整执行参数。旧版 SQLite 数据库会自动增加任务可靠性字段，无需重新导入 FERMAT。

## Seed 主记忆库与批处理运维

`config/model_routing.json` 固化当前路由策略：先调用 Seed 2.1 Turbo；结果不可用、要求复核、置信度低于 0.9 或不确定时升级到 Seed 2.1 Pro。选中结果会物化为稳定别名 `seed-education-router-v1`，同时保留原模型运行和路由证据。

全量主库构建脚本支持幂等续跑、失败重试和延迟派生索引重建：

```powershell
python scripts\run_seed_main_memory.py `
  --db "data\seed_memory.db" `
  --max-workers 6 `
  --max-passes 3
```

查看当前覆盖率、模型选择、升级数量、待重试任务、最近吞吐和预计剩余时间：

```powershell
python -m m3_edu_memory.cli batch-status `
  --db "data\seed_memory.db" `
  --routing-config "config\model_routing.json" `
  --recent-minutes 30 `
  --output "outputs\seed-main-memory\live-status.json"
```

只有 `materialized_run_id` 已写入的作答才算完成。已有路由决策但尚未物化的作答仍会被后续批次重新领取，避免一次写入中断后永久漏处理。

全量路由运行期间可以启动独立收尾进程。它会等待覆盖率达到 100%，随后离线重验证历史订正、重建规范化掌握状态、检查语义索引完整性、运行 14 条检索评测，并生成微积分、代数、几何、三角学和概率统计抽样报告：

```powershell
python scripts\finalize_seed_main_memory.py
```

进度写入 `outputs\seed-main-memory\finalization\status.json`，最终结果写入同目录下的 `summary.json` 和 `retrieval-evaluation.json`。索引数量与已完成记忆数一致时不会重复计算文档向量。

批处理内置了账户级错误熔断。一旦检测到 HTTP 401/403、`AccountOverdueError`、额度不足或密钥/权限错误，它只等待当前最多 `max-workers` 个请求结束，然后停止领取新题并将状态标记为失败。余额或权限恢复后重新执行 `run_seed_main_memory.py`，会自动跳过已物化记忆并从剩余记录续跑。

## 版本化订正

提交人工或规则订正：

```http
POST /v1/errors/{attempt_id}/correction
Content-Type: application/json

{
  "model": "qwen3.7-plus",
  "source": "human",
  "created_by": "teacher-1",
  "corrected_solution": "完整订正过程",
  "corrected_steps": [
    {"step_index": 0, "latex": "...", "explanation": "..."}
  ],
  "final_answer": "1/2",
  "verification_expression": "1/3 + 1/6 = 1/2",
  "confidence": 1.0,
  "activate": true,
  "render": true
}
```

查看全部订正版本：`GET /v1/errors/{attempt_id}/corrections?model=...`。

订正版本支持 `draft`、`active`、`superseded` 和 `rejected`。激活新版本时，旧 active 版本变为 superseded；原始 VLM 订正不会被覆盖。系统重新执行数学验证、更新 `MasteryState`、刷新混合检索索引，并可生成新审计图。旧数据库中的历史 VLM 订正会自动回填为第一代 active 版本。

## 可插拔 VLM 与固定评测集

模型连接信息可以集中写入配置档案。复制
`config/vlm_providers.example.json` 后修改服务地址和模型名；配置只引用环境变量名，不能保存 API Key 明文。

```powershell
python -m m3_edu_memory.cli diagnose `
  --db "data\education_memory.db" `
  --provider-config "config\vlm_providers.json" `
  --provider-profile "volcengine-seed-2.1-pro" `
  --stratified-pilot --limit 20
```

`process-jobs` 同样支持 `--provider-config` 和 `--provider-profile`，因此替换模型无需修改 worker 代码。

当前提供 `volcengine-seed-2.1-pro` 和 `volcengine-seed-2.1-turbo` 两个火山方舟档案。它们共用本地环境变量 `ARK_API_KEY`；配置文件不保存密钥。可以运行 `powershell -ExecutionPolicy Bypass -File scripts\configure_ark_key.ps1` 隐藏输入并保存 Key，随后重新打开 PowerShell/Codex。

先建立一次固定评测集，再让不同模型处理相同的 `attempt_id`：

```powershell
python -m m3_edu_memory.cli create-eval-suite `
  --db "data\education_memory.db" `
  --name "fermat-vlm-pilot-100" `
  --limit 100 `
  --seed "fermat-vlm-eval-v1"

python -m m3_edu_memory.cli show-eval-suite `
  --db "data\education_memory.db" `
  --suite "fermat-vlm-pilot-100"

python -m m3_edu_memory.cli diagnose `
  --db "data\education_memory.db" `
  --evaluation-suite "fermat-vlm-pilot-100" `
  --provider-config "config\vlm_providers.json" `
  --provider-profile "volcengine-seed-2.1-pro" `
  --only-unprocessed --max-workers 2

python -m m3_edu_memory.cli evaluate `
  --db "data\education_memory.db" `
  --model "doubao-seed-2-1-pro-260915" `
  --provider "volcengine-ark" `
  --suite "fermat-vlm-pilot-100" `
  --persist

python -m m3_edu_memory.cli compare-eval-suite `
  --db "data\education_memory.db" `
  --suite "fermat-vlm-pilot-100"
```

评测直接以 `benchmark_truth` 中的 FERMAT `has_error`、`pert_reasoning` 映射错误类型和 `orig_a` 正确答案为 ground truth，不读取教师审核或运行时覆盖。评测保存错误识别 accuracy/precision/recall/F1、错误类型精确匹配率、参考答案严格匹配率、订正验证率及复核请求率。FERMAT 没有步骤级 bbox 和知识点真值，因此首错步骤、bbox 与知识点只报告覆盖率，不表述为准确率。评测集对每个 `orig_q` 只保留一个扰动版本，避免同一道原题重复计权。

参考答案同时报告 `reference_answer_exact_match` 和
`reference_answer_equivalent_match`。后者保守归一化常见 LaTeX、单位和等式末项，再使用安全符号验证判断数学等价；无法安全解析的答案不会被猜测为正确。VLM 原始知识点标签保持不变以便审计，薄弱状态重建与检索文档会额外执行知识点别名归一化，例如将 `chain rule`、`复合函数求导` 和 `链式法则` 聚合为同一个课程知识点。

## 查询报告中的教师修订

教师修订属于查询与展示层，不参与离线模型评测。启动本地服务后，打开查询报告中的“教师修改 VLM 诊断或订正”，或者访问：

`http://127.0.0.1:8765/review?attempt_id={attempt_id}&model={model}`

教师可以修改是否有错、错误类型、首错步骤、归一化错误框、知识点和错因说明：

```http
POST /v1/errors/{attempt_id}/review
Content-Type: application/json

{
  "model": "qwen3.7-plus",
  "reviewer": "teacher-1",
  "verdict": "modified",
  "has_error": true,
  "error_type": "omitted_step",
  "first_error_step": 2,
  "error_bbox": [0.12, 0.38, 0.86, 0.52],
  "knowledge_points": ["chain rule"],
  "error_explanation": "求复合函数导数时遗漏内层函数导数。",
  "notes": "查询报告人工修订",
  "render": true
}
```

如果 VLM 订正错误，教师通过 `POST /v1/errors/{attempt_id}/correction` 提交新的 `source=human` 订正版本。原始 VLM 诊断和订正不会被覆盖；查询、薄弱点聚合、混合检索和新审计图使用最新教师生效版本。`GET /v1/attempts/{attempt_id}` 同时返回原始预测字段、`*_effective` 字段、教师信息及 `display_source`。

启动本地服务：

```powershell
python -m m3_edu_memory.server `
  --db "data\education_memory.db" `
  --model "qwen3.7-plus" `
  --host 127.0.0.1 `
  --port 8765
```

教师页面读取原图、当前 VLM 结果和最新人工覆盖，但不会读取或修改 `benchmark_truth`。首次读取 parquet 图片后会缓存原始字节，后续打开不再扫描数据分片。

## 真实端到端验证

`scripts/run_e2e_validation.py` 会从真实 FERMAT 分片读取一张图片，启动隔离的本地 HTTP 服务，完成上传、Seed VLM 诊断、记忆写入、Doubao 多模态向量索引、自然语言检索、薄弱点聚合和审计图生成。脚本使用隔离数据库，不会修改主数据库。

```powershell
$env:PYTHONPATH = "vendor\python;src"
python scripts\run_e2e_validation.py `
  --source-db "data\education_memory.db" `
  --source-attempt-id "fermat-00001-00010-000061" `
  --provider-config "config\vlm_providers.json" `
  --provider-profile "volcengine-seed-2.1-turbo" `
  --embedding-config "config\embedding_providers.json" `
  --embedding-profile "volcengine-doubao-embedding-vision" `
  --run-prefix "seed-turbo-e2e"
```

全量路由完成后，可显式重建规范化薄弱状态与最终语义索引。该命令也用于批处理进程启动后代码发生升级的情况：

```powershell
python -m m3_edu_memory.cli rebuild-derived-memory `
  --db "data\seed_memory.db" `
  --model "seed-education-router-v1" `
  --embedding-config "config\embedding_providers.json" `
  --embedding-profile "volcengine-doubao-embedding-vision" `
  --change-reason "seed-main-final-rebuild"
```

如果批处理已经生成了完整云端向量，只需要应用新版知识点归一化而不替换现有索引，可增加 `--skip-index`。

验证器升级后，可以离线重新验证历史订正，不会再次调用 VLM。当前支持数值、代数恒等式、矩阵、导数、原函数、极限以及由逗号分隔的多个验证断言：

```powershell
python -m m3_edu_memory.cli reverify-corrections `
  --db "data\seed_memory.db" `
  --model "seed-education-router-v1"
```

2026-09-22 的百炼真实付费调用作为历史链路证据保留。当前脚本默认使用火山方舟 Seed 2.1 Turbo 和 Doubao 多模态向量；API Key 仍只从 `ARK_API_KEY` 环境变量读取。
