# 数学 VLM 接入

## 数据隔离

推理代码只读取 `attempts.orig_q`、`attempts.image_source_path` 和原始图片。它不会查询 `benchmark_truth`。推理完成后，`evaluate` 命令才会将预测与 FERMAT 的 `has_error` 和映射后的错误类型比较。

## 模型配置档案

推荐从 `config/vlm_providers.example.json` 复制配置。每个 profile 保存协议、服务地址、模型名和 API Key 对应的环境变量名，不保存密钥本身。当前适配所有支持 OpenAI Chat Completions 图片消息格式的服务；新增原生协议时，只需实现 `VisionClient` 并在 `providers.py` 注册。

`diagnose` 和 `process-jobs` 均可使用：

```powershell
--provider-config "config\vlm_providers.json" --provider-profile "volcengine-seed-2.1-pro"
```

## 固定评测与字节模型验收

使用 `create-eval-suite` 创建固定样本列表。选择过程由 seed 决定、每个 `orig_q` 只取一个版本，并交替选择有错和无错样本。`diagnose --evaluation-suite <名称>` 会让每个模型处理完全相同的图片；配合 `--only-unprocessed` 可以安全续跑。随后使用 `evaluate --suite ... --provider ... --persist` 保存结果，最后用 `compare-eval-suite` 查看各模型最近一次结果。

有 FERMAT 真值的指标包括错误识别、映射后的错误类型和参考答案。首错步骤、bbox 和知识点缺少相应真值，只能用于检查模型是否提供了字段，不能用这些覆盖率宣称定位准确率或知识点准确率。教师在查询报告中的修订不进入离线评测。

每次推理都会产生新的 `analysis_runs.run_id`，保存模型、provider、prompt版本、原始响应和解析结果。诊断写入 `diagnoses`、`diagnosis_steps` 与 `corrections`，同时生成带模型和提示词版本的情景记忆。VLM结论初始状态为 `hypothesis`，不会覆盖 benchmark reference 或以前的模型版本。

## 按模型配置图片预处理

原始图片始终保留为不可变证据。系统只为本次 VLM 请求生成推理副本，并把源图/推理图尺寸、字节数、SHA-256、缩放和压缩参数写入 `analysis_runs.preprocessing_json`。整图等比例缩放保持归一化 bbox 坐标不变。

预处理不是统一压缩规则，而是 provider profile 的一部分：

- 百炼档案保留为历史兼容配置，不再作为后续对比基线。
- 通用默认档案以保真为主：约 4 百万像素、长边 3072、4 MB。
- 火山方舟 Seed 1.8/2.0 档案允许约 903 万像素、长边 4096、8 MB，并使用 `image_detail=xhigh`。旧模型应把 `max_pixels` 降至 4014080。
- `image_preprocessing.enabled=false` 可以完全关闭推理副本转换，让服务商执行原生动态分辨率处理。

手写数学依赖上下标、根号、分数线和细小符号，因此字节模型默认采用高保真预算。API 到位后先在固定评测集上测试该档案，再依据错误样本决定是否启用局部高清复核。火山方舟 Chat API 的图片上限因模型代际而异，配置实际模型 ID 时应再次核对对应版本。

调用观测保存在 `analysis_runs`：推理图片尺寸/体积、预处理元数据、总延迟、请求次数以及服务商返回的 token usage。聚合接口：

```http
GET /v1/analysis/metrics?model=qwen3.7-plus
```

## OpenAI兼容视觉端点

支持接受 Chat Completions 多模态消息的云端API、vLLM或其他OpenAI兼容服务。密钥只从环境变量读取，不写入配置文件或数据库。

```powershell
$env:VLM_API_KEY="your-key"
$env:PYTHONPATH="vendor\python;..\dataset\.python-libs;src"

python -m m3_edu_memory.cli diagnose `
  --db "data\education_memory.db" `
  --base-url "https://your-endpoint.example/v1" `
  --model "your-math-vision-model" `
  --api-key-env VLM_API_KEY `
  --domain clc `
  --limit 10 `
  --only-unprocessed
```

加入 `--audit-output-dir "outputs\audit"` 后，每条带订正的成功诊断会自动生成可审计 PNG。V3 prompt 会要求模型返回结构化订正和可验证表达式；系统验证数值、代数、导数、原函数、极限和显式数值矩阵等式，并支持逗号分隔的多断言验证。无完整 SymPy 时使用内置精确有理数与矩阵验证器，再写入 `corrections` 与 `Correction` 记忆节点。

```powershell
python -m m3_edu_memory.cli weaknesses-vlm `
  --db "data\education_memory.db" `
  --model "qwen3.7-plus" `
  --domain clc
```

该命令只读取最新 VLM 诊断和订正验证记录，不连接 `benchmark_truth`。输出中的薄弱点保持 `hypothesis` 状态，并列出全部支持证据 attempt IDs。

每次 VLM 诊断、教师修改或激活订正导致掌握状态发生变化时，系统会保存一份不可变快照；无变化的重建不会产生重复记录：

```http
GET /v1/memory/mastery-history?model=<字节模型ID>&domain=clc&limit=100
```

对于不需要鉴权的本地端点，使用 `--api-key-env -`。如果服务不支持 `response_format={"type":"json_object"}`，增加 `--no-json-mode`。程序仍会提取响应中的JSON并做严格字段校验。

## 单条重跑

```powershell
python -m m3_edu_memory.cli diagnose `
  --db "data\education_memory.db" `
  --base-url "http://127.0.0.1:8000/v1" `
  --model "local-vlm" `
  --api-key-env - `
  --attempt-id "fermat-00002-00010-000116"
```

## 无模型集成测试

此模式不会调用模型，只验证真实图片读取、JSON校验、版本化写入、记忆图连接和评测流程：

```powershell
python -m m3_edu_memory.cli diagnose `
  --db "data\vlm_integration_test.db" `
  --attempt-id "fermat-00002-00010-000116" `
  --fixture-response "config\example_diagnosis.json"
```

## 离线评测

```powershell
python -m m3_edu_memory.cli evaluate `
  --db "data\education_memory.db" `
  --model "your-math-vision-model"
```

当前指标包括是否有错准确率、错误类型严格匹配率和参考答案严格匹配率。首错步骤与 bbox 没有 FERMAT 真值，因此只报告覆盖率。

参考答案还会报告数学等价匹配率。该指标保守处理常见 LaTeX 分数、单位、等式末项和安全符号恒等式，不能解析的开放式证明仍按不可比较处理。

## 检索VLM记忆

下面的查询只读取指定模型最新一次成功诊断，不连接 `benchmark_truth`：

```powershell
python -m m3_edu_memory.cli query-vlm `
  --db "data\education_memory.db" `
  --model "your-math-vision-model" `
  --domain clc `
  --errors-only `
  --limit 10
```
批量分层试跑（错误/正确参考标签仅用于离线抽样分层，不进入模型输入）：

```powershell
.\scripts\test_bailian.ps1 -Limit 6 -MaxWorkers 3 -StratifiedPilot
```

完整结果写入 `outputs\bailian-v2-audit\batch-results.json`，命令行只显示进度和成功/失败计数。
