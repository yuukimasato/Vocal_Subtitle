# 阶段五：全局 ASR 默认路径切换实施计划

关联设计：`docs/superpowers/specs/2026-07-26-phase-five-default-path-design.md`

## 实施目标

让离线 Pipeline 默认优先使用全局 ASR + 物理对齐，并在全局依赖、资源、执行或结果校验失败时完整回退到旧分段 ASR。所有路径必须在统计、缓存、CLI、WebUI、任务历史和 benchmark 中可观察。

## 变更边界

保留现有 faster-whisper、whisper.cpp、FunASR 和分段 Pipeline；不修改流式 ASR 协议；不在模块导入时加载 WhisperX/Torch；不撤销工作区中其他用户改动。仅修改与路径选择、结果诊断、缓存兼容、验收脚本和相关测试直接相关的代码。

## 步骤 1：路径状态与配置契约

目标：建立统一的运行路径状态，兼容现有 `global_asr.enabled` 和 `fallback_to_segmented`。

任务：

- 在 `PipelineStats` 中增加 `asr_path`、`global_attempted`、`fallback_category`，并序列化到任务结果。
- 增加稳定的降级分类和异常归类函数，脱敏错误文本。
- 让离线默认 profile 开启全局优先；保留 `enabled=false` 的显式 legacy 行为。
- 增加任务级 `auto/global/segmented` 覆盖，覆盖只存在于当前 Pipeline 实例。
- 在 streaming 入口强制 legacy，并写入 `explicit_legacy` 或对应路径诊断。

验收：配置加载、任务覆盖和流式强制路径均可用；导入 `vocal_subtitle` 不触发 WhisperX 导入。

## 步骤 2：Pipeline 全局优先与安全降级

目标：全局成功时只使用完整全局结果；失败时只使用完整 legacy 结果。

任务：

- 在离线 `run()` 开始阶段解析有效路径，避免 skeleton/streaming 等显式旧路径误入全局分支。
- 全局尝试前设置 `global_attempted`，成功后设置 `asr_path=global`。
- 按依赖缺失、资源不足、执行异常、无效结果区分降级类别。
- 全局结果为空、状态 degraded 或物理校验不通过时，不接受部分结果，按配置回退或失败。
- legacy 正常运行时设置 `asr_path=legacy`；由全局失败触发时设置 `legacy_degraded`。
- 保持现有全局物理边界、严格断句和最终校验调用顺序。

验收：fake engine 可覆盖成功、缺失依赖、异常、空结果和不可降级分支；不会出现全局/legacy 事件混合。

## 步骤 3：全流水线缓存与任务历史

目标：默认全局请求不被旧 legacy 结果静默满足。

任务：

- 为完整结果增加 schema/path 标识，并将路径策略纳入配置或缓存身份。
- 恢复缓存时反序列化完整 stats，保留全局诊断和降级信息。
- 对旧缓存缺失路径标识的情况执行兼容判断：默认全局请求不直接恢复；显式 legacy 请求可恢复旧结果并标记 legacy。
- 确认 WebUI 的 full-pipeline cache 可用性检查与 Pipeline 规则一致。

验收：同一音频在 global/segmented 配置下不会互相命中错误结果；缓存恢复后的 stats 与原结果路径一致。

## 步骤 4：CLI、WebUI 与任务状态

目标：用户能选择路径并看到是否降级。

任务：

- CLI `run` 增加路径选项，帮助文本说明 `auto/global/segmented`。
- CLI 完成输出显示实际 `asr_path`；降级时显示分类和原因摘要。
- WebUI run 请求支持路径覆盖，并将结果 stats、任务历史和 WebSocket complete 事件透传。
- 更新 Pydantic 任务/统计响应模型和字幕事件归一化逻辑，保持旧字段兼容。
- 不在前端根据缺失字段猜测路径，直接使用显式诊断字段。

验收：CLI help、API 请求和 WebSocket payload 均能看到路径；旧请求不传新字段仍按 profile 工作。

## 步骤 5：阶段五 benchmark 验收门

目标：提供真实素材对照和“是否具备切换资格”的机器可读报告。

任务：

- `scripts/run_benchmarks.py` 增加 `--asr-path auto|global|segmented`，记录每个场景的实际路径和 fallback 信息。
- 增加最少三类场景校验及 ground-truth/metadata 完整性检查。
- 汇总 `rollout_eligible`、路径覆盖、降级数、物理越界、时间误差和健康度。
- CI 模式在素材不足或指标未达标时返回非零状态；不把空目录视为通过。
- 文档说明当前 benchmark 素材缺口和补齐后执行命令。

验收：无素材时报告明确为不可发布；使用合成/fixture 数据时只验证脚本逻辑，不冒充真实质量验收。

## 步骤 6：测试与文档同步

目标：完成无模型回归并同步用户可读文档。

测试文件：

- 新增或扩展 `tests/test_phase_five.py`：路径策略、降级分类、统计序列化和缓存兼容。
- 扩展 `tests/test_physical/test_phase_three.py`：全局成功/失败入口与不混合结果。
- 扩展 `tests/test_cli.py`、`tests/test_webui.py`：CLI 选项、结果状态和 API 兼容。
- 扩展 `tests/test_deployment_defaults.py`：离线默认全局优先、流式 legacy、可显式关闭。
- 为 benchmark 脚本增加无音频的扫描/指标单元测试，避免加载模型。

文档文件：

- 新增 `docs/phase-five-default-path-development.md`，说明配置、运行、故障排查、回滚和验收命令。
- 更新 `docs/Vocal_Subtitle-离线高精度字幕系统设计文档.md` 阶段状态，区分代码完成与真实素材验收完成。
- 更新 `README.md` 和 `docs/DEPLOYMENT.md` 的离线默认路径、WhisperX 可选依赖及降级说明。

## 验证顺序

1. `git diff --check`。
2. `uv run pytest` 的无模型阶段五、阶段三、阶段四、部署默认值、CLI、WebUI 测试。
3. `uv run python scripts/run_benchmarks.py --ci` 的素材完整性检查；当前素材不足时预期为不可发布状态。
4. 若环境安装 WhisperX 且具备真实素材，分别执行 global/segmented/auto 对照并保存报告。
5. 检查 `git status`，确保不覆盖用户已有未提交改动。

## 完成定义

- 离线默认路径实际优先尝试全局 ASR；失败时完整、可诊断地回退。
- 流式和显式 legacy 行为保持稳定。
- 路径状态、降级原因和统计在 Pipeline、缓存、CLI、WebUI 和任务历史中一致。
- 无模型测试通过；benchmark 验收门能对素材不足和指标失败正确阻断发布。
- 开发文档、README、部署说明和主设计文档与实际行为一致。
