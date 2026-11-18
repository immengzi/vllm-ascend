# DeepSeek V4.1 三池：2026-11-18 审阅快照

本分支保存当前 MRV2 历史缓存与请求状态分离实现，以及独立参考 R 的审计工具。
这是待继续验收的开发快照，不代表 W4A8 或完整 P/D 生命周期已通过验证。

## 设计与入口

历史 KV 和索引随上下文增长；SWA 窗口与压缩器状态按请求保留。
三池分别管理历史页、窗口槽和压缩器槽；先计入状态和工作区，再为历史页分配剩余预算。
沿用已有缓存管理器、联合准入与回滚机制；没有恢复写死的 3.33 GiB 预算。

功能默认关闭，通过 additional-config 显式启用：

```json
{"enable_deepseek_v41_request_state_pools": true}
```

已有 additional-config 时应合并此字段。它是严格布尔值；非法启用配置报错。
当前仅支持 DeepSeek V4.1 Flash 的未打包缓存和 MRV2，
要求 PP=PCP=DCP=1、BF16 缓存几何、开启 hybrid cache management；
不支持前缀缓存、SWA bounded replay、extensible KV、推测解码及 sleep mode。
P/D 使用 MooncakePullConnector，并要求一致的模型身份和精度及失败关闭策略。
同 TP 和 P TP4→D TP2 有代码路径；其他异构 TP 组合拒绝。

P 处理原始输入前 N−1 个 token，D 加载对应状态后处理最后一个输入 token。
TP4→TP2 为每个 D rank 选取同一个 P 副本的所有缓存平面：D0←P0、D1←P2。
源租约、聚合完成通知及未完成 DMA 隔离延续原有路径。

## 审阅顺序

1. `vllm_ascend/models/deepseek_v41/cache_config.py`：开关、适用条件和三池规划。
2. `vllm_ascend/core/kv_cache_config_builder.py`、`kv_cache_pool_plan.py` 和
   `request_state_manager.py`：规划接入、独立容量、准入和回滚。
3. `vllm_ascend/models/deepseek_v41/request_state.py`、`window_state.py`、
   `state_kernels.py` 和 MRV2 runner：状态绑定、窗口暂存和图执行。
4. `vllm_ascend/distributed/kv_transfer/kv_p2p/dsv41_state.py` 及
   `mooncake/pull_scheduler.py`、`pull_worker.py`：协议、边界、rank 映射与释放。
5. `tests/ut/core/test_dsv41_state_tp_handoff.py`：映射、错误拓扑、模型指纹及源租约测试。
6. [独立参考与审计工具](../../../tools/dsv41_handoff_reference/README.md)：观测和业务实现分离。

## 验证边界

| 层次 | 已有证据 | 限制 |
| --- | --- | --- |
| 源码 | 当前 6 个业务文件及 2 个测试相对 2304428 的改动已逐文件提取 | 本地部分无关文件读取异常，保留 GitHub 基线对应文件，未作为删除或空文件上传 |
| CPU | 本次复跑参考契约 10、调度 4、观测控制 3、smoke 门禁 7，共 24 项通过；8 个业务/测试文件 Python AST 与基础 Ruff 检查通过 | 当前 Python 无 torch，完整张量回归及业务 pytest 未复跑 |
| 格式 | 基础 Ruff 检查通过 | 保留源码原字节；两份业务/测试文件未通过独立 Ruff format 检查；完整 format.sh ci 因审阅目录不是完整 Git checkout 未完成 |
| 已有 NPU | W8A8、单机 EP12、P TP4/DP3、D TP2/DP6、上下文 4096、实际 DP0/批次1；N128/129/130/257 各两次。原布局 P 8 次导出，两种 D 布局各 8 次恢复；配对 308 个输出 token 和 API 概率一致，44 组原始样本对一致 | 只采样输出步 1/2/32/128/130，按输出长度裁剪；没有每一步 raw |
| 仍待完成 | 匹配 W4A8 的独立 R、实际 P 源状态、每 D rank SDK READ、D 加载状态及连续输出、故障生命周期、无观测同步复验、性能 | 先前 EP12/4096 资格不能套用到新模型和拓扑；真实 READ 发生不等于完整正确性通过 |

上述 NPU 结论来自已有验收记录，不是本次发布重新执行的结果。
当前工具包含随后修改的审计逻辑，不能宣称这些修改已随先前 NPU 实验验证。

## 下一轮部署

后续默认模型为 DeepSeek-V4.1-Flash-1017-w4a8，单台 A3 内 P/D 各 8 设备。
兼容当前 PCP=1 约束的候选是 P TP4/DP2、D TP2/DP4。
容量、Engram 主机内存、通信端口和模型完整性仍待实测，W4A8 必须重新生成匹配 R。
性能需在正确性门禁后使用干净服务与官方 benchmark，每轮至少 100 请求。

## 版本

业务快照基于 `2304428a8d64d0b417b2dcde3c571bd597fc7a07` 加当前工作区改动；
完整特性比较基线为 `1f49be4c3e5e52c990a10cfcfac9626a4af08667`。
仅凭分支名不能确认部署身份；运行时仍需绑定实际源码、镜像、权重与拓扑。

既有平台补丁负责把独立状态容量接入 vLLM 缓存管理；MRV2 runner 负责工作区、
状态视图和执行边界，不能只靠 connector 替代这些职责。
后续需推动相应缓存规划/生命周期扩展点上游化，再移除临时平台补丁。
当前分支用于用户仓库内部审阅，没有宣称满足上游合入条件。
