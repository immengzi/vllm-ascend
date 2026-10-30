# DeepSeek V4.1 历史/请求状态分离与 PD 调试

本分支为 `feat/dsv41-state-pd-mooncake`，父分支是
[`feat/dsv41-history-request-state`](https://github.com/immengzi/vllm-ascend/tree/feat/dsv41-history-request-state)。
父提交 `f77bee84a5299c2e62d749409a29352aaf7e48b9` 保存准确已测源码。
PD 分支是待上卡调试的实现；离线协议检查通过不等于模型或 DMA 正确性通过。

固定依赖：vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，
Ascend 基线 `8fde6d4e335fe300e2b949aab0e31659cce0811d`，
原镜像 ID `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`。
不要求为了这次调试升级主线或重编译原有算子。

## 先做离线检查

```bash
python3 examples/dsv41_state/check_preserved_source.py
python3 tests/ut/distributed/kv_transfer/kv_p2p/test_dsv41_state_protocol.py
```

第一个命令在 PD 分支校验父提交的六个归档文件，需要 Git 中保留该父提交；
不会把当前 PD 实现标为“已测”。第二个命令只需标准 Python，直接检查当前协议及接入方法。
固定镜像内还可执行 `python3 tests/ut/distributed/kv_transfer/kv_p2p/test_dsv41_state_wire.py`，
使用 `msgspec==0.22.0` 验证 3 项真实消息编解码。该测试无需 torch/NPU。
运行时代码已移入 `vllm_ascend/models/deepseek_v41/request_state.py` 和 `state_kernels.py`，
examples 中不再保留第二套执行模块。当前 planner 只使用框架/CLI 的预算。

## 在新环境调试

在新建的、与固定镜像匹配的隔离容器中检出此分支；不要覆盖正在服务的安装。
确认 Python 实际导入此 checkout，并保留镜像自身的 CANN/Python/HCCL 环境。
launcher 会设置 MRV1、单 token 窗口路径和可写审计目录：

```bash
python3 examples/dsv41_state/launch.py --results-dir /results/state -- \
  /path/to/DeepSeek-V4.1-Flash-W8A8 \
  --tensor-parallel-size 4 --data-parallel-size 3 --enable-expert-parallel \
  --block-size 128 --kv-cache-memory-bytes 3576615936 \
  --max-num-seqs 48 --max-num-batched-tokens 8192 --max-model-len 150000 \
  --quantization ascend --no-enable-prefix-caching --async-scheduling \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_fused_mc2":1,"enable_dsa_cp":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}' \
  --kv-transfer-config '{"kv_connector":"MooncakePullConnector","kv_role":"kv_producer","kv_port":14500,"kv_load_failure_policy":"fail","kv_connector_extra_config":{"dsv41_state_pd":true,"state_model_id":"REPLACE_WITH_IDENTICAL_WEIGHT_MANIFEST_SHA"}}'
```

这是服务参数示例，尚未运行 PD。D 端使用相同模型、精度、布局和 TP，
改 `kv_role` 为 `kv_consumer`，使用自身未占用的端口及结果目录。
`state_model_id` 必须由双方设置成相同的真实权重身份；配置相同不能证明权重相同。
复用仓库 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` 的 V2 pull 代理，
它转发交接 metadata，保留原始 D prompt，并丢弃 P 的占位采样结果。
不要使用 layerwise 代理、AscendStore 或旧 V1 connector。

先用同一 PD 分支、不带 `--kv-transfer-config` 的服务建立非 PD 数值对照。
第一组建议沿用已知可装载的每侧 TP4/DP3/EP12；两侧合计 24 张卡。
更小拓扑的模型能否装载未经此项目验证。设备、网络、时间窗应由新环境所有者自行确认。

## 必须看到的证据

- 真实 source/destination ID 不同，注册的历史容量与状态容量可不同。
- 每个接收 rank 有 `DSV4.1 state READ complete`，记录 transfer、边界、字节、范围和耗时。
- `kv_load_failure_policy=fail`，收到完整 READ 完成后才允许首次 D forward；不得回退重算 prompt。
- N 为原始 prompt 长度。交接 T=N−1 的状态，D 处理最后一个 prompt token 后产生第一个真实输出。
- 对照首步和连续 Decode 的输出及窗口、索引、压缩状态；HTTP 200 不能替代这些检查。
- 取消、部分失败、超时和复用期间，不得提前回收任何一端的在途资源。

`test_state_kernels.py`、`test_state_cp_attention.py`、`test_state_graph_replay.py` 需要 NPU。
历史生命周期客户端要求 `/model`、`/results/config.json` 和 `/results/state` 布局，
应使用新的截止时间配置，不能复用旧实验目录。

完整的交接流程、兼容边界、异常处理和硬件验证门禁见
[PD 适配设计](../../docs/source/developer_guide/Design_Documents/dsv41_state_pd.md)。
