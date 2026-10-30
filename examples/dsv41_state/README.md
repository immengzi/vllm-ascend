# DeepSeek V4.1 历史与请求状态分离：已测源码分支

本分支固定在 vLLM-Ascend `8fde6d4e335fe300e2b949aab0e31659cce0811d`，
配套 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
它保存既有原型的准确运行源码，便于复现和作为 PD 开发的父分支。
不是相对当前主线的性能结论，也不是正式合入版本。

历史池、SWA 请求状态池、压缩器请求状态池分别管理，容量启动时固定。
历史使用 token 页表，两类状态各按请求占用一槽。不存在池间物理内存互借。
准确安装后的四个接入文件和两个核心模块的 SHA 见 `tested-source.json`；
为保留字节级对照，原型钩子和历史命名未在这个分支内重写。
新增 launcher 和校验脚本不在历史性能证据内。

## 使用

在固定镜像内以 editable 方式安装本 checkout。不要升级配套 vLLM 或编译依赖。
先运行离线源码检查：

```bash
python3 examples/dsv41_state/check_preserved_source.py
```

launcher 为子进程及其 worker 设置实验模块的 import 路径：

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
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_fused_mc2":1,"enable_dsa_cp":true,"enable_flashcomm1":true,"enable_shared_expert_dp":true}'
```

这是参数示例。部署还需要匹配原镜像的 HCCL、可见设备、CPU/共享内存与模型加载环境。
不要在非授权设备上运行。状态日志目录需要所有 worker 可写。
`MENGZI_*` 是保留原型已有接口，没有新增为正式产品配置。

## 边界与验证

已有证据仅覆盖 DSV4.1 Flash W8A8、TP4/DP3/EP12、MRV1、PP1、生成图、异步及 DSA CP。
关闭前缀缓存和推测解码；本分支仍显式拒绝 KV transfer。
MRV2 只保存视图接入代码，尚无运行正确性结论。

`test_state_kernels.py`、`test_state_cp_attention.py` 和 `test_state_graph_replay.py`
需要 NPU，只能在独立授权环境运行。生命周期测试及其辅助客户端保留供固定 `/model`、
`/results/config.json`、`/results/state` 布局使用。新环境需先生成匹配的配置，不能套用旧截止时间。
这些是历史测试源码，不表示本次提交已重新通过硬件测试。

正式工程化仍需替换实验哨兵准入、明确多资源事务及失败回滚。
PD 适配由另一个分支继承本分支，单独记录源码确认、离线检查和待上卡验证项。
