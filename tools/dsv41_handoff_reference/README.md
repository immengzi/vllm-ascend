# 独立混合参考 R 与 P/D 审计工具

这是 2026-11-18 工作副本中可审阅的验证代码，保留原始内容。
机器专用启动器、SSH 配置、截止守护、原始张量与过期 SHA 清单不随本目录发布。
这些工具没有自动注册到业务包，`append_probe.py` 只用于一次性源码副本。

R 使用原布局 P/TP4 计算前 N−1 token，独立导出有效逻辑状态，
再独立 scatter 到 D/TP2 的新资源；D 计算最后一个输入 token 并连续生成。
导出/恢复不调用生产 connector 的 copy plan 或地址计算函数。

- `reference_contract.py` / `tensor_reference.py`：独立行映射、边界与逐位比较。
- `scheduler_reference.py` / `reference_runtime.py`：调度边界、模型线程快照、延迟采样及 Engram 观测。
- `audit_reference.py` / `audit_transport.py`：源/加载状态、重复参考、实际 READ 与连续数值审计。
- `smoke.py` / `observer_controls.py`：API 来源计数、部署资格及观测控制。
- `append_probe.py`：对一次性源码插入钩子，记录原布局实际 READ 源 rank。

本次修补接受 P 侧已裁剪的 N−1 输入，记录原布局实际源 rank，
并请求和核对完整 top-5 API 概率及连续样本身份。
完整 PyTorch 回归尚未完成，不能视为正式验收工具包。
`smoke.py` 保留的默认时窗已过期；实际请求必须显式提供适用的新时窗和 endpoints。
这些代码不是单机 W4A8 的现成部署启动器。

在仓库根目录运行无需 torch 的检查：

```bash
python3 -m unittest discover -s tools/dsv41_handoff_reference/tests -p 'test_reference_contract.py'
python3 -m unittest discover -s tools/dsv41_handoff_reference/tests -p 'test_scheduler_reference.py'
python3 -m unittest discover -s tools/dsv41_handoff_reference/tests -p 'test_observer_controls.py'
python3 -m unittest discover -s tools/dsv41_handoff_reference/tests -p 'test_smoke_gate.py'
```

本次共 24 项通过。有可用 CPU PyTorch 后运行完整目录：

```bash
python3 -m unittest discover -s tools/dsv41_handoff_reference/tests
```

HTTP 成功、`execution_passed` 或单个用例数值一致都不代表全套正确性通过。
真实传输、连续计算、故障回收和撤去观测同步后的稳定性分别验收。
观测会引入设备同步，因此观测服务的数据不能直接作为性能结果。
