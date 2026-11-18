"""Attach test fixtures only to a disposable plugin/core source overlay."""
import sys
from pathlib import Path

plugin=Path(sys.argv[1]);core=Path(sys.argv[2])
items=[
    (plugin/'vllm_ascend/models/deepseek_v41/model.py',
     'from reference_runtime import install_model\ninstall_model(AscendDeepseekV41LLMForCausalLM)'),
    (plugin/'vllm_ascend/worker/v2/model_runner.py',
     'from reference_runtime import install_runner\ninstall_runner(NPUModelRunner)'),
    (plugin/'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py',
     'from reference_runtime import install_receiver\ninstall_receiver(MooncakePullRecvingThread)'),
    (core/'vllm/v1/core/sched/scheduler.py',
     'from scheduler_reference import install\ninstall(Scheduler)'),
]
# Keep the successful native SDK call, session and actual TP rank in the same
# log record. An empty bucket and a failed call do not reach this logger.
worker = items[2][0]
source = worker.read_text()
for anchor, replacement in (
    ('"local_ip %s local_device_id %s remote_session_id %s",',
     '"local_ip %s local_device_id %s remote_session_id %s source_rank=%d",'),
    ('            self.tp_rank,\n            session_id,\n        )',
     '            self.tp_rank,\n            session_id,\n            remote_tp_rank,\n        )'),
):
    if source.count(anchor) != 1:
        raise ValueError('Frozen native READ logging site differs')
    source = source.replace(anchor, replacement)
compile(source, str(worker), 'exec')
worker.write_text(source)
for path,tail in items:
    text=path.read_text()+'\n# Isolated reference-R validation fixture.\n'+tail+'\n'
    compile(text,str(path),'exec');path.write_text(text)
