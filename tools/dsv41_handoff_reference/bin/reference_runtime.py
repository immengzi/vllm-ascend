"""Validation-only MRV2 snapshot/replay, with all host copies on execution thread.

No Mooncake copy-plan/address helper is used. Production code is patched only
in a disposable source overlay by append_probe.py.
"""
import functools
import json
import os
import threading
from dataclasses import asdict
from pathlib import Path

import torch

from reference_contract import Plane, audit_wire_boundary, boundary_counters, check_snapshot_metadata, prompt_digest, progress_counters
from tensor_reference import capture,compare,digest,restore

ROOT=Path('/results')
CONTEXT=None
PENDING={}
PENDING_LOCK=threading.Lock()


def control():
    try:value=json.loads((ROOT/'reference-control.json').read_text())
    except FileNotFoundError:return None
    if not value['key'].isalnum() or Path(value['folder']).name!=value['folder']:
        raise ValueError('Invalid reference control path')
    return value


def ranks(runner):
    from vllm.distributed import get_tensor_model_parallel_rank
    return dict(dp=runner.vllm_config.parallel_config.data_parallel_rank,
                tp_rank=get_tensor_model_parallel_rank(),tp=runner.vllm_config.parallel_config.tensor_parallel_size)


def descriptors(runner):
    result=[]
    config=runner.kv_cache_config
    layout='pools' if getattr(config,'request_state_plan',None) is not None else 'native'
    context=runner.compilation_config.static_forward_context
    for group_id,group in enumerate(config.kv_cache_groups):
        for name in group.layer_names:
            layer=context[name];spec=getattr(layer,'spec',None)
            if spec is None:raise ValueError('Missing independent layer specification')
            values=layer.kv_cache
            if not isinstance(values,list) or len(values)!=1:raise ValueError('Unsupported cache binding')
            values=values[0]
            values=values if isinstance(values,tuple) else (values,)
            if name.endswith('.swa_cache'):kind='window'
            elif name.endswith('.compressor.state_cache'):kind='compressor'
            elif name.endswith(('.long_kv_cache','.indexer.k_cache')):kind='history'
            else:raise ValueError(f'Unsupported cache plane: {name}')
            for part,tensor in enumerate(values):
                if not isinstance(tensor,torch.Tensor) or tensor.ndim<3:raise ValueError('Invalid cache tensor')
                compression=int(getattr(spec,'tokens_per_state',1)) if kind=='history' else 1
                plane=Plane(name,part,kind,layout,int(tensor.shape[0]),int(tensor.shape[1]),
                            int(runner.cache_config.block_size),compression,str(tensor.dtype).removeprefix('torch.'),tuple(tensor.shape[2:]))
                plane.validate();result.append((plane,tensor,group_id))
    return result


def write_snapshot(runner,spec,rid,total,phase,*,logits=None,hidden=None):
    # Correctness-only stable snapshot. Never run this observer on a receive
    # thread or use it for performance measurements.
    torch.npu.synchronize()
    state=capture(descriptors(runner),runner._reference_resources[rid],total)
    stamp=ranks(runner)
    if phase=='source' and stamp['tp']!=4:raise ValueError('Reference source must actually execute TP4')
    counters=(boundary_counters(total,len(spec['original_prompt'])) if phase!='sample' else
              dict(original_prompt_tokens=len(spec['original_prompt']),**progress_counters(total)))
    state['metadata']=dict(schema=1,runtime_identity=spec['runtime_identity'],
        prompt_sha256=prompt_digest(spec['original_prompt']),counters=counters,
        producer_tp=4,source_rank=stamp['tp_rank'] if phase=='source' else None,
        expected_source_rank=None if phase=='source' else stamp['tp_rank']*2,phase=phase,request_id=rid,
        resources=runner._reference_resources[rid],layout=state['geometry'][next(iter(state['geometry']))]['layout'],
        execution_thread=threading.get_ident(),**stamp)
    if logits is not None:
        state['logits']=logits.detach().cpu().contiguous();state['logits_sha256']=digest(state['logits'])
    if hidden is not None:state['hidden']=hidden.detach().cpu().contiguous()
    folder=ROOT/spec['folder'];folder.mkdir(exist_ok=True)
    filename=f'{spec["key"]}-{phase}-t{total}-tp{stamp["tp_rank"]}.pt'
    torch.save(state,folder/filename)
    meta=dict(state['metadata'],file=filename,hashes=state['hashes'],planes=len(state['planes']))
    if logits is not None:meta.update(logits_sha256=state['logits_sha256'],argmax=state['logits'].argmax(-1).tolist())
    (folder/(filename+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
    return state


def install_runner(cls):
    execute=cls.execute_model;update=cls.update_requests;sample=cls.sample_tokens

    def execute_model(self,output,*args,**kwargs):
        global CONTEXT
        if not hasattr(self,'_reference_resources'):self._reference_resources={}
        for rid in output.finished_req_ids:self._reference_resources.pop(rid,None)
        for req in output.scheduled_new_reqs:self._reference_resources[req.req_id]=tuple(tuple(v) for v in req.block_ids)
        cached=output.scheduled_cached_reqs
        for rid,extra in zip(cached.req_ids,cached.new_block_ids):
            if extra is not None and rid in self._reference_resources:
                self._reference_resources[rid]=tuple(a+tuple(b) for a,b in zip(self._reference_resources[rid],extra))
        spec=control()
        selected=[] if spec is None else [rid for rid in output.num_scheduled_tokens if spec['key'] in rid]
        CONTEXT=None;self._reference_sample_context=None
        if selected:
            if len(selected)!=1 or len(output.num_scheduled_tokens)!=1:raise ValueError('Reference observer requires actual batch1')
            CONTEXT=(self,output,spec,selected[0],threading.get_ident())
            for req in output.scheduled_new_reqs:
                if req.req_id==selected[0]:
                    expected=spec['original_prompt'][:-1] if spec['action'] in ('export', 'observe_source') else spec['original_prompt']
                    if req.prompt_token_ids!=expected:raise ValueError('Observed request prompt differs from control')
        try:
            result=execute(self,output,*args,**kwargs)
            # MRV2 defers compute_logits to sample_tokens, after execute_model
            # has returned. Keep only this step's explicit observation context.
            self._reference_sample_context=CONTEXT
            return result
        finally:CONTEXT=None

    def sample_tokens(self,*args,**kwargs):
        global CONTEXT
        selected=getattr(self,'_reference_sample_context',None)
        CONTEXT=None if selected is None else (*selected[:4],threading.get_ident())
        try:return sample(self,*args,**kwargs)
        finally:
            CONTEXT=None;self._reference_sample_context=None

    def update_requests(self,output):
        update(self,output)
        if CONTEXT is None or CONTEXT[0] is not self:return
        runner,scheduled,spec,rid,thread=CONTEXT
        if thread!=threading.get_ident():raise RuntimeError('Wrong observation thread')
        new=next((req for req in output.scheduled_new_reqs if req.req_id==rid),None)
        if new is None:return
        total=len(spec['original_prompt'])-1
        if spec['action']=='replay':
            if new.num_computed_tokens!=total or output.num_scheduled_tokens[rid]!=1:
                raise ValueError('Reference bootstrap not applied')
            rank=ranks(self)['tp_rank'];source_rank=rank*2
            file=ROOT/spec['source_files'][str(source_rank)]
            snapshot=torch.load(file,map_location='cpu',weights_only=True)
            check_snapshot_metadata(snapshot['metadata'],spec['original_prompt'],spec['runtime_identity'])
            if snapshot['metadata']['source_rank']!=source_rank:raise ValueError('Wrong source replica')
            torch.npu.synchronize()  # Finish all fresh-allocation zeroing first.
            restore(snapshot,descriptors(self),self._reference_resources[rid],total)
            # Core update_requests has already zeroed the new allocations.
            # Finish the main-stream writes before publishing the loaded state.
            torch.npu.current_stream().synchronize()
            loaded=write_snapshot(self,spec,rid,total,'loaded')
            result=compare(snapshot,loaded)
            if not result['passed']:raise ValueError(f'Independent restore audit failed: {result}')
        elif spec['action']=='observe':
            with PENDING_LOCK:event=PENDING.pop(rid,None)
            if event is None:raise ValueError('No fenced READ event for observed handoff')
            if new.num_computed_tokens != total or output.num_scheduled_tokens[rid] != 1:
                raise ValueError('Observed D must compute exactly the last prompt token')
            if event['snapshot'] is not None:
                audit_wire_boundary(event['snapshot'],total,len(spec['original_prompt']))
            write_snapshot(self,spec,rid,total,'loaded')

    cls.execute_model=execute_model
    cls.update_requests=update_requests
    cls.sample_tokens=sample_tokens


def install_model(cls):
    original=cls.compute_logits
    def compute_logits(self,hidden,*args,**kwargs):
        logits=original(self,hidden,*args,**kwargs)
        if CONTEXT is None or logits is None:return logits
        runner,output,spec,rid,thread=CONTEXT
        if thread!=threading.get_ident():raise RuntimeError('Host copy outside execution thread')
        index=runner.req_states.req_id_to_index[rid]
        before=int(runner.req_states.num_computed_tokens_np[index])
        total=before+int(output.num_scheduled_tokens[rid])
        boundary=len(spec['original_prompt'])-1
        valid_logits=logits[:1]
        valid_hidden=hidden[:1]
        if spec['action'] in ('export','observe_source'):
            if total!=boundary:raise ValueError('Export must stop exactly at T')
            write_snapshot(runner,spec,rid,total,'source',logits=valid_logits,hidden=valid_hidden)
        elif total in (boundary+1,boundary+2,boundary+32,boundary+128,boundary+130):
            # A sample is a continuation state, not a new T=N-1 handoff.
            state=write_snapshot(runner,spec,rid,total,'sample',logits=valid_logits,hidden=valid_hidden)
            folder=ROOT/spec['folder'];stamp=ranks(runner)
            filename=f'{spec["key"]}-sample-t{total}-tp{stamp["tp_rank"]}.pt'
            state['metadata']['original_boundary']=boundary
            actual_tables=[]
            for group,ids in enumerate(runner._reference_resources[rid]):
                factor=runner.block_tables.blocks_per_kv_block[group]
                expected=[bid*factor+offset for bid in ids for offset in range(factor)]
                observed=runner.block_tables.input_block_tables[group][0,:len(expected)].cpu().tolist()
                if observed!=expected:raise ValueError(f'Actual input block-table mismatch in group {group}')
                actual_tables.append(observed)
            state['metadata']['actual_input_block_tables']=actual_tables
            model_state=runner.model_state
            lookback=getattr(model_state,'lookback_token_ids',None)
            if lookback is not None:state['engram_lookback']=lookback[0].detach().cpu().contiguous()
            hash_module=getattr(self.model,'engram_hash',None)
            if hash_module is not None and hasattr(hash_module,'_reference_hash'):
                row=int(hash_module._reference_starts[1].cpu())-1
                if row<0 or int(hash_module._reference_positions[row].cpu())!=total-1:
                    raise ValueError('Stale Engram hash observation')
                state['engram_hash']=hash_module._reference_hash[row].detach().cpu().contiguous()
                epoch=int(hash_module._reference_epoch.cpu())
                if epoch<=getattr(self,'_reference_last_epoch',-1):raise ValueError('Engram observation epoch did not advance')
                self._reference_last_epoch=epoch;state['metadata']['engram_epoch']=epoch
                state['engram_lookup']={}
                for layer in self.model.layers:
                    engram=getattr(layer,'engram',None)
                    if engram is not None:
                        if not hasattr(engram,'_reference_lookup'):raise ValueError('Missing actual Engram lookup observation')
                        state['engram_lookup'][str(layer.layer_idx)]=engram._reference_lookup[row].detach().cpu().contiguous()
            torch.save(state,folder/filename)
            meta=dict(state['metadata'],file=filename,hashes=state['hashes'],planes=len(state['planes']),
                      logits_sha256=state['logits_sha256'],argmax=state['logits'].argmax(-1).tolist())
            (folder/(filename+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
        return logits
    cls.compute_logits=compute_logits
    initialize=cls.__init__
    def init(self,*args,**kwargs):
        initialize(self,*args,**kwargs)
        h=getattr(self.model,'engram_hash',None)
        if h is not None:
            cfg=kwargs.get('vllm_config',args[0] if args else None)
            count=cfg.scheduler_config.max_num_batched_tokens+128
            layers,width=h.multipliers.shape;cols=(width-1)*h.primes.shape[-1]
            device=h.multipliers.device
            h._reference_hash=torch.empty((count,layers,cols),device=device,dtype=torch.int32)
            h._reference_positions=torch.empty(count,device=device,dtype=torch.int64)
            h._reference_starts=torch.empty(cfg.scheduler_config.max_num_seqs+129,device=device,dtype=torch.int32)
            h._reference_epoch=torch.zeros((),device=device,dtype=torch.int64)
            forward=h.forward
            def hash_forward(input_ids,positions,starts,*a,**kw):
                result=forward(input_ids,positions,starts,*a,**kw)
                h._reference_hash[:result.shape[0]].copy_(result)
                h._reference_positions[:positions.numel()].copy_(positions)
                h._reference_starts[:starts.numel()].copy_(starts)
                h._reference_epoch.add_(1)
                return result
            h.forward=hash_forward
            def record_lookup(module,inputs):
                value=inputs[1]
                if not hasattr(module,'_reference_lookup'):
                    module._reference_lookup=torch.empty((count,*value.shape[1:]),dtype=value.dtype,device=value.device)
                if value.shape[0]>count:raise ValueError('Engram lookup observer capacity exceeded')
                module._reference_lookup[:value.shape[0]].copy_(value)
            for layer in getattr(self.model,'layers',[]):
                if getattr(layer,'engram',None) is not None:layer.engram.register_forward_pre_hook(record_lookup)
    cls.__init__=init


def install_receiver(cls):
    # Receiver hooks only enqueue CPU metadata. No .cpu(), NPU op, or model
    # state access is allowed from this thread.
    original=cls._handle_requests
    def read(self,remote_engine_id,remote_host,remote_port,requests):
        failed=original(self,remote_engine_id,remote_host,remote_port,requests)
        spec=control()
        if spec is not None and spec['action']=='observe':
            with PENDING_LOCK:
                for rid,meta in requests.items():
                    if rid not in failed and spec['key'] in rid:
                        if rid in PENDING:raise ValueError('Duplicate loaded observer event')
                        PENDING[rid]=dict(snapshot=meta.state_snapshot,expected_source_rank=self.tp_rank*2,
                                          actual_source_rank=None,dma_fenced=True)
        return failed
    cls._handle_requests=read
