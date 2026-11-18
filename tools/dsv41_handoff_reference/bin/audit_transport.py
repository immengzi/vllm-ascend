"""CPU-only audit of actual P/D observations against the qualified hybrid R.

Input is a consolidated evidence bundle, not running service memory. This
command neither issues HTTP requests nor invokes a transfer address planner.
"""
import argparse
import ast
import re
import hashlib
import json
from pathlib import Path

from audit_reference import audit_loaded,audit_samples,load,sample_steps
from reference_contract import audit_wire_boundary,check_hybrid_reference,prompt_digest
from tensor_reference import compare
from smoke import NATIVE_READ, TRANSFER_READ, check_read_log, check_reference_deployment


def read_receipts(raw, row, rank, rid, native):
    """Use the exact engine request ID from the loaded snapshot."""
    receipts = []
    for line in raw.splitlines():
        if native:
            match = NATIVE_READ.search(line)
            if not match:
                continue
            ids, ranges, size, local = match.groups()
            requests = ast.literal_eval(ids)
            if rid not in requests or int(local) != rank:
                continue
            if requests != [rid]:
                raise ValueError('Cannot attribute batched native READ bytes to this request')
            peer = re.search(r'\bremote_session_id (\S+) source_rank=(\d+)\b', line)
            if not peer:
                raise ValueError('Native READ lacks observed session and producer rank')
            session, source_rank = peer.groups()
        else:
            match = TRANSFER_READ.search(line)
            if not match:
                continue
            transfer, _, local, ranges, size = match.groups()
            if transfer != row['transfer_id'] or int(local) != rank:
                continue
            if f'request={rid} ' not in line:
                raise ValueError('State READ belongs to another engine request')
            source_rank = re.search(r'\bsource_rank=(\d+)\b', line)[1]
            session = None
        if int(ranges) < 1 or int(size) < 1 or int(source_rank) not in range(4):
            raise ValueError('Invalid READ ranges, bytes or source rank')
        receipts.append(dict(source_rank=int(source_rank), session=session,
                             ranges=int(ranges), bytes=int(size)))
    if not receipts:
        raise ValueError('Missing successful READ for this engine request and rank')
    return receipts


def check_api_series(row, expected):
    count = expected['output_count']
    for case in (row, expected):
        choice, = case['output']['choices']
        if choice['token_ids'] != case['token_ids'] or len(choice['token_ids']) != count:
            raise ValueError('Incomplete API output IDs')
        probs = choice['logprobs']
        for field in ('tokens', 'token_logprobs', 'top_logprobs'):
            if len(probs[field]) != count or any(value is None for value in probs[field]):
                raise ValueError('Incomplete API probabilities')
        if any(not candidates for candidates in probs['top_logprobs']):
            raise ValueError('Missing API probability candidates')
    if row['output']['choices'][0]['logprobs'] != expected['output']['choices'][0]['logprobs']:
        raise ValueError('Transport API probabilities differ from R')


def audit(bundle,producer_stage,consumer_stage,execution_path,reference_path,read_log):
    execution=json.loads(execution_path.read_text());reference=json.loads(reference_path.read_text())
    if execution.get('mode')!='pd' or execution.get('execution_passed') is not True:
        raise ValueError('Need completed P/D execution; HTTP success alone is insufficient')
    if (execution.get('producer_tp'),execution.get('consumer_tp'))!=(4,2):raise ValueError('Wrong transfer topology')
    check_hybrid_reference(reference,4,2,execution.get('runtime_identity'),reference_path.parent)
    check_reference_deployment(reference,execution.get('live_topology',{}))
    if not execution.get('rows'):raise ValueError('Empty P/D execution')
    raw=read_log.read_text()
    variant = execution['variant']
    if variant not in ('control', 'state'):
        raise ValueError('Unknown layout')
    native = variant == 'control'
    if not native:
        check_read_log(raw, execution['rows'], 2, producer_tp=4)
    results=[];artifacts=[]
    def observed(stage,key,phase,total,rank):
        path=bundle/stage/f'{key}-{phase}-t{total}-tp{rank}.pt'
        artifacts.append(dict(path=str(path.relative_to(bundle)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return load(path)
    for row in execution['rows']:
        key=row.get('observer_key')
        if not key or not key.isalnum():raise ValueError('Missing raw observer request binding')
        matches=[case for case in reference['rows'] if (case['dp'],case['prompt'],case['output_count'],case['repeat'])==
                 (row['dp'],row['prompt'],row['output_count'],0)]
        if len(matches)!=1:raise ValueError('Missing unique R comparison case')
        expected=matches[0];total=len(row['prompt'])-1
        check_api_series(row, expected)
        if row['token_ids']!=expected['token_ids']:raise ValueError('Transport output IDs differ from R')
        if row.get('decode_prompt_sources')!={'local_compute':1,'local_cache_hit':0,'external_kv_transfer':total}:
            raise ValueError('No evidence that D computed only the last input token')
        params=row['producer']['kv_transfer_params']
        source_request=params['remote_request_id']
        if execution.get('variant')!='control':
            wire=params['dsv41_state_snapshot']
            audit_wire_boundary(wire,total,total+1)
            if wire['source_request_id']!=source_request or wire['transfer_id']!=row['transfer_id']:
                raise ValueError('Wire transfer/request identity mismatch')
        ranks=[]
        expected_source=load(reference_path.parent/expected['source_files']['0'])
        sources = {}
        layout = 'native' if native else 'pools'
        for rank in range(4):
            source=observed(producer_stage,key,'source',total,rank)
            sources[rank] = source
            meta=source['metadata']
            if meta['layout'] != layout or meta['tp_rank'] != rank:
                raise ValueError('Unexpected producer layout or TP rank')
            if (meta['tp'],meta['source_rank'],meta['dp'])!=(4,rank,row.get('producer_dp',row['dp'])) or meta['runtime_identity']!=reference['runtime_identity'] or meta['prompt_sha256']!=prompt_digest(row['prompt']):
                raise ValueError('Wrong actual producer snapshot identity')
            if meta['request_id']!=source_request:raise ValueError('Source snapshot belongs to another request')
            result=compare(source,expected_source)
            if not result['passed']:raise ValueError('Actual P prefix differs from independently captured R source')
        for rank in (0,1):
            loaded=observed(consumer_stage,key,'loaded',total,rank)
            if loaded['metadata']['dp']!=row['dp']:raise ValueError('Wrong actual Decode DP observer')
            if row['request_id'] not in loaded['metadata']['request_id']:raise ValueError('Loaded snapshot belongs to another request')
            meta = loaded['metadata']
            if meta['tp_rank'] != rank or meta['layout'] != layout:
                raise ValueError('Unexpected consumer layout or TP rank')
            rid = meta['request_id']
            receipts = read_receipts(raw, row, rank, rid, native)
            source_ranks = sorted({item['source_rank'] for item in receipts})
            comparisons = [dict(source_rank=peer,
                                **audit_loaded(sources[peer], loaded, actual_source_rank=peer))
                           for peer in source_ranks]
            result = dict(passed=all(item['passed'] for item in comparisons),
                          sources=comparisons, reads=receipts)
            steps=[]
            for step in sample_steps(row['output_count']):
                actual=observed(consumer_stage,key,'sample',total+step,rank)
                sample_meta = actual['metadata']
                if (sample_meta['phase'], sample_meta['request_id'], sample_meta['dp'],
                    sample_meta['tp_rank'], sample_meta['layout']) != ('sample', rid, row['dp'], rank, layout):
                    raise ValueError('Continuation belongs to a different request, layout or rank')
                # Use the real stage recorded by the R artifact, rather than
                # inferring a native/pools path when the bundle was renamed.
                artifacts_for_step=[item for item in expected['reference_audit']['repeat_audit']['artifacts']
                    if item['rank']==rank and item['step']==step and Path(item['path']).name.startswith(expected['key']+'-')]
                if len(artifacts_for_step)!=1:raise ValueError('Missing R continuation artifact binding')
                control=load(reference_path.parent/artifacts_for_step[0]['path'])
                steps.append(dict(step=step,**audit_samples(control,actual)))
            ranks.append(dict(rank=rank,loaded=result,continuation=steps,
                              passed=result['passed'] and all(item['passed'] for item in steps)))
        results.append(dict(dp=row['dp'],prompt_tokens=total+1,key=key,ranks=ranks,api_probabilities_equal=True,passed=all(item['passed'] for item in ranks)))
    return dict(passed=False,correctness_passed=all(row['passed'] for row in results),results=results,artifacts=artifacts,
                pending_gates=['lifecycle','uncovered_DP_and_configuration_cases'],scope='Only the observed cases and frozen R topology')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path);parser.add_argument('--producer-stage',required=True);parser.add_argument('--consumer-stage',required=True)
    parser.add_argument('--execution',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--read-log',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Do not overwrite an earlier audit')
    result=audit(args.bundle,args.producer_stage,args.consumer_stage,args.execution,args.reference,args.read_log)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    if not result['correctness_passed']:raise SystemExit(1)


if __name__=='__main__':main()
