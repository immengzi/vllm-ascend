"""Independent CPU audit for source/loaded/continuation snapshots."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from reference_contract import boundary_counters,prompt_digest
from tensor_reference import compare,digest


def sample_steps(count):
    return [step for step in (1,2,32,128,130) if step<=count]


def compare_case_samples(root,left_stage,left,right_stage,right):
    if left['prompt']!=right['prompt'] or left['output_count']!=right['output_count']:
        raise ValueError('Continuation case mismatch')
    boundary=len(left['prompt'])-1
    results=[];artifacts=[]
    for step in sample_steps(left['output_count']):
        for rank in (0,1):
            pair=[]
            for stage,case in ((left_stage,left),(right_stage,right)):
                path=root/stage/f'{case["key"]}-sample-t{boundary+step}-tp{rank}.pt'
                pair.append(load(path))
                artifacts.append(dict(phase='continuation',rank=rank,step=step,
                    path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            results.append(dict(rank=rank,step=step,**audit_samples(*pair)))
    ids_equal=left['token_ids']==right['token_ids']
    probabilities_equal=left['output']['choices'][0]['logprobs']==right['output']['choices'][0]['logprobs']
    return dict(passed=ids_equal and probabilities_equal and all(r['passed'] for r in results),
                ids_equal=ids_equal,api_probabilities_equal=probabilities_equal,samples=results,artifacts=artifacts)


def audit_repeats(root,stage,rows):
    groups={}
    for row in rows:
        key=(row['dp'],tuple(row['prompt']),row['output_count'])
        groups.setdefault(key,[]).append(row)
    for values in groups.values():
        if len(values)!=2 or {v['repeat'] for v in values}!={0,1}:
            raise ValueError('Reference requires exactly two independent R repeats per case')
        left,right=sorted(values,key=lambda v:v['repeat'])
        result=compare_case_samples(root,stage,left,stage,right)
        for row in values:
            row['reference_audit']['repeat_audit']=result
            row['reference_audit']['passed'] &= result['passed']
    return groups


def load(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def audit_loaded(source,loaded, *, actual_source_rank=None):
    a=source['metadata'];b=loaded['metadata']
    if a['phase']!='source' or b['phase']!='loaded':raise ValueError('Wrong snapshot phase')
    for field in ('runtime_identity','prompt_sha256','counters'):
        if a[field]!=b[field]:raise ValueError(f'Loaded metadata mismatch: {field}')
    if a['producer_tp']!=4 or a['tp']!=4 or b['tp']!=2:raise ValueError('Wrong replay topology')
    expected_source = 2*b['tp_rank'] if actual_source_rank is None else actual_source_rank
    if a['source_rank']!=expected_source:raise ValueError('Wrong producer replica for D rank')
    return compare(source,loaded)


def audit_samples(left,right):
    a=left['metadata'];b=right['metadata']
    for key in ('runtime_identity','prompt_sha256','counters','tp_rank','tp','original_boundary'):
        if a[key]!=b[key]:raise ValueError(f'Continuation metadata mismatch: {key}')
    result=compare(left,right)
    for key in ('logits','hidden','engram_hash','engram_lookback'):
        if key not in left or key not in right:raise ValueError(f'Missing continuation evidence: {key}')
        x,y=left[key],right[key]
        equal=x.shape==y.shape and x.dtype==y.dtype and digest(x)==digest(y)
        result[key+'_equal']=equal;result['passed'] &= equal
    x=left.get('engram_lookup',{});y=right.get('engram_lookup',{})
    if not x or set(x)!=set(y):raise ValueError('Incomplete actual Engram lookup set')
    result['engram_lookup_equal']=all(digest(x[k])==digest(y[k]) for k in x)
    result['passed'] &= result['engram_lookup_equal']
    return result


def audit_case(root,stage,case):
    folder=root/stage
    key=case['key'];boundary=len(case['prompt'])-1
    results=[];artifacts=[]
    for rank in (0,1):
        source_path=root/case['source_files'][str(rank*2)]
        loaded_path=folder/f'{key}-loaded-t{boundary}-tp{rank}.pt'
        sample_path=folder/f'{key}-sample-t{boundary+1}-tp{rank}.pt'
        source=load(source_path);loaded=load(loaded_path);sample=load(sample_path)
        result=audit_loaded(source,loaded)
        if sample['metadata']['counters']['processed_tokens']!=boundary+1:
            raise ValueError('First sample boundary mismatch')
        if sample['logits'].argmax(-1).tolist()!=case['token_ids'][:1]:
            raise ValueError('Raw first logits do not match API first output')
        if sample['metadata']['prompt_sha256']!=prompt_digest(case['prompt']):
            raise ValueError('Continuation prompt binding mismatch')
        results.append(dict(rank=rank,**result))
        for phase,path,identity in (('source',source_path,rank*2),('loaded',loaded_path,rank),('sample',sample_path,rank)):
            artifacts.append(dict(phase=phase,rank=identity,path=str(path.relative_to(root)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    first=load(folder/f'{key}-sample-t{boundary+1}-tp0.pt')
    named={phase:next(a for a in artifacts if a['phase']==phase and a['rank']==0) for phase in ('source','loaded')}
    return dict(passed=all(r['passed'] for r in results),transport='local_snapshot_replay',rank_audits=results,
                source_sha256=named['source']['sha256'],loaded_sha256=named['loaded']['sha256'],
                first_logits_sha256=first['logits_sha256'],artifacts=artifacts)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('stage')
    parser.add_argument('--compare-stage');args=parser.parse_args()
    document=json.loads((args.root/args.stage/'responses.json').read_text())
    identity=json.loads((args.root/'identity.json').read_text())
    if identity['runtime_identity']!=document['runtime_identity']:raise ValueError('Reference deployment identity changed')
    document['runtime_context']=identity
    for case in document['rows']:
        case['reference_audit']=audit_case(args.root,args.stage,case)
    audit_repeats(args.root,args.stage,document['rows'])
    document.update(mode='reference',reference_kind='hybrid_snapshot_replay',producer_tp=4,consumer_tp=2,
                    passed=all(c['reference_audit']['passed'] for c in document['rows']))
    target=args.root/f'{args.stage}-reference.json'
    target.write_text(json.dumps(document,indent=2)+'\n')
    all_passed=document['passed']
    if args.compare_stage:
        other=json.loads((args.root/args.compare_stage/'responses.json').read_text())
        identify=lambda row:(row['dp'],tuple(row['prompt']),row['output_count'],row['repeat'])
        left={identify(row):row for row in document['rows']}
        right={identify(row):row for row in other['rows']}
        if set(left)!=set(right):raise ValueError('Native/pools comparison cases differ')
        results=[compare_case_samples(args.root,args.stage,left[key],args.compare_stage,right[key]) for key in left]
        comparison=dict(passed=all(r['passed'] for r in results),left=args.stage,right=args.compare_stage,results=results,
                        scope='Same hybrid P4-prefix / D2 continuation, not SDK READ')
        (args.root/'r-layout-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
        all_passed &= comparison['passed']
    print(json.dumps(dict(stage=args.stage,passed=document['passed'],cases=len(document['rows']),file=str(target))))
    if not all_passed:raise SystemExit(1)


if __name__=='__main__':main()
