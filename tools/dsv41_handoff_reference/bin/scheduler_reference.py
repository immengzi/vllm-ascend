"""Explicitly armed test-only bootstrap; no change to normal requests."""
import json
import os
from pathlib import Path


def apply_bootstrap(scheduler, output, spec):
    if spec.get('action')!='replay':return False
    matches=[r for r in output.scheduled_new_reqs if spec['key'] in r.req_id]
    if not matches:return False
    if len(matches)!=1 or len(output.num_scheduled_tokens)!=1:
        raise ValueError('Independent reference requires one actual scheduled request')
    request=matches[0];tokens=spec['original_prompt'];total=len(tokens)-1
    if total<1 or request.prompt_token_ids!=tokens or request.prefill_token_ids!=tokens:
        raise ValueError('Reference prompt/bootstrap mismatch')
    if request.num_computed_tokens!=0:
        raise ValueError('Reference bootstrap must begin from a fresh request')
    count=output.num_scheduled_tokens[request.req_id]
    if count!=len(tokens):
        raise ValueError('Reference bootstrap requires initial prompt within one scheduling budget')
    owner=scheduler.requests[request.req_id]
    # schedule() has already advanced CPU computed/in-flight counts by N.
    # Keeping computed=N is correct for executing the last prompt token;
    # only one token is actually in flight after this fixture replaces N.
    if owner.num_computed_tokens!=len(tokens) or owner.num_in_flight_tokens<count:
        raise ValueError('Unexpected scheduler advancement semantics')
    request.num_computed_tokens=total
    owner.num_in_flight_tokens-=total
    output.num_scheduled_tokens[request.req_id]=1
    output.total_num_scheduled_tokens-=total
    return True


def install(cls,root=Path('/results')):
    original=cls.schedule
    def schedule(self,*args,**kwargs):
        output=original(self,*args,**kwargs)
        try:spec=json.loads((root/'reference-control.json').read_text())
        except FileNotFoundError:return output
        changed=apply_bootstrap(self,output,spec)
        selected={rid:n for rid,n in output.num_scheduled_tokens.items() if spec['key'] in rid}
        if selected:
            path=root/spec['folder'];path.mkdir(exist_ok=True)
            with (path/f'schedule-{os.getpid()}.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(key=spec['key'],action=spec['action'],bootstrap=changed,
                    counts=selected,new=[dict(id=r.req_id,before=r.num_computed_tokens) for r in output.scheduled_new_reqs if r.req_id in selected]))+'\n')
        return output
    cls.schedule=schedule
