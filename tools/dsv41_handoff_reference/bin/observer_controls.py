"""Arm disposable model-thread observers; never launch or restart a service."""
import json
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path


REMOTE = '''import json,sys
from pathlib import Path
payload=json.load(sys.stdin)
root=Path(payload['root']);path=root/'reference-control.json'
if not root.is_absolute() or not root.is_dir():raise ValueError('Missing observer root')
spec=payload['spec'];key=spec['key']
if not key.isalnum() or Path(spec['folder']).name!=spec['folder']:raise ValueError('Unsafe observer identity')
if payload['operation']=='arm':
    if path.exists():raise ValueError('Observer already armed; do not overwrite another request')
    (root/spec['folder']).mkdir(exist_ok=True)
    temp=path.with_suffix('.'+key+'.tmp')
    temp.write_text(json.dumps(spec)+'\\n');temp.replace(path)
elif payload['operation']=='clear':
    if path.exists():
        if json.loads(path.read_text()).get('key')!=key:raise ValueError('Observer ownership changed')
        path.unlink()
else:raise ValueError('Unknown observer operation')
'''


def update(endpoint,operation,spec):
    payload=json.dumps(dict(root=endpoint['root'],operation=operation,spec=spec))
    if 'ssh_host' in endpoint:
        config=Path(endpoint['ssh_config']).resolve()
        if not config.is_file():raise ValueError('Missing explicit SSH configuration')
        command=['ssh','-F',str(config),'-o','BatchMode=yes',endpoint['ssh_host'],shlex.join(['python3','-c',REMOTE])]
    else:
        import sys
        command=[sys.executable,'-c',REMOTE]
    subprocess.run(command,input=payload,text=True,check=True,timeout=20,capture_output=True)


@contextmanager
def armed(config,key,prompt,runtime_identity):
    endpoints=json.loads(Path(config).read_text());owned=[]
    try:
        for role,action in (('producer','observe_source'),('consumer','observe')):
            endpoint=endpoints[role]
            spec=dict(action=action,key=key,folder=endpoint['folder'],original_prompt=prompt,runtime_identity=runtime_identity)
            update(endpoint,'arm',spec);owned.append((endpoint,spec))
        yield
    finally:
        errors=[]
        for endpoint,spec in reversed(owned):
            try:update(endpoint,'clear',spec)
            except Exception as exc:errors.append(repr(exc))
        if errors:raise RuntimeError('Observer disarm failed: '+ '; '.join(errors))
