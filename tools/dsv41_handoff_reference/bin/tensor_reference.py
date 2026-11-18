"""Torch adapter for independent row extraction and scatter, CPU/NPU alike."""
from dataclasses import asdict
import hashlib

from reference_contract import row_locations


def digest(value):
    import torch
    host=value.detach().cpu().contiguous()
    if host.is_floating_point() and not bool(torch.isfinite(host).all()):
        raise ValueError("Nonfinite valid state")
    return hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()


def extract(plane, tensor, ids, total):
    import torch
    locations=row_locations(plane,ids,total)
    if not locations:
        return torch.empty((0,*plane.row_shape),dtype=tensor.dtype)
    blocks=torch.tensor([p[0] for p in locations],device=tensor.device,dtype=torch.long)
    rows=torch.tensor([p[1] for p in locations],device=tensor.device,dtype=torch.long)
    result=tensor[blocks,rows].detach().cpu().contiguous()
    digest(result)
    return result


def capture(descriptors, ids, total):
    values={};geometry={}
    for plane,tensor,group in descriptors:
        if plane.key in values:raise ValueError("Duplicate plane")
        values[plane.key]=extract(plane,tensor,ids[group],total)
        geometry[plane.key]=asdict(plane)
    if not values:raise ValueError("No cache planes")
    return dict(planes=values,geometry=geometry,
                hashes={name:digest(value) for name,value in values.items()})


def restore(snapshot, descriptors, ids, total):
    """Validate every plane before writing any; only scatter valid rows."""
    import torch
    expected={p.key for p,_,_ in descriptors}
    if set(snapshot['planes'])!=expected or set(snapshot['geometry'])!=expected:
        raise ValueError("Incomplete snapshot plane set")
    prepared=[]
    for plane,tensor,group in descriptors:
        locations=row_locations(plane,ids[group],total)
        value=snapshot['planes'][plane.key]
        source=snapshot['geometry'][plane.key]
        if source['kind']!=plane.kind or source['compression']!=plane.compression:
            raise ValueError("Snapshot logical format mismatch")
        if tuple(value.shape)!=(len(locations),*plane.row_shape) or value.dtype!=tensor.dtype:
            raise ValueError("Snapshot row shape/dtype mismatch")
        if digest(value)!=snapshot['hashes'][plane.key]:
            raise ValueError("Snapshot data digest mismatch")
        prepared.append((tensor,locations,value))
    for tensor,locations,value in prepared:
        if locations:
            blocks=torch.tensor([p[0] for p in locations],device=tensor.device,dtype=torch.long)
            rows=torch.tensor([p[1] for p in locations],device=tensor.device,dtype=torch.long)
            tensor[blocks,rows]=value.to(tensor.device)


def compare(left,right):
    if set(left['planes'])!=set(right['planes']):
        raise ValueError("State audit plane set mismatch")
    mismatches=[]
    for key,x in left['planes'].items():
        y=right['planes'][key]
        if x.shape!=y.shape or x.dtype!=y.dtype or digest(x)!=digest(y):mismatches.append(key)
    return dict(passed=not mismatches,planes=len(left['planes']),mismatches=mismatches,
                scope='All enumerated valid logical rows, original dtype bytes; ignores padding and local IDs')
