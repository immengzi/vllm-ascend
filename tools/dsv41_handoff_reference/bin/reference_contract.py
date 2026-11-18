"""Independent logical row addressing for validation, not a transport planner.

Only standard-library dependencies. Never import the implementation's copy
planner, transfer address helper, or serialized state-counter constructor.
"""
from dataclasses import dataclass
import hashlib
import json
import math

DTYPES = {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1, "uint8": 1}


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"Invalid {name}")
    return value


def prompt_digest(tokens):
    if not tokens or any(type(x) is not int or x < 0 for x in tokens):
        raise ValueError("Invalid original prompt tokens")
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def boundary_counters(total, original_length):
    positive(total, "boundary")
    if original_length != total + 1:
        raise ValueError("Reference requires T=N-1")
    return dict(original_prompt_tokens=original_length,**progress_counters(total))


def progress_counters(total):
    positive(total,'processed tokens')
    return dict(processed_tokens=total, absolute_next_position=total, window_valid=min(total,128), window_next=total%128,
                compressor_valid=min(total,32), compressor_next=total%32,
                compressed_completed=total//2, compressed_residual=total%2)


@dataclass(frozen=True)
class Plane:
    name: str
    part: int
    kind: str
    layout: str
    capacity: int
    rows: int
    logical_block: int
    compression: int
    dtype: str
    row_shape: tuple

    @property
    def key(self):
        return f"{self.name}:{self.part}"

    @property
    def row_bytes(self):
        return math.prod(self.row_shape)*DTYPES[self.dtype]

    def validate(self):
        if not self.name or type(self.part) is not int or self.part < 0:
            raise ValueError("Invalid plane identity")
        if self.kind not in ("history","window","compressor") or self.layout not in ("native","pools"):
            raise ValueError("Invalid plane classification")
        for name in ("capacity","rows","logical_block","compression"):
            positive(getattr(self,name),name)
        if self.dtype not in DTYPES or not self.row_shape or any(type(n) is not int or n<1 for n in self.row_shape):
            raise ValueError("Invalid plane row format")
        if self.kind=="history" and self.rows*self.compression!=self.logical_block:
            raise ValueError("History token/page geometry mismatch")
        if self.kind=="window" and (self.rows!=128 or self.logical_block!=128):
            raise ValueError("Unsupported SWA geometry")
        if self.kind=="compressor" and self.rows!=32:
            raise ValueError("Unsupported compressor ring")


def row_locations(plane, ids, total):
    """Return physical (allocation, row) for logical, chronological valid rows."""
    plane.validate();positive(total,"processed tokens")
    if any(type(i) is not int or i<0 or i>=plane.capacity for i in ids):
        raise ValueError("Allocation ID out of bounds")
    if plane.kind=="history":
        count=total//plane.compression
        logical=range(count)
        positions=[(row//plane.rows,row%plane.rows) for row in logical]
    elif plane.kind=="window" and plane.layout=="native":
        positions=[(p//128,p%128) for p in range(max(0,total-128),total)]
    else:
        if len(ids)!=1:
            raise ValueError("Request ring requires exactly one allocation")
        positions=[(0,p%plane.rows) for p in range(max(0,total-plane.rows),total)]
    result=[]
    for logical_id,row in positions:
        if logical_id>=len(ids) or ids[logical_id]==0:
            raise ValueError("Missing valid allocation or null allocation")
        result.append((ids[logical_id],row))
    active={logical_id for logical_id,_ in positions}
    allocations=[ids[i] for i in sorted(active)]
    if len(set(allocations))!=len(allocations):
        raise ValueError("Aliased valid allocations")
    if len(set(result))!=len(result):
        raise ValueError("Overlapping logical rows")
    return result


def audit_wire_boundary(wire, total, original_length):
    expected=boundary_counters(total,original_length)
    mapping={"boundary":"processed_tokens","original_prompt_tokens":"original_prompt_tokens",
             "window_valid":"window_valid","window_next":"window_next",
             "compressor_valid":"compressor_valid","compressor_next":"compressor_next",
             "compressed_completed":"compressed_completed","compressed_residual":"compressed_residual"}
    for source,target in mapping.items():
        if type(wire.get(source)) is not int or wire[source]!=expected[target]:
            raise ValueError(f"Wire boundary mismatch: {source}")
    return expected


def check_snapshot_metadata(meta, original_prompt, runtime_identity, *, boundary=None):
    total=len(original_prompt)-1 if boundary is None else boundary
    expected=boundary_counters(total,len(original_prompt))
    if meta.get("schema")!=1 or meta.get("runtime_identity")!=runtime_identity:
        raise ValueError("Snapshot schema or runtime identity mismatch")
    if meta.get("prompt_sha256")!=prompt_digest(original_prompt):
        raise ValueError("Snapshot prompt mismatch")
    if meta.get("counters")!=expected:
        raise ValueError("Snapshot counters mismatch")
    if meta.get("producer_tp")!=4 or meta.get('tp')!=4 or meta.get("source_rank") not in range(4):
        raise ValueError("Unsupported producer identity")


def check_hybrid_reference(document, producer_tp, consumer_tp, runtime_identity, base_dir=None):
    if not document.get("passed") or document.get("mode")!="reference":
        raise ValueError("Reference is incomplete")
    if producer_tp!=consumer_tp:
        if (producer_tp,consumer_tp)!=(4,2):raise ValueError('Unsupported heterogeneous TP reference')
        if document.get("reference_kind")!="hybrid_snapshot_replay":
            raise ValueError("Heterogeneous TP requires hybrid reference; full D Prefill is diagnostic only")
        if document.get("producer_tp")!=producer_tp or document.get("consumer_tp")!=consumer_tp:
            raise ValueError("Reference topology mismatch")
        if not runtime_identity or document.get("runtime_identity")!=runtime_identity:
            raise ValueError("Reference runtime mismatch")
        for row in document.get("rows",[]):
            receipt=row.get("reference_audit",{})
            if receipt.get("passed") is not True or receipt.get("transport")!="local_snapshot_replay":
                raise ValueError("Missing independent replay audit")
            repeat=receipt.get('repeat_audit',{})
            if repeat.get('passed') is not True or not repeat.get('samples'):
                raise ValueError('Missing complete raw R repeat audit')
            count=positive(row.get('output_count'),'reference output count')
            expected={(rank,step) for rank in (0,1) for step in (1,2,32,128,130) if step<=count}
            samples=repeat['samples']
            if len(samples)!=len(expected) or {(s.get('rank'),s.get('step')) for s in samples}!=expected or any(s.get('passed') is not True for s in samples):
                raise ValueError('Incomplete raw R repeat sample coverage')
            for name in ("source_sha256","loaded_sha256","first_logits_sha256"):
                value=receipt.get(name)
                if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError("Missing reference artifact digest")
            if base_dir is not None:
                from pathlib import Path
                base=Path(base_dir).resolve()
                artifacts=receipt.get('artifacts',[])
                coverage={phase:{a.get('rank') for a in artifacts if a.get('phase')==phase} for phase in ('source','loaded','sample')}
                if coverage!={'source':{0,2},'loaded':{0,1},'sample':{0,1}}:
                    raise ValueError('Missing reference raw artifacts')
                from collections import Counter
                if Counter((a.get('rank'),a.get('step')) for a in repeat.get('artifacts',[]))!={key:2 for key in expected}:
                    raise ValueError('Missing raw artifacts from both R repeats')
                for item in artifacts+repeat.get('artifacts',[]):
                    path=(base/item['path']).resolve()
                    if not path.is_relative_to(base) or not path.is_file():
                        raise ValueError('Reference artifact outside bundle or missing')
                    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                    if actual!=item['sha256']:raise ValueError('Reference artifact checksum mismatch')
        if not document.get("rows"):
            raise ValueError("Empty hybrid reference")
    return document
