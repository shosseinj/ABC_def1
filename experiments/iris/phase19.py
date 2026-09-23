"""Preregistered Phase-19 frozen-representation head diagnostic utilities."""
import hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from contextlib import contextmanager

HEADS=("HEAD_A_CURRENT","HEAD_B_LINEAR_REINIT","HEAD_C_NORMALIZED_LINEAR","HEAD_D_COSINE","HEAD_E_COSINE_MARGIN","HEAD_F_LINEAR_MARGIN")

def derived_seed(split_seed,model_seed,head):
    return int.from_bytes(hashlib.sha256(f"phase19:{split_seed}:{model_seed}:{head}".encode()).digest()[:4],"big")

class DiagnosticHead(nn.Module):
    def __init__(self,name,current=None,initial_weight=None):
        super().__init__(); self.name=name
        self.weight=nn.Parameter(torch.empty(3,4)); self.bias=None if name in HEADS[3:5] else nn.Parameter(torch.zeros(3))
        if initial_weight is not None: self.weight.data.copy_(initial_weight)
        elif current is None: nn.init.xavier_uniform_(self.weight)
        else:
            self.weight.data.copy_(current.weight.data)
            self.bias.data.copy_(current.bias.data)
    def forward(self,x,labels=None,training_loss=False):
        if self.name in HEADS[2:5]: x=F.normalize(x,dim=1)
        w=F.normalize(self.weight,dim=1) if self.name in HEADS[3:5] else self.weight
        cosine=F.linear(x,w,self.bias)
        if training_loss and self.name=="HEAD_E_COSINE_MARGIN":
            cosine=cosine-F.one_hot(labels,3)*.1
        z=cosine*(10.0 if self.name in HEADS[3:5] else 1.0)
        return z

def margins(logits,y):
    other=logits.clone(); other[torch.arange(len(y)),y]=-torch.inf
    return logits[torch.arange(len(y)),y]-other.max(1).values

def shared_init_seed(split_seed,model_seed):
    return int.from_bytes(hashlib.sha256(f"phase19-shared-init:{split_seed}:{model_seed}".encode()).digest()[:4],"big")

def shared_xavier_weight(seed):
    g=torch.Generator().manual_seed(seed); w=torch.empty(3,4); nn.init.xavier_uniform_(w,generator=g); return w

def tensor_hash(tensor): return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def train_head(name,train_x,train_y,val_x,val_y,current,seed,initial_weight=None):
    torch.manual_seed(seed); h=DiagnosticHead(name,current if name==HEADS[0] else None,initial_weight)
    opt=torch.optim.Adam(h.parameters(),lr=.01); best=None; key=None; epoch0=None
    for epoch in range(1,201):
        h.train(); opt.zero_grad(); z=h(train_x,train_y,True); loss=F.cross_entropy(z,train_y)
        if name=="HEAD_F_LINEAR_MARGIN": loss=loss+.5*F.relu(.1-margins(z,train_y)).mean()
        loss.backward(); opt.step(); h.eval()
        with torch.no_grad():
            vz=h(val_x); ce=float(F.cross_entropy(vz,val_y)); acc=float((vz.argmax(1)==val_y).float().mean())
        candidate=(-acc,ce,epoch)
        if key is None or candidate<key: key=candidate; epoch0=epoch; best={k:v.detach().clone() for k,v in h.state_dict().items()}
    h.load_state_dict(best); h.eval(); return h,{"best_epoch":epoch0,"validation_accuracy":-key[0],"validation_ce":key[1]}

def paired_states(a_fail,b_fail):
    out={"rescued":0,"broken":0,"both_fail":0,"both_robust":0}
    for a,b in zip(a_fail,b_fail): out["both_fail" if a and b else "rescued" if a else "broken" if b else "both_robust"]+=1
    return out
def common_clean_pair(current,candidate):
    a={r["sample_id"]:r for r in current if r["clean_correct"]}; b={r["sample_id"]:r for r in candidate if r["clean_correct"]}
    ids=sorted(set(a)&set(b)); states=paired_states([a[i]["attacked_fail"] for i in ids],[b[i]["attacked_fail"] for i in ids])
    af=sum(a[i]["attacked_fail"] for i in ids); bf=sum(b[i]["attacked_fail"] for i in ids); n=len(ids)
    return {"N_common":n,"current_failures":af,"candidate_failures":bf,**states,"net_gain":af-bf,"current_paired_asr":af/n if n else None,"candidate_paired_asr":bf/n if n else None,"common_ids":ids}

def module_hash(module):
    """Stable full state (parameters and persistent buffers) hash."""
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

@contextmanager
def no_test_guard():
    """Fail closed on both public and training-module held-out loader paths."""
    import experiments.iris.data as data
    import experiments.iris.training as training
    calls={"prohibited_loader_calls":0,"evaluate_test_true_calls":0}
    old_data,old_training,old_train=data.load_iris_splits,training.load_iris_splits,training.train_iris_model
    def prohibited(*args,**kwargs): calls["prohibited_loader_calls"]+=1; raise RuntimeError("Phase19 prohibits held-out loading")
    def guarded_train(config,*args,**kwargs):
        if kwargs.get("evaluate_test",True): calls["evaluate_test_true_calls"]+=1; raise RuntimeError("Phase19 requires evaluate_test=False")
        return old_train(config,*args,**kwargs)
    data.load_iris_splits=prohibited; training.load_iris_splits=prohibited; training.train_iris_model=guarded_train
    try: yield calls,guarded_train
    finally: data.load_iris_splits=old_data; training.load_iris_splits=old_training; training.train_iris_model=old_train
