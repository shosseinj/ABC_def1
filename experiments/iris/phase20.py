"""Phase-20 preregistered cosine-margin calibration utilities."""
import hashlib
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F

HEAD_A_CURRENT="HEAD_A_CURRENT"
HEAD_E_COSINE_MARGIN="HEAD_E_COSINE_MARGIN"
SCALES=(5,10,15); MARGINS=(.05,.10,.15)

class CalibrationHead(nn.Module):
    def __init__(self,name,scale=10,margin=.1,current=None,initial_weight=None):
        super().__init__(); self.name=name; self.scale=float(scale); self.margin=float(margin)
        if name==HEAD_A_CURRENT:
            self.weight=nn.Parameter(current.weight.detach().clone()); self.bias=nn.Parameter(current.bias.detach().clone())
        else:
            self.weight=nn.Parameter(initial_weight.detach().clone()); self.register_parameter("bias",None)
    def forward(self,x,labels=None,training_loss=False):
        if self.name==HEAD_A_CURRENT: return F.linear(x,self.weight,self.bias)
        cos=F.linear(F.normalize(x,dim=1),F.normalize(self.weight,dim=1))
        if training_loss: cos=cos-F.one_hot(labels,3).to(cos)*self.margin
        return self.scale*cos

def margins(logits,y):
    other=logits.clone(); other[torch.arange(len(y)),y]=-torch.inf
    return logits[torch.arange(len(y)),y]-other.max(1).values

def derived_seed(split_seed,model_seed):
    return int.from_bytes(hashlib.sha256(f"phase20-shared-init:{split_seed}:{model_seed}".encode()).digest()[:4],"big")

def shared_xavier(seed):
    g=torch.Generator().manual_seed(seed); w=torch.empty(3,4); nn.init.xavier_uniform_(w,generator=g); return w

def tensor_hash(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def module_hash(module):
    h=hashlib.sha256()
    for n,v in sorted(module.state_dict().items()):
        a=v.detach().cpu().contiguous().numpy(); h.update(n.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def fit_head(head,train_x,train_y,val_x,val_y):
    opt=torch.optim.Adam(head.parameters(),lr=.01); best=None; key=None; best_epoch=None
    for epoch in range(1,201):
        head.train();opt.zero_grad();loss=F.cross_entropy(head(train_x,train_y,True),train_y);loss.backward();opt.step()
        head.eval()
        with torch.no_grad():
            z=head(val_x); acc=float((z.argmax(1)==val_y).float().mean());ce=float(F.cross_entropy(z,val_y))
        k=(-acc,ce,epoch)
        if key is None or k<key: key=k;best_epoch=epoch;best={n:v.detach().clone() for n,v in head.state_dict().items()}
    head.load_state_dict(best);head.eval()
    return {"best_epoch":best_epoch,"validation_accuracy":-key[0],"validation_ce":key[1]}

def common_clean_pair(a,b):
    aa={r["sample_id"]:r for r in a if r["clean_correct"]};bb={r["sample_id"]:r for r in b if r["clean_correct"]};ids=sorted(set(aa)&set(bb))
    out={"rescued":0,"broken":0,"both_fail":0,"both_robust":0}
    for i in ids:
        x,y=aa[i]["attacked_fail"],bb[i]["attacked_fail"]
        out["both_fail" if x and y else "rescued" if x else "broken" if y else "both_robust"]+=1
    af=sum(aa[i]["attacked_fail"] for i in ids);bf=sum(bb[i]["attacked_fail"] for i in ids);n=len(ids)
    return {"N_common":n,"current_failures":af,"candidate_failures":bf,**out,"net_gain":af-bf,"current_paired_asr":af/n if n else None,"candidate_paired_asr":bf/n if n else None,"common_ids":ids}

def select_config(split_summaries):
    """Aggregate-metric-only selection; sample IDs/diagnostics are not inputs."""
    eligible=[r for r in split_summaries if r["eligible"]]
    ranked=sorted(eligible,key=lambda r:(-r["mean_accuracy_delta"],-r["minimum_class_delta_nonnegative_splits"],-r["class12_margin_delta_mean"],r["scale"],r["margin"]))
    return ranked[0] if ranked else None

@contextmanager
def no_test_guard():
    import experiments.iris.data as data, experiments.iris.training as training
    oldd,oldt,oldtrain=data.load_iris_splits,training.load_iris_splits,training.train_iris_model
    audit={"prohibited_loader_calls":0,"evaluate_test_true_calls":0}
    def deny(*a,**k): audit["prohibited_loader_calls"]+=1;raise RuntimeError("Phase20 prohibits held-out loading")
    def train(config,*a,**k):
        if k.get("evaluate_test",True): audit["evaluate_test_true_calls"]+=1;raise RuntimeError("Phase20 requires evaluate_test=False")
        return oldtrain(config,*a,**k)
    data.load_iris_splits=deny;training.load_iris_splits=deny;training.train_iris_model=train
    try: yield audit,train
    finally: data.load_iris_splits=oldd;training.load_iris_splits=oldt;training.train_iris_model=oldtrain
