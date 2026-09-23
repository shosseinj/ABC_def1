"""Train true Binary-grid CIFAR10-DVS SNN checkpoints from scratch."""
from __future__ import annotations
import csv, hashlib, json, os, random, statistics, sys, time
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from models.cifar10_dvs_snn import CIFAR10DVSConvSNN
PYTHON=Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe");SEEDS=(42,123,777)
OUT=ROOT/'Reports/results/cifar10_dvs_binary_true';CKPT=ROOT/'checkpoints/cifar10_dvs_binary_true';LOG=ROOT/'Reports/logs/cifar10_dvs_binary_true';CACHE=ROOT/'Reports/checkpoints/cifar10_dvs_binary_true'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def binary_frames(events,bins=10,size=128):
 out=np.zeros((bins,2,size,size),dtype=np.uint8)
 if not len(events):return out
 x=np.asarray(events['x'],dtype=np.int64);y=np.asarray(events['y'],dtype=np.int64);t=np.asarray(events['t'],dtype=np.int64);p=np.asarray(events['p'],dtype=np.int64)
 if np.any(np.diff(t)<0):
  order=np.argsort(t,kind='stable');x,y,t,p=x[order],y[order],t[order],p[order]
 duration=max(int(t[-1])-int(t[0])+1,1);tb=np.minimum(((t-t[0])*bins)//duration,bins-1);valid=(x>=0)&(x<size)&(y>=0)&(y<size)&((p==0)|(p==1));out[tb[valid],p[valid],y[valid],x[valid]]=1;return out
def build_cache(native):
 CACHE.mkdir(parents=True,exist_ok=True);fp=CACHE/'frames_t10_binary_uint8.npy';lp=CACHE/'labels.npy';mp=CACHE/'complete.json';shape=(len(native),10,2,128,128)
 if mp.exists() and fp.exists() and lp.exists():
  m=json.loads(mp.read_text());x=np.load(fp,mmap_mode='r');y=np.load(lp,mmap_mode='r')
  if tuple(x.shape)==shape and m.get('frames_sha256')==sha(fp) and m.get('labels_sha256')==sha(lp):return x,y,fp
 x=np.lib.format.open_memmap(fp,mode='w+',dtype=np.uint8,shape=shape);y=np.empty(len(native),dtype=np.int64)
 for i in range(len(native)):
  e,l=native[i];x[i]=binary_frames(e);y[i]=l
  if (i+1)%100==0:print(f'CIFAR10-DVS Binary preprocessing {i+1}/{len(native)}',flush=True)
 x.flush();np.save(lp,y);atomic(mp,{'shape':shape,'dtype':'uint8','representation':'binary_occupancy','normalization':'none','frames_sha256':sha(fp),'labels_sha256':sha(lp)});return np.load(fp,mmap_mode='r'),np.load(lp,mmap_mode='r'),fp
class MemmapSet(Dataset):
 def __init__(self,x,y,ids):self.x,self.y,self.ids=x,y,np.asarray(ids,dtype=np.int64)
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  j=int(self.ids[i]);return torch.from_numpy(np.array(self.x[j],dtype=np.float32,copy=True)),int(self.y[j])
def loader(x,y,ids,batch,shuffle,seed):return DataLoader(MemmapSet(x,y,ids),batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed))
@torch.no_grad()
def evaluate(model,data,device):
 model.eval();ys=[];ps=[];loss=0.
 for x,y in data:
  x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True);z=model(x);loss+=float(F.cross_entropy(z,y,reduction='sum'));ys+=y.cpu().tolist();ps+=z.argmax(1).cpu().tolist()
 ys=np.asarray(ys);ps=np.asarray(ps);cm=confusion_matrix(ys,ps,labels=range(10));tot=cm.sum(1)
 return {'samples':len(ys),'correct':int((ys==ps).sum()),'accuracy':float((ys==ps).mean()),'loss':loss/len(ys),'macro_f1':float(f1_score(ys,ps,average='macro',zero_division=0)),'confusion_matrix':cm.tolist(),'per_class_accuracy':[float(cm[i,i]/tot[i]) for i in range(10)],'prediction_histogram':np.bincount(ps,minlength=10).tolist(),'true_histogram':np.bincount(ys,minlength=10).tolist(),'predicted_classes_used':int(np.count_nonzero(np.bincount(ps,minlength=10)))}
def aggregate():
 rs=[json.loads((OUT/f'seed{s}_result.json').read_text()) for s in SEEDS if (OUT/f'seed{s}_result.json').exists()];rows=[{'dataset':'CIFAR10-DVS','representation':'binary','seed':r['seed'],'checkpoint':r['checkpoint'],'test_samples':r['test']['samples'],'correct':r['test']['correct'],'accuracy_percent':100*r['test']['accuracy'],'status':r['status']} for r in rs]
 with (OUT/'clean_accuracy_by_seed.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={'status':'RUNNING','completed_seeds':len(rs),'remaining_seeds':3-len(rs)}
 if len(rs)==3 and all(r['status']=='PASS' for r in rs):
  v=[100*r['test']['accuracy'] for r in rs];m=statistics.mean(v);sd=statistics.stdev(v);summary={'status':'PASS','seeds':list(SEEDS),'mean_accuracy_percent':m,'sample_std_accuracy_percent':sd,'ci95_low':m-4.302652729911275*sd/(3**.5),'ci95_high':m+4.302652729911275*sd/(3**.5)}
 atomic(OUT/'clean_accuracy_summary.json',summary)
def train(seed,x,y,split,cache_path,device):
 checkpoint=CKPT/f'cifar10_dvs_binary_seed{seed}_best.pt';latest=CKPT/f'cifar10_dvs_binary_seed{seed}_latest.pt';rp=OUT/f'seed{seed}_result.json';marker=OUT/f'seed{seed}.complete.json'
 if marker.exists() and rp.exists() and checkpoint.exists() and json.loads(marker.read_text()).get('checkpoint_sha256')==sha(checkpoint):return json.loads(rp.read_text())
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True)
 config={'dataset':'CIFAR10-DVS','representation':'binary_occupancy','seed':seed,'T':10,'shape':[2,128,128],'binary_rule':'occupied cell = 1','normalization':'none','cache_dtype':'uint8; cast to float32 at model input','initialization':'from scratch','max_epochs':350,'patience':20,'batch_size':16,'learning_rate':.001,'weight_decay':.0001}
 model=CIFAR10DVSConvSNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=8,min_lr=1e-6);validation=loader(x,y,split['validation_indices'],16,False,seed);best=-1.;best_loss=float('inf');stale=0;hist=[];started=time.perf_counter();start_epoch=1
 if latest.exists():
  state=torch.load(latest,map_location=device,weights_only=True);model.load_state_dict(state['model_state'],strict=True);opt.load_state_dict(state['optimizer_state']);scheduler.load_state_dict(state['scheduler_state']);best=float(state['best_accuracy']);best_loss=float(state['best_loss']);stale=int(state['stale']);hist=list(state['history']);start_epoch=int(state['epoch'])+1;print(f'CIFAR Binary seed={seed} resuming at epoch={start_epoch}',flush=True)
 elif checkpoint.exists() and not marker.exists():
  interrupted=checkpoint.with_name(checkpoint.stem+'_interrupted_epoch1.pt')
  if not interrupted.exists():os.replace(checkpoint,interrupted)
  print(f'CIFAR Binary seed={seed} restarting incomplete pre-resume run from scratch',flush=True)
 for epoch in range(start_epoch,351):
  model.train();seen=correct=0;loss_sum=0.;tic=time.perf_counter()
  for xb,yb in loader(x,y,split['train_indices'],16,True,seed+epoch):
   xb,yb=xb.to(device,non_blocking=True),yb.to(device,non_blocking=True);opt.zero_grad(set_to_none=True);z=model(xb);loss=F.cross_entropy(z,yb);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();seen+=len(yb);correct+=int((z.argmax(1)==yb).sum());loss_sum+=float(loss.detach())*len(yb)
  val=evaluate(model,validation,device);scheduler.step(val['accuracy']);improved=val['accuracy']>best or (val['accuracy']==best and val['loss']<best_loss)
  if improved:best,best_loss,stale=val['accuracy'],val['loss'],0;torch.save({'model_state':model.state_dict(),'seed':seed,'best_epoch':epoch,'validation_accuracy':best,'validation_loss':best_loss,'config':config},checkpoint)
  else:stale+=1
  row={'epoch':epoch,'train_accuracy':correct/seen,'train_loss':loss_sum/seen,'validation_accuracy':val['accuracy'],'validation_loss':val['loss'],'lr':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-tic};hist.append(row)
  torch.save({'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'scheduler_state':scheduler.state_dict(),'epoch':epoch,'best_accuracy':best,'best_loss':best_loss,'stale':stale,'history':hist,'seed':seed,'config':config},latest)
  print(f'CIFAR Binary seed={seed} epoch={epoch} train={correct/seen:.4f} val={val["accuracy"]:.4f} best={best:.4f} lr={row["lr"]:.6g}',flush=True)
  if stale>=20:break
 payload=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(payload['model_state']);test=evaluate(model,loader(x,y,split['test_indices'],16,False,seed),device);collapse=test['predicted_classes_used']==10
 result={'status':'PASS' if collapse else 'FAIL_CLASS_COLLAPSE','seed':seed,'checkpoint':str(checkpoint.relative_to(ROOT)).replace('\\','/'),'checkpoint_sha256':sha(checkpoint),'cache_path':str(cache_path.relative_to(ROOT)).replace('\\','/'),'cache_sha256':sha(cache_path),'best_epoch':payload['best_epoch'],'validation_accuracy':best,'test':test,'class_collapse_check_pass':collapse,'runtime_seconds':time.perf_counter()-started,'config':config};atomic(rp,result);atomic(marker,{'status':result['status'],'checkpoint_sha256':result['checkpoint_sha256']})
 with (LOG/f'seed{seed}_history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
 print(f'CIFAR10-DVS | binary | {seed} | test | {test["samples"]} | Acc {100*test["accuracy"]:.2f}% | {result["status"]}',flush=True);return result
def main():
 if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError('wrong interpreter')
 from tonic.datasets import CIFAR10DVS
 OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True);native=CIFAR10DVS(save_to=str(ROOT/'data/cifar10_dvs'));x,y,cp=build_cache(native);split=json.loads((ROOT/'results/cifar10_dvs_snn_seed42_split.json').read_text())
 for seed in SEEDS:train(seed,x,y,split,cp,torch.device('cuda'));aggregate()
if __name__=='__main__':main()
