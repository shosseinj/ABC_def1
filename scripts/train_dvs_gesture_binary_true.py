"""Train true Binary-grid DVS-Gesture SNN checkpoints from scratch."""
from __future__ import annotations
import csv, hashlib, json, os, random, statistics, sys, time
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from models.dvs_gesture_snn import DVSGestureConvSNN

PYTHON=Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
SEEDS=(42,123,777);OUT=ROOT/'Reports/results/dvs_gesture_binary_true';CKPT=ROOT/'checkpoints/dvs_gesture_binary_true';LOG=ROOT/'Reports/logs/dvs_gesture_binary_true'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def binary_frames(events,bins=10,size=64):
 out=np.zeros((bins,2,size,size),dtype=np.uint8)
 if not len(events):return out
 x=np.asarray(events['x'],dtype=np.int64)//2;y=np.asarray(events['y'],dtype=np.int64)//2;t=np.asarray(events['t'],dtype=np.int64);p=np.asarray(events['p'],dtype=np.int64)
 duration=max(int(t.max())-int(t.min())+1,1);tb=np.minimum(((t-t.min())*bins)//duration,bins-1);valid=(x>=0)&(x<size)&(y>=0)&(y<size)&((p==0)|(p==1))
 out[tb[valid],p[valid],y[valid],x[valid]]=1
 return out
def cache(native,name):
 path=OUT/f'{name}_t10_binary_uint8.npz'
 if path.exists():
  d=np.load(path);return d['frames'],d['labels'],path
 x=np.empty((len(native),10,2,64,64),dtype=np.uint8);y=np.empty(len(native),dtype=np.int64)
 for i in range(len(native)):
  e,l=native[i];x[i]=binary_frames(e);y[i]=l
  if (i+1)%100==0 or i+1==len(native):print(f'DVS Binary preprocessing {name} {i+1}/{len(native)}',flush=True)
 tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,frames=x,labels=y);os.replace(tmp,path);return x,y,path
class Arrays(Dataset):
 def __init__(self,x,y):self.x,self.y=x,y
 def __len__(self):return len(self.y)
 def __getitem__(self,i):return torch.from_numpy(np.array(self.x[i],dtype=np.float32,copy=True)),int(self.y[i])
def loader(x,y,ids,batch,shuffle,seed):return DataLoader(Subset(Arrays(x,y),list(map(int,ids))),batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed))
@torch.no_grad()
def evaluate(model,data,device):
 model.eval();ys=[];ps=[];loss=0.
 for x,y in data:
  x,y=x.to(device),y.to(device);z=model(x);loss+=float(F.cross_entropy(z,y,reduction='sum'));ys+=y.cpu().tolist();ps+=z.argmax(1).cpu().tolist()
 ys=np.asarray(ys);ps=np.asarray(ps);cm=confusion_matrix(ys,ps,labels=range(11));tot=cm.sum(1)
 return {'samples':len(ys),'correct':int((ys==ps).sum()),'accuracy':float((ys==ps).mean()),'loss':loss/len(ys),'macro_f1':float(f1_score(ys,ps,average='macro',zero_division=0)),'confusion_matrix':cm.tolist(),'per_class_accuracy':[float(cm[i,i]/tot[i]) for i in range(11)],'prediction_histogram':np.bincount(ps,minlength=11).tolist(),'true_histogram':np.bincount(ys,minlength=11).tolist(),'predicted_classes_used':int(np.count_nonzero(np.bincount(ps,minlength=11)))}
def aggregate():
 rs=[json.loads((OUT/f'seed{s}_result.json').read_text()) for s in SEEDS if (OUT/f'seed{s}_result.json').exists()];rows=[{'dataset':'DVS-Gesture','representation':'binary','seed':r['seed'],'checkpoint':r['checkpoint'],'test_samples':r['test']['samples'],'correct':r['test']['correct'],'accuracy_percent':100*r['test']['accuracy'],'status':r['status']} for r in rs]
 with (OUT/'clean_accuracy_by_seed.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={'status':'RUNNING','completed_seeds':len(rs),'remaining_seeds':3-len(rs)}
 if len(rs)==3 and all(r['status']=='PASS' for r in rs):
  v=[100*r['test']['accuracy'] for r in rs];m=statistics.mean(v);sd=statistics.stdev(v);summary={'status':'PASS','seeds':list(SEEDS),'mean_accuracy_percent':m,'sample_std_accuracy_percent':sd,'ci95_low':m-4.302652729911275*sd/(3**.5),'ci95_high':m+4.302652729911275*sd/(3**.5)}
 atomic(OUT/'clean_accuracy_summary.json',summary)
def train(seed,tx,ty,vx,vy,train_ids,val_ids,device):
 checkpoint=CKPT/f'dvs_gesture_binary_seed{seed}_best.pt';result_path=OUT/f'seed{seed}_result.json';marker=OUT/f'seed{seed}.complete.json'
 if marker.exists() and result_path.exists() and checkpoint.exists() and json.loads(marker.read_text()).get('checkpoint_sha256')==sha(checkpoint):return json.loads(result_path.read_text())
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True)
 config={'dataset':'DVS-Gesture','representation':'binary_occupancy','seed':seed,'T':10,'shape':[2,64,64],'binary_rule':'occupied cell = 1','normalization':'none','initialization':'from scratch','epochs':60,'minimum_epochs':12,'patience':10,'batch_size':16,'learning_rate':.001,'weight_decay':.0001}
 model=DVSGestureConvSNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);validation=loader(tx,ty,val_ids,16,False,seed);best=-1.;best_loss=float('inf');stale=0;hist=[];started=time.perf_counter()
 for epoch in range(1,61):
  model.train();seen=correct=0;loss_sum=0.
  for x,y in loader(tx,ty,train_ids,16,True,seed+epoch):
   x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);z=model(x);loss=F.cross_entropy(z,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();seen+=len(y);correct+=int((z.argmax(1)==y).sum());loss_sum+=float(loss.detach())*len(y)
  val=evaluate(model,validation,device);improved=val['accuracy']>best or (val['accuracy']==best and val['loss']<best_loss)
  if improved:best,best_loss,stale=val['accuracy'],val['loss'],0;torch.save({'model_state':model.state_dict(),'seed':seed,'epoch':epoch,'config':config},checkpoint)
  else:stale+=1
  hist.append({'epoch':epoch,'train_accuracy':correct/seen,'train_loss':loss_sum/seen,'validation_accuracy':val['accuracy'],'validation_loss':val['loss']});print(f'DVS Binary seed={seed} epoch={epoch} train={correct/seen:.4f} val={val["accuracy"]:.4f} best={best:.4f}',flush=True)
  if epoch>=12 and stale>=10:break
 payload=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(payload['model_state']);test=evaluate(model,loader(vx,vy,range(len(vy)),16,False,seed),device);collapse=test['predicted_classes_used']==11
 result={'status':'PASS' if collapse else 'FAIL_CLASS_COLLAPSE','seed':seed,'checkpoint':str(checkpoint.relative_to(ROOT)).replace('\\','/'),'checkpoint_sha256':sha(checkpoint),'best_epoch':payload['epoch'],'validation_accuracy':best,'test':test,'class_collapse_check_pass':collapse,'runtime_seconds':time.perf_counter()-started,'config':config};atomic(result_path,result);atomic(marker,{'status':result['status'],'checkpoint_sha256':result['checkpoint_sha256']})
 with (LOG/f'seed{seed}_history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
 print(f'DVS-Gesture | binary | {seed} | full test | {len(vy)} | Acc {100*test["accuracy"]:.2f}% | {result["status"]}',flush=True);return result
def main():
 if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError('wrong interpreter')
 from tonic.datasets import DVSGesture
 OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True);native_train=DVSGesture(save_to=str(ROOT/'data/dvs_gesture'),train=True);native_test=DVSGesture(save_to=str(ROOT/'data/dvs_gesture'),train=False);tx,ty,_=cache(native_train,'train');vx,vy,_=cache(native_test,'test');ids=np.arange(len(ty));train_ids,val_ids=train_test_split(ids,test_size=.2,random_state=42,stratify=ty)
 for seed in SEEDS:train(seed,tx,ty,vx,vy,train_ids,val_ids,torch.device('cuda'));aggregate()
if __name__=='__main__':main()
