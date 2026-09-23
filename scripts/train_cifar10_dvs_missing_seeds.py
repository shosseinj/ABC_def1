"""Train missing CIFAR10-DVS seeds without modifying seed 42."""
from __future__ import annotations
import csv, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.cifar10_dvs.snn_baseline import events_to_frames
from models.cifar10_dvs_snn import CIFAR10DVSConvSNN

SEEDS=[123,777,2026,6543]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

class MemmapFrames(Dataset):
    def __init__(self, frames, labels, ids): self.frames,self.labels,self.ids=frames,labels,np.asarray(ids,dtype=np.int64)
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        j=int(self.ids[i]); return torch.from_numpy(np.array(self.frames[j],dtype=np.float32,copy=True)),int(self.labels[j])

def loader(frames,labels,ids,batch,shuffle,seed):
    return DataLoader(MemmapFrames(frames,labels,ids),batch_size=batch,shuffle=shuffle,num_workers=0,generator=torch.Generator().manual_seed(seed),pin_memory=True)
def evaluate(model,data,device):
    model.eval();n=correct=0;loss=0.;ys=[];ps=[]
    with torch.no_grad():
        for x,y in data:
            x,y=x.to(device),y.to(device);z=model(x);loss+=float(F.cross_entropy(z,y,reduction='sum'));n+=len(y);p=z.argmax(1);correct+=int((p==y).sum());ys+=y.cpu().tolist();ps+=p.cpu().tolist()
    return {'loss':loss/n,'accuracy':correct/n,'macro_f1':float(f1_score(ys,ps,average='macro',zero_division=0)),'samples':n}

def main():
    from tonic.datasets import CIFAR10DVS
    config_path=ROOT/'configs/cifar10_dvs_snn_seed42.json';base=json.loads(config_path.read_text())
    preserved=ROOT/'checkpoints/cifar10_dvs_snn_seed42_best.pt';before=sha(preserved)
    split_path=ROOT/'results/cifar10_dvs_snn_seed42_split.json';split=json.loads(split_path.read_text())
    dataset=CIFAR10DVS(save_to=str(ROOT/base['data_root']))
    cache_dir=ROOT/'Reports/checkpoints';cache_dir.mkdir(parents=True,exist_ok=True)
    frame_path=cache_dir/'cifar10_dvs_t10_128_float16.npy';label_path=cache_dir/'cifar10_dvs_labels.npy';complete=cache_dir/'cifar10_dvs_cache.complete.json'
    expected=(len(dataset),10,2,128,128)
    if not complete.exists():
        mm=np.lib.format.open_memmap(frame_path,mode='w+',dtype=np.float16,shape=expected);labels=np.empty(len(dataset),dtype=np.int64)
        for i in range(len(dataset)):
            events,label=dataset[i];mm[i]=events_to_frames(events,10);labels[i]=label
            if (i+1)%100==0: print(f'[CIFAR cache] {i+1}/{len(dataset)}',flush=True)
        mm.flush();np.save(label_path,labels);atomic(complete,{'shape':expected,'frames_sha256':sha(frame_path),'labels_sha256':sha(label_path)})
    meta=json.loads(complete.read_text());frames=np.load(frame_path,mmap_mode='r');labels=np.load(label_path,mmap_mode='r')
    if tuple(frames.shape)!=expected or sha(frame_path)!=meta['frames_sha256']: raise RuntimeError('CIFAR cache validation failed')
    device=torch.device(base['device']);summaries=[]
    for seed in SEEDS:
        checkpoint=ROOT/f'checkpoints/cifar10_dvs_snn_seed{seed}_best.pt';marker=cache_dir/f'cifar10_dvs_seed{seed}.complete.json'
        if marker.exists() and checkpoint.exists() and json.loads(marker.read_text()).get('checkpoint_sha256')==sha(checkpoint):
            done=json.loads(marker.read_text());print(f'[seed={seed}] valid completion marker; skipping',flush=True);summaries.append(done);continue
        config=dict(base);config['model_seed']=seed;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=CIFAR10DVSConvSNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=8,min_lr=1e-6)
        best_acc=-1.;best_loss=float('inf');stale=0;history=[];started=time.perf_counter()
        val_loader=loader(frames,labels,split['validation_indices'],config['batch_size'],False,seed)
        for epoch in range(1,config['max_epochs']+1):
            model.train();seen=correct=0;loss_sum=0.;tic=time.perf_counter()
            for x,y in loader(frames,labels,split['train_indices'],config['batch_size'],True,seed+epoch):
                x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);z=model(x);loss=F.cross_entropy(z,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();seen+=len(y);correct+=int((z.argmax(1)==y).sum());loss_sum+=float(loss.detach())*len(y)
            val=evaluate(model,val_loader,device);scheduler.step(val['accuracy']);improved=val['accuracy']>best_acc or (val['accuracy']==best_acc and val['loss']<best_loss)
            if improved: best_acc,best_loss,stale=val['accuracy'],val['loss'],0;torch.save({'model_state':model.state_dict(),'best_epoch':epoch,'validation_accuracy':best_acc,'validation_loss':best_loss,'learning_rate':opt.param_groups[0]['lr'],'config':config},checkpoint)
            else: stale+=1
            row={'epoch':epoch,'train_loss':loss_sum/seen,'train_accuracy':correct/seen,'validation_loss':val['loss'],'validation_accuracy':val['accuracy'],'lr':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-tic};history.append(row)
            print(f"seed={seed} epoch={epoch} train_acc={row['train_accuracy']:.4f} val_acc={val['accuracy']:.4f} best={best_acc:.4f} lr={row['lr']:.6g} seconds={row['seconds']:.1f}",flush=True)
            if stale>=config['early_stop_patience']: break
        payload=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(payload['model_state']);test=evaluate(model,loader(frames,labels,split['test_indices'],config['batch_size'],False,seed),device)
        done={'status':'COMPLETE','seed':seed,'best_epoch':payload['best_epoch'],'validation_accuracy':best_acc,'test':test,'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint),'base_config_sha256':sha(config_path),'split_sha256':sha(split_path),'cache_sha256':meta['frames_sha256'],'runtime_seconds':time.perf_counter()-started,'parameters':model.trainable_parameter_count()};atomic(marker,done);summaries.append(done)
        with (ROOT/f'Reports/logs/cifar10_dvs_seed{seed}_history.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
        print(json.dumps(done,indent=2),flush=True)
    if sha(preserved)!=before: raise RuntimeError('seed-42 checkpoint changed')
    atomic(ROOT/'Reports/results/cifar10_dvs_missing_seed_training_summary.json',{'preserved_seed42_sha256':before,'runs':summaries})

if __name__=='__main__': main()
