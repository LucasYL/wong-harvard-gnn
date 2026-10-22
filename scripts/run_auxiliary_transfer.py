"""Frozen, sequential auxiliary pretraining and target fine-tuning experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(key,'2')
import argparse,csv,fcntl,gzip,hashlib,json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn as tnn
from chemprop import data,featurizers,nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
from antibacterial.descriptor_batch import check_protocol,verify_run
from antibacterial.neural import make_loader,Progress,machine_runtime
from antibacterial.neural_descriptors import model,dataset,all_metrics,mean_scores
from antibacterial.descriptors import read_rows,load_features
from antibacterial.baseline import write_predictions
from antibacterial.compare import verify_predictions

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/transfer_v1.json';FROZEN=ROOT/'configs/transfer_v1_frozen.json'
BATCH=ROOT/'outputs/coadd-transfer-v1'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def identity():
 return {str(p.relative_to(ROOT)):digest(p) for p in [CONFIG,Path(__file__),ROOT/'data/processed/coadd_auxiliary_v1/manifest.json',ROOT/'configs/descriptor_experiment_frozen_v1.json']}
def check():
 check_protocol();f=json.loads(FROZEN.read_text())
 if identity()!=f['identity']:raise ValueError('Transfer experiment identity changed')
 d=ROOT/'data/processed/coadd_auxiliary_v1';m=json.loads((d/'manifest.json').read_text())
 for name,h in m['output_sha256'].items():
  if digest(d/name)!=h:raise ValueError('Auxiliary data changed')
 return json.loads(CONFIG.read_text())
def source_network():
 return nn.BondMessagePassing(d_h=300,depth=3,dropout=0.0),nn.MeanAggregation(),tnn.Sequential(tnn.Linear(300,300),tnn.ReLU(),tnn.Linear(300,1))
def source_dataset(rows):
 ds=data.MoleculeDataset([data.MoleculeDatapoint.from_smi(r['parent_smiles'],np.asarray([r['inhibition_percent']/100],dtype=np.float32)) for r in rows],featurizers.SimpleMoleculeMolGraphFeaturizer());ds.cache=False;return ds
def budget(config):
 if machine_runtime()['peak_rss_mib']>config['memory_budget_mib']:raise RuntimeError('Process memory budget exceeded')
def source(seed,pilot):
 config=check();pl.seed_everything(seed,workers=True);start=time.perf_counter()
 dest=ROOT/f'outputs/coadd-transfer-v1-source{seed}'
 if pilot:dest=ROOT/f'outputs/coadd-transfer-smoke-source{seed}'
 dest.mkdir(parents=True,exist_ok=False)
 d=ROOT/config['auxiliary_dataset'];rs={s:[json.loads(x) for x in gzip.open(d/f'{s}.jsonl.gz','rt')] for s in ('train','validation')}
 if pilot:rs={s:rows[:1024 if s=='train' else 128] for s,rows in rs.items()}
 val_keys=[r['compound_key'] for r in rs['validation']]
 loaders={s:make_loader(source_dataset(rows),32,shuffle=s=='train',seed=seed) for s,rows in rs.items()};del rs
 mp,agg,head=source_network();params=list(mp.parameters())+list(head.parameters());opt=torch.optim.Adam(params,lr=config['source_lr']);loss_fn=tnn.SmoothL1Loss(beta=1.0)
 best=float('inf');history=[]
 for epoch in range(1,(1 if pilot else config['source_epochs'])+1):
  begun=time.perf_counter();mp.train();head.train()
  for i,b in enumerate(loaders['train']):
   graph,vd,xd,y,*_=b;opt.zero_grad();pred=head(agg(mp(graph,vd),graph.batch));loss=loss_fn(pred,y)
   if not torch.isfinite(loss):raise ValueError('Nonfinite source loss')
   loss.backward();opt.step()
   if i%100==0:budget(config)
  mp.eval();head.eval();total=0.;n=0
  with torch.no_grad():
   for graph,vd,xd,y,*_ in loaders['validation']:
    p=head(agg(mp(graph,vd),graph.batch));v=loss_fn(p,y)
    if not torch.isfinite(v):raise ValueError('Nonfinite source validation loss')
    total+=float(v)*len(y);n+=len(y)
  value=total/n;history.append({'epoch':epoch,'validation_smooth_l1':value,'seconds':time.perf_counter()-begun,'peak_rss_mib':machine_runtime()['peak_rss_mib']})
  state={'encoder':mp.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'seed':seed,'identity':identity()}
  torch.save(state,dest/'last.pt')
  if value<best:best=value;torch.save(state,dest/'best.pt')
  save(dest/'epochs.json',history);print(json.dumps(history[-1]),flush=True)
 state=torch.load(dest/'best.pt',map_location='cpu',weights_only=False);mp.load_state_dict(state['encoder']);head.load_state_dict(state['head']);mp.eval();head.eval();predictions=[];observed=[];total=0
 with torch.no_grad():
  for graph,vd,xd,y,*_ in loaders['validation']:
   p=head(agg(mp(graph,vd),graph.batch));total+=float(loss_fn(p,y))*len(y);predictions.extend(p.reshape(-1).tolist());observed.extend(y.reshape(-1).tolist())
 if abs(total/len(observed)-best)>1e-7:raise ValueError('Source checkpoint loss mismatch')
 with (dest/'validation.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['compound_key','inhibition_fraction','prediction']);writer.writerows(zip(val_keys,observed,predictions))
 check();save(dest/'results.json',{'complete':True,'pilot_only':pilot,'seed':seed,'identity':identity(),'best_validation_loss':best,'epochs':history,'checkpoint_sha256':digest(dest/'best.pt'),'seconds':time.perf_counter()-start,'runtime':machine_runtime()})
def target(seed,pilot):
 conf=check();base=json.loads((ROOT/conf['target_config']).read_text());pl.seed_everything(seed,workers=True);start=time.perf_counter()
 tag='coadd-transfer-smoke' if pilot else 'coadd-transfer-v1';dest=ROOT/f'outputs/{tag}-target{seed}';dest.mkdir(parents=True,exist_ok=False)
 src=ROOT/f'outputs/{tag}-source{seed}';sr=json.loads((src/'results.json').read_text())
 if not sr['complete'] or sr['pilot_only']!=pilot or sr['identity']!=identity() or digest(src/'best.pt')!=sr['checkpoint_sha256']:raise ValueError('Source checkpoint mismatch')
 rows={s:read_rows(s,labels=True) for s in ('train','validation')};features={s:load_features(s,rows[s]) for s in rows}
 if pilot:
  for s in rows:rows[s]=rows[s][:1024 if s=='train' else 256];features[s]=features[s][:len(rows[s])]
 loaders={s:make_loader(dataset(rows[s],features[s]),32,shuffle=s=='train',seed=seed) for s in rows}
 network=model(base,pilot);head_before={k:v.clone() for k,v in network.predictor.state_dict().items()};network.message_passing.load_state_dict(torch.load(src/'best.pt',map_location='cpu',weights_only=False)['encoder'],strict=True)
 if not all(torch.equal(v,network.predictor.state_dict()[k]) for k,v in head_before.items()):raise ValueError('Target head changed during encoder transfer')
 cp=ModelCheckpoint(dirpath=dest/'checkpoints',filename='best-{epoch:02d}',monitor='val/ap',mode='max',save_top_k=1,save_last=True)
 trainer=pl.Trainer(accelerator='cpu',devices=1,precision='32-true',deterministic=True,max_epochs=2 if pilot else base['max_epochs'],min_epochs=1 if pilot else base['min_epochs'],callbacks=[Progress(dest,False,conf['memory_budget_mib']),cp,EarlyStopping(monitor='val/ap',mode='max',patience=base['early_stopping_patience'],check_finite=True)],logger=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0)
 trainer.fit(network,loaders['train'],loaders['validation']);network.load_state_dict(torch.load(cp.best_model_path,map_location='cpu',weights_only=False)['state_dict'])
 scores=torch.cat(trainer.predict(network,loaders['validation'])).detach().numpy().reshape(-1);topk=[20,50,100]
 if len(scores)!=len(rows['validation']) or not np.isfinite(scores).all():raise ValueError('Invalid target predictions')
 m=all_metrics([r['activity'] for r in rows['validation']],scores,[r['compound_key'] for r in rows['validation']],topk)
 if abs(m['average_precision']-float(cp.best_model_score))>1e-6:raise ValueError('Checkpoint prediction mismatch')
 write_predictions(dest/'validation.csv',rows['validation'],[r['activity'] for r in rows['validation']],scores)
 verify_predictions(dest/'validation.csv',rows['validation'],topk);check()
 save(dest/'results.json',{'complete':True,'pilot_only':pilot,'seed':seed,'identity':identity(),'source_checkpoint_sha256':sr['checkpoint_sha256'],'checkpoint':str(Path(cp.best_model_path).relative_to(ROOT)),'checkpoint_sha256':digest(Path(cp.best_model_path)),'metrics':m,'seconds':time.perf_counter()-start,'runtime':machine_runtime(),'final_test_evaluated':False})
def summarize():
 conf=check();rows=read_rows('validation',labels=True);scores=[];records=[]
 for seed in conf['seeds']:
  p=ROOT/f'outputs/coadd-transfer-v1-target{seed}';f=p/'results.json'
  if not f.exists():continue
  r=json.loads(f.read_text())
  if not r['complete'] or r['pilot_only'] or r['identity']!=identity() or digest(ROOT/r['checkpoint'])!=r['checkpoint_sha256']:raise ValueError('Target output changed')
  s,m,_=verify_predictions(p/'validation.csv',rows,[20,50,100]);scores.append(s);records.append({'seed':seed,'metrics':all_metrics([x['activity'] for x in rows],s,[x['compound_key'] for x in rows],[20,50,100])})
  if abs(m['average_precision']-r['metrics']['average_precision'])>1e-12:raise ValueError('Saved metrics mismatch')
 ensemble=None
 if len(scores)==len(conf['seeds']):
  s=mean_scores(scores);write_predictions(BATCH/'ensemble_validation.csv',rows,[r['activity'] for r in rows],s);ensemble=all_metrics([r['activity'] for r in rows],s,[r['compound_key'] for r in rows],[20,50,100])
 baseline=[];scratch=[];ys=[r['activity'] for r in rows];keys=[r['compound_key'] for r in rows]
 for seed in conf['seeds']:
  name=f'chemprop-descriptors-v1-seed{seed}';_,s=verify_run(name,rows,[20,50,100]);scratch.append(s);baseline.append({'name':name,'metrics':all_metrics(ys,s,keys,[20,50,100])})
 baseline.append({'name':'descriptors_3','metrics':all_metrics(ys,mean_scores(scratch),keys,[20,50,100])})
 old_config=json.loads((ROOT/conf['target_config']).read_text());s,_,_=verify_predictions(ROOT/'outputs'/old_config['baseline_fingerprint_run']/'nearest_neighbour_validation.csv',rows,[20,50,100]);baseline.append({'name':'nearest_neighbour_validation','metrics':all_metrics(ys,s,keys,[20,50,100])})
 result={'complete':ensemble is not None,'members':records,'ensemble':ensemble,'baselines':baseline,'evaluation_partition':'validation','final_test_evaluated':False,'limitations':['Development comparison after prior V0 test exposure.','Measures this fixed auxiliary encoder-pretraining procedure; does not isolate every possible effect of extra optimization.','No new independent, wet-lab or actinomycete validation.']};save(ROOT/'reports/coadd_transfer_comparison_v1.json',result);return result
def queue():
 conf=check();BATCH.mkdir(parents=True,exist_ok=True)
 with (BATCH/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (BATCH/'status.json').exists():raise ValueError('Existing queue; preserve it and inspect before recovery')
  status={'state':'running','pid':os.getpid(),'started_at_utc':now(),'completed_stages':[]};save(BATCH/'status.json',status)
  try:
   for seed in conf['seeds']:
    for stage in ('source','target'):
     check();status.update(current_seed=seed,current_stage=stage);save(BATCH/'status.json',status)
     with (ROOT/f'logs/coadd-transfer-v1-{stage}{seed}.log').open('x') as log:
      child=subprocess.run([sys.executable,str(Path(__file__).resolve()),stage,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
     if child.returncode:raise RuntimeError(f'{stage} seed {seed} failed with {child.returncode}; inspect log')
     status['completed_stages'].append(f'{stage}{seed}');save(BATCH/'status.json',status)
     if stage=='target':summarize()
   status.update(state='complete',completed_at_utc=now(),current_seed=None,current_stage=None);save(BATCH/'status.json',status)
  except Exception as e:status.update(state='failed',error=str(e),updated_at_utc=now());save(BATCH/'status.json',status);raise
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','source','target','queue','summarize']);p.add_argument('--seed',type=int,default=17);p.add_argument('--pilot',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if a.action=='freeze':
  if FROZEN.exists():raise ValueError('Cannot overwrite frozen experiment')
  check_protocol();save(FROZEN,{'created_at_utc':now(),'identity':identity(),'config':json.loads(CONFIG.read_text())});return
 if a.seed not in json.loads(CONFIG.read_text())['seeds']:raise ValueError('Undeclared seed')
 if a.action in ('source','target'):globals()[a.action](a.seed,a.pilot)
 else:globals()[a.action]()
if __name__=='__main__':main()
