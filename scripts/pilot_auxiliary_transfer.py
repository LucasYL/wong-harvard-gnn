"""Bounded resource/transfer-shape pilot; no Wong prediction scores or final test labels."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(key,'2')
import gzip,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem,RDLogger
from chemprop import data,featurizers,nn
from antibacterial.neural import make_loader
from antibacterial.neural_descriptors import model as target_model
from antibacterial.descriptor_batch import check_protocol
ROOT=Path(__file__).resolve().parents[1]
def main():
 start=time.perf_counter();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20260910);RDLogger.DisableLog('rdApp.warning');check_protocol()
 path=ROOT/'data/processed/coadd_auxiliary_v1/train.jsonl.gz'
 manifest=json.loads((path.parent/'manifest.json').read_text())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['output_sha256'][path.name]
 rows=[json.loads(x) for x in gzip.open(path,'rt')]
 # Resource sample selection is independent of measured responses.
 selected=sorted(rows,key=lambda r:hashlib.sha256(r['compound_key'].encode()).hexdigest())[:1024]
 largest=sorted(rows,key=lambda r:Chem.MolFromSmiles(r['parent_smiles']).GetNumHeavyAtoms(),reverse=True)[:32]
 feat=featurizers.SimpleMoleculeMolGraphFeaturizer()
 def dataset(rs):return data.MoleculeDataset([data.MoleculeDatapoint.from_smi(r['parent_smiles'],np.array([r['inhibition_percent']/100],dtype=np.float32)) for r in rs],feat)
 mp=nn.BondMessagePassing(d_h=300,depth=3,dropout=0.0);agg=nn.MeanAggregation();head=torch.nn.Sequential(torch.nn.Linear(300,300),torch.nn.ReLU(),torch.nn.Linear(300,1))
 parameters=list(mp.parameters())+list(head.parameters());opt=torch.optim.Adam(parameters,lr=1e-4);criterion=torch.nn.SmoothL1Loss(beta=1.0)
 def step(batch,update=True):
  bmg,vd,xd,ys,*_=batch;opt.zero_grad();pred=head(agg(mp(bmg,vd),bmg.batch));loss=criterion(pred,ys);loss.backward()
  assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
  if update:opt.step()
  return float(loss.detach())
 step(next(iter(make_loader(dataset(largest),32))))
 fit=time.perf_counter()
 for batch in make_loader(dataset(selected),32,shuffle=True,seed=20260910):step(batch)
 pilot_fit=time.perf_counter()-fit
 # Check the shared encoder loads exactly into our existing target architecture.
 config=json.loads((ROOT/'configs/chemprop_descriptors_v1.json').read_text());target=target_model(config,False)
 target.message_passing.load_state_dict(mp.state_dict(),strict=True)
 assert all(torch.equal(v,target.message_passing.state_dict()[k]) for k,v in mp.state_dict().items())
 report={'pilot_only':True,'complete':True,'sample_n':1024,'sample_selection':'compound-key hash independent of responses','source_train_n':len(rows),'largest_batch_molecules':32,'largest_heavy_atoms':max(Chem.MolFromSmiles(r['parent_smiles']).GetNumHeavyAtoms() for r in largest),'target':'continuous inhibition_percent / 100, without clipping','loss':'SmoothL1 beta=1.0','source_architecture':'D-MPNN hidden300 depth3 mean pooling; regression head300->300->1; no descriptors','target_transfer':'encoder only into existing D-MPNN plus 200 descriptors; target head not transferred','finite_forward_backward':True,'encoder_state_transfer_verified':True,'model_weights_saved':False,'wong_labels_or_predictions_used':False,'full_training_started':False,'sample_fit_seconds':pilot_fit,'rough_source_5_epoch_minutes':pilot_fit*(len(rows)/1024)*5/60,'extrapolation_caveat':'Small-sample throughput estimate, excludes whole-dataset startup and thermal effects; not a confirmed full-run duration.','seconds':time.perf_counter()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'auxiliary_manifest_sha256':hashlib.sha256((path.parent/'manifest.json').read_bytes()).hexdigest()}
 (ROOT/'reports/auxiliary_transfer_resource_pilot_v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
