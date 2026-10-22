"""Freeze existing models, reproduce validation inference, then score a reserved assay once."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'2')
import argparse
import csv
import gc
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from descriptastorus.descriptors import rdNormalizedDescriptors
from sklearn.metrics import average_precision_score
from antibacterial.common import sha256, source_hashes
from antibacterial.descriptor_batch import check_protocol
from antibacterial.descriptors import vector, load_features, read_rows
from antibacterial.neural_descriptors import model, dataset, all_metrics, mean_scores
from antibacterial.neural import make_loader, machine_runtime
from antibacterial.baseline import fingerprints, neighbour_predict

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/external_synthemol_v1.json'
FROZEN=ROOT/'configs/external_synthemol_v1_frozen.json'
DEST=ROOT/'outputs/external-synthemol-v1'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def specification():
    check_protocol()
    for name in ('synthemol_dataset_frozen_v1.json',):
        for path,h in json.loads((ROOT/'configs'/name).read_text())['sha256'].items():
            assert sha256(ROOT/path)==h
    for name in ('transfer_v1_frozen.json','synthemol_transfer_v1_frozen.json'):
        for path,h in json.loads((ROOT/'configs'/name).read_text())['identity'].items():
            assert sha256(ROOT/path)==h
    config=json.loads(CONFIG.read_text());entries=[]
    paths=[CONFIG,Path(__file__),ROOT/'configs/chemprop_descriptors_v1.json',ROOT/'configs/v0.json',
           ROOT/'data/processed/synthemol_dataset_v1/manifest.json',ROOT/'data/processed/wong_v0/train.parquet',
           ROOT/'data/processed/wong_v0/validation.parquet',ROOT/'data/processed/wong_descriptors_v1/manifest.json',
           ROOT/'outputs/development-20260909-numeric-fix/nearest_neighbour_validation.csv']
    for family,prefix in [('scratch_descriptors_3','chemprop-descriptors-v1-seed'),('coadd_transfer_3','coadd-transfer-v1-target'),('synthemol_transfer_3','synthemol-transfer-v1-target')]:
        for seed in config['seeds']:
            directory=ROOT/f'outputs/{prefix}{seed}';run=json.loads((directory/'results.json').read_text())
            assert run['complete'] and not run['pilot_only']
            checkpoint=run.get('checkpoint_path',run.get('checkpoint'));assert sha256(ROOT/checkpoint)==run['checkpoint_sha256']
            pred='validation.csv' if family!='scratch_descriptors_3' else 'validation.csv'
            # The descriptor runner uses chemprop_validation.csv in some older runs.
            if not (directory/pred).exists():pred='chemprop_validation.csv'
            assert (directory/pred).exists()
            entries.append({'family':family,'seed':seed,'checkpoint':checkpoint,'validation_predictions':str((directory/pred).relative_to(ROOT))})
            paths.extend([directory/'results.json',ROOT/checkpoint,directory/pred])
    for path in (ROOT/'data/processed/synthemol_dataset_v1').iterdir():
        if path.is_file():paths.append(path)
    return {'config':config,'models':entries,'input_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(paths))},
            'source_hashes':source_hashes(),'packages':machine_runtime()['packages']}


def checked():
    spec=json.loads(FROZEN.read_text())['specification']
    assert specification()==spec,'Frozen external protocol changed'
    return spec


def budget(config):
    assert machine_runtime()['peak_rss_mib']<=config['memory_budget_mib'],'Memory budget exceeded'


def nn_predict(query):
    base=json.loads((ROOT/'configs/v0.json').read_text())
    train=pq.read_table(ROOT/'data/processed/wong_v0/train.parquet').to_pylist()
    train_fp,unused=fingerprints(train,base['fingerprint']);del unused
    query_fp,unused=fingerprints(query,base['fingerprint']);del unused
    scores,_,sim,_=neighbour_predict(train_fp,query_fp,np.asarray([r['activity'] for r in train]),base['nearest_neighbours'])
    return np.asarray(scores),np.asarray(sim)


def neural_predict(entry,query,features,config):
    network=model(json.loads((ROOT/'configs/chemprop_descriptors_v1.json').read_text()))
    network.load_state_dict(torch.load(ROOT/entry['checkpoint'],map_location='cpu',weights_only=False)['state_dict'],strict=True)
    network.eval();scores=[]
    # Placeholder targets satisfy the existing Dataset interface; targets never enter forward().
    rows=[{**r,'activity':0} for r in query]
    loader=make_loader(dataset(rows,features),config['batch_size'])
    with torch.inference_mode():
        for graph,vd,xd,*_ in loader:scores.extend(network(graph,vd,xd).numpy().reshape(-1).tolist())
    result=np.asarray(scores);assert len(result)==len(query) and np.isfinite(result).all()
    assert ((result>=0)&(result<=1)).all()
    del network,loader;gc.collect();budget(config)
    return result


def reproduce(spec):
    config=spec['config'];all_rows=read_rows('validation',labels=False)
    order=sorted(range(len(all_rows)),key=lambda i:hashlib.sha256(all_rows[i]['compound_key'].encode()).hexdigest())[:64]
    rows=[all_rows[i] for i in order];features=load_features('validation',all_rows)[order]
    generator=rdNormalizedDescriptors.RDKit2DNormalized()
    fresh=np.stack([vector(generator,r['parent_smiles'])[0] for r in rows])
    np.testing.assert_allclose(fresh,features,rtol=0,atol=1e-7)
    nn_scores,_=nn_predict(rows);checks=[]
    with (ROOT/'outputs/development-20260909-numeric-fix/nearest_neighbour_validation.csv').open() as f:old={r['compound_key']:float(r['predicted_score']) for r in csv.DictReader(f)}
    assert np.max(np.abs(nn_scores-np.array([old[r['compound_key']] for r in rows])))<1e-12
    for entry in spec['models']:
        scores=neural_predict(entry,rows,features,config)
        with (ROOT/entry['validation_predictions']).open() as f:old={r['compound_key']:float(r['predicted_score']) for r in csv.DictReader(f)}
        error=float(np.max(np.abs(scores-np.array([old[r['compound_key']] for r in rows]))))
        assert error<=config['prediction_tolerance'],(entry,error)
        checks.append({'family':entry['family'],'seed':entry['seed'],'maximum_absolute_error':error})
    return {'passed':True,'n':len(rows),'descriptor_recalculation_matched':True,'nn_matched':True,'neural_checks':checks}


def score_external():
    spec=checked();config=spec['config'];DEST.mkdir(exist_ok=False)
    started=time.perf_counter();save(DEST/'status.json',{'state':'running','stage':'validation_reproduction'})
    try:
        verification=reproduce(spec);save(DEST/'validation_reproduction.json',verification)
        path=ROOT/config['dataset']/'external.jsonl.gz'
        with gzip.open(path,'rt') as f:query=[{k:r[k] for k in ('compound_key','parent_smiles','component_id')} for r in map(json.loads,f)]
        assert len(query)==config['expected_compounds'] and len({r['compound_key'] for r in query})==len(query)
        save(DEST/'status.json',{'state':'running','stage':'structure_only_prediction','n':len(query)})
        generator=rdNormalizedDescriptors.RDKit2DNormalized();features=[];imputed=[]
        for i,r in enumerate(query):
            values,missing=vector(generator,r['parent_smiles']);features.append(values)
            if missing:imputed.append({'compound_key':r['compound_key'],'descriptor_indices':missing})
            if (i+1)%250==0:print(f'Descriptors {i+1}/{len(query)}',flush=True)
        features=np.stack(features);np.save(DEST/'descriptors.npy',features,allow_pickle=False);save(DEST/'descriptor_imputation.json',imputed)
        nn_scores,similarity=nn_predict(query);predictions={'nearest_neighbour':nn_scores};member_metrics=[]
        for entry in spec['models']:
            name=f"{entry['family']}_seed{entry['seed']}";predictions[name]=neural_predict(entry,query,features,config)
            print(f'Predicted {name}',flush=True)
        for family in config['model_groups'][1:]:
            predictions[family]=mean_scores([predictions[f'{family}_seed{s}'] for s in config['seeds']])
        ids=[r['compound_key'] for r in query];names=list(predictions)
        np.save(DEST/'unscored_predictions.npy',np.column_stack([predictions[n] for n in names]),allow_pickle=False)
        save(DEST/'prediction_ids.json',ids)
        save(DEST/'predictions_sealed.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'model_order':names,
            'scores_sha256':sha256(DEST/'unscored_predictions.npy'),'ids_sha256':sha256(DEST/'prediction_ids.json'),
            'query_labels_used_for_predictions':False,'note':'Query labels existed in the source archive but were excluded from inference inputs.'})
        # Only after all scores are saved, attach published labels by stable identity.
        with gzip.open(path,'rt') as f:labels={r['compound_key']:r['activity'] for r in map(json.loads,f)}
        y=np.asarray([labels[k] for k in ids]);assert len(set(y))==2
        measured={name:all_metrics(y,predictions[name],ids,config['top_k']) for name in names}
        rows=[]
        for i,r in enumerate(query):rows.append({**r,'observed_label':int(y[i]),'nearest_wong_train_similarity':float(similarity[i]),**{n:float(predictions[n][i]) for n in names}})
        with (DEST/'predictions.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with (DEST/'predictions.csv').open() as f:saved=list(csv.DictReader(f))
        assert [r['compound_key'] for r in saved]==ids
        for name in names:
            verified=all_metrics([int(r['observed_label']) for r in saved],[float(r[name]) for r in saved],ids,config['top_k'])
            assert verified==measured[name],name
        _,inverse=np.unique([r['component_id'] for r in query],return_inverse=True);n_groups=int(inverse.max()+1)
        rng=np.random.default_rng(config['bootstrap']['seed']);deltas={name:[] for name in config['model_groups'] if name!='scratch_descriptors_3'}
        for _ in range(config['bootstrap']['iterations']):
            weights=np.bincount(rng.integers(n_groups,size=n_groups),minlength=n_groups)[inverse]
            if len(set(y[weights>0]))!=2:continue
            reference=average_precision_score(y,predictions['scratch_descriptors_3'],sample_weight=weights)
            for name in deltas:deltas[name].append(float(average_precision_score(y,predictions[name],sample_weight=weights)-reference))
        result={'complete':True,'protocol_sha256':sha256(FROZEN),'n':len(query),'hits':int(y.sum()),
            'groups':{n:measured[n] for n in config['model_groups']},'members':{n:measured[n] for n in names if '_seed' in n},
            'paired_ap_difference_vs_scratch':{n:{'observed':measured[n]['average_precision']-measured['scratch_descriptors_3']['average_precision'],
                'percentile_95_interval':np.quantile(v,[.025,.975]).tolist(),'valid_replicates':len(v)} for n,v in deltas.items()},
            'descriptor_imputed_compounds':len(imputed),'inference_reproduction':verification,'predictions_sha256':sha256(DEST/'predictions.csv'),
            'seconds':time.perf_counter()-started,'runtime':machine_runtime(),'labels_used_for_training_or_tuning':False,
            'limits':['Related external assay with its own label definition; AP cannot be compared directly to Wong AP because populations/prevalence differ.',
                'This external set is now scored and cannot remain an untouched test for adaptive changes.',
                'Conditional bootstrap intervals do not include retraining uncertainty. No actinomycete-specific or wet-lab validation.']}
        checked();save(DEST/'results.json',result);save(ROOT/'reports/external_synthemol_comparison_v1.json',result)
        save(DEST/'status.json',{'state':'complete','seconds':result['seconds']});print(json.dumps(result['groups'],indent=2))
    except Exception as e:
        save(DEST/'status.json',{'state':'failed','error':str(e)});raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','evaluate']);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if args.action=='freeze':
        assert not FROZEN.exists();save(FROZEN,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'specification':specification()})
    else:score_external()


if __name__=='__main__':main()
