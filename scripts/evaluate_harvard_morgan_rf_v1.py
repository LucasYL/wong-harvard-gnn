"""Post-hoc addition of already trained Morgan-RF baselines; no fitting or tuning."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(k, '2')
import argparse
import csv
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from antibacterial.baseline import fingerprints, metrics
from antibacterial.common import sha256, runtime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed/harvard_rn4220_external_v1/scaffold_unseen.jsonl.gz'
OLD = ROOT/'outputs/harvard-rn4220-evaluation-v1'
OUT = ROOT/'outputs/harvard-morgan-rf-v1'
PROTOCOL = ROOT/'configs/harvard_morgan_rf_v1_frozen.json'
SEEDS = [17,29,43]


def save(p, obj):
    p.parent.mkdir(exist_ok=True, parents=True)
    q = p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2)+'\n');os.replace(q,p)


def status(stage, **kw):
    obj = {'state':'running','stage':stage,'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw}
    save(OUT/'status.json',obj);print(json.dumps(obj),flush=True)


def freeze():
    assert not PROTOCOL.exists()
    old = json.loads((ROOT/'configs/final_evaluation_v1.json').read_text())['specification']
    entries = [e for e in old['models'] if e['family']=='random_forest']
    assert [e['seed'] for e in entries] == SEEDS
    for e in entries:
        assert sha256(ROOT/e['model']) == old['input_hashes'][e['model']]
    pilot = ROOT/'reports/harvard_morgan_rf_pilot_v1.json'
    assert max(json.loads(pilot.read_text())['reproduction_errors']) < 1e-12
    paths = [Path(__file__), DATA, pilot, ROOT/'configs/v0.json',ROOT/'configs/final_evaluation_v1.json',
             ROOT/'src/antibacterial/baseline.py', ROOT/'src/antibacterial/common.py',
             OLD/'predictions.csv', OLD/'results.json', OLD/'verification.json',
             ROOT/'data/processed/wong_v0/train.parquet']+[ROOT/e['model'] for e in entries]
    save(PROTOCOL, {'frozen_at_utc':datetime.now(timezone.utc).isoformat(), 'entries':entries,
       'input_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in paths},
       'n':225829,'hits':601,'chunk_size':4096,'seeds':SEEDS,'rf_summary':'three saved members plus equal-mean ensemble; no best-test-seed selection',
       'top_k':[20,50,100], 'primary':'AP', 'tie_policy':'score descending, ID ascending; also cutoff-tie expectation',
       'bootstrap':{'seed':20260911,'iterations':2000,'unit':'Murcko scaffold','comparison':'RF ensemble minus each original comparator'},
       'design':'Post-hoc additional baseline after original Harvard outcomes were reviewed. RF weights were trained and saved before that evaluation.',
       'no_training_tuning':True,'do_not_overwrite_original_evaluation':True,
       'scope':'Same Harvard cohort and own trained models; not a reproduction of Wong RF hyperparameter search.'})


def checked():
    c = json.loads(PROTOCOL.read_text())
    for p,h in c['input_hashes'].items():assert sha256(ROOT/p)==h,p
    return c


def cached_ap(y,w,cache):
    order,starts=cache
    pos=np.add.reduceat((w*y)[order],starts);total=np.cumsum(np.add.reduceat(w[order],starts))
    prec=np.divide(np.cumsum(pos),total,out=np.zeros_like(total,dtype=float),where=total>0)
    return float(np.dot(pos,prec)/pos.sum())


def evaluate():
    c=checked();OUT.mkdir(exist_ok=False);start=time.perf_counter()
    try:
        status('load_saved_RF_models')
        models=[joblib.load(ROOT/e['model']) for e in c['entries']]
        for m in models:assert m.classes_.tolist()==[0,1] and m.n_features_in_==2048
        base=json.loads((ROOT/'configs/v0.json').read_text())
        with gzip.open(DATA,'rt') as f:query=[{k:r[k] for k in ['compound_key','parent_smiles']} for r in map(json.loads,f)]
        ids=[r['compound_key'] for r in query];assert len(ids)==len(set(ids))==c['n']
        matrix=np.empty((len(ids),3),dtype=np.float64)
        for off in range(0,len(query),c['chunk_size']):
            rows=query[off:off+c['chunk_size']];_,x=fingerprints(rows,base['fingerprint'])
            for j,m in enumerate(models):matrix[off:off+len(rows),j]=m.predict_proba(x)[:,1]
            status('predicting',completed=off+len(rows),total=len(query),seconds=time.perf_counter()-start,peak_rss_mib=runtime()['peak_rss_mib'])
            assert runtime()['peak_rss_mib']<3072
        assert np.isfinite(matrix).all() and ((matrix>=0)&(matrix<=1)).all()
        np.save(OUT/'RF_member_scores.npy',matrix,allow_pickle=False);save(OUT/'prediction_ids.json',ids)
        save(OUT/'predictions_sealed.json',{'protocol_sha256':sha256(PROTOCOL),'scores_sha256':sha256(OUT/'RF_member_scores.npy'),
             'ids_sha256':sha256(OUT/'prediction_ids.json'),'labels_used_in_forward':False,'seeds':SEEDS})
        prediction_seconds=time.perf_counter()-start
        del query,models,x
        with gzip.open(DATA,'rt') as f:observed=list(map(json.loads,f))
        assert [r['compound_key'] for r in observed]==ids
        y=np.array([r['author_activity'] for r in observed]);assert y.sum()==c['hits']
        _,groups=np.unique([r['scaffold'] for r in observed],return_inverse=True);del observed
        pred={f'morgan_rf_seed{s}':matrix[:,j] for j,s in enumerate(SEEDS)}
        pred['morgan_rf_3']=matrix.mean(axis=1)
        measured={n:metrics(y,v,ids,c['top_k']) for n,v in pred.items()}
        prior=json.loads((OLD/'results.json').read_text())
        with (OLD/'predictions.csv').open() as f:
            prior_rows=list(csv.DictReader(f))
        assert [r['compound_key'] for r in prior_rows]==ids
        assert [int(r['observed_label']) for r in prior_rows]==y.tolist()
        all_pred={n:np.array([float(r[n]) for r in prior_rows]) for n in prior['groups']}
        del prior_rows
        all_pred['morgan_rf_3']=pred['morgan_rf_3']
        all_metrics={n:metrics(y,v,ids,c['top_k']) for n,v in all_pred.items()}
        for n in prior['groups']:
            assert all_metrics[n]['average_precision']==prior['groups'][n]['average_precision']
            assert all_metrics[n]['top_k']==prior['groups'][n]['top_k']
        caches={}
        for n,v in all_pred.items():
            order=np.argsort(-v,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(v[order]))+1];caches[n]=(order,starts)
            assert abs(cached_ap(y,np.ones(len(y)),caches[n])-average_precision_score(y,v))<1e-12
        # Verify weighted/tied AP independently before uncertainty calculation.
        rng=np.random.default_rng(123); toy_y=np.array([0,1]*100);toy_s=rng.integers(7,size=200);w=rng.integers(4,size=200).astype(float)
        order=np.argsort(-toy_s);starts=np.r_[0,np.flatnonzero(np.diff(toy_s[order]))+1]
        assert abs(cached_ap(toy_y,w,(order,starts))-average_precision_score(toy_y,toy_s,sample_weight=w))<1e-12
        g=int(groups.max()+1);rng=np.random.default_rng(c['bootstrap']['seed']);diff={n:[] for n in prior['groups']}
        for i in range(c['bootstrap']['iterations']):
            weights=np.bincount(rng.integers(g,size=g),minlength=g)[groups].astype(float)
            if (weights*y).sum()==0 or (weights*(1-y)).sum()==0:continue
            rf=cached_ap(y,weights,caches['morgan_rf_3'])
            for n in diff:diff[n].append(rf-cached_ap(y,weights,caches[n]))
            if (i+1)%200==0:status('paired_scaffold_bootstrap',iteration=i+1,iterations=c['bootstrap']['iterations'])
        with (OUT/'predictions.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['compound_key','observed_label',*pred])
            writer.writerows([key,int(y[i]),*[float(v[i]) for v in pred.values()]] for i,key in enumerate(ids))
        # Verify saved artifact against independently loaded source identities and sealed arrays.
        with (OUT/'predictions.csv').open() as f:saved=list(csv.DictReader(f))
        assert [r['compound_key'] for r in saved]==ids
        for name in pred:
            arr=np.array([float(r[name]) for r in saved]);np.testing.assert_array_equal(arr,pred[name])
            assert abs(average_precision_score(y,arr)-measured[name]['average_precision'])<1e-12
            ranking=sorted(range(len(ids)),key=lambda i:(-arr[i],ids[i]))
            for k in c['top_k']:assert sum(int(y[i]) for i in ranking[:k])==measured[name]['top_k'][str(k)]['hits']
        checked()
        result={'complete':True,'design':c['design'],'n':len(y),'hits':int(y.sum()),'rf_members_and_ensemble':measured,
                'comparison_groups':all_metrics,'rf_ensemble_minus_comparator_AP':{n:{'observed':measured['morgan_rf_3']['average_precision']-all_metrics[n]['average_precision'],
                    'percentile_95_interval':np.quantile(d,[.025,.975]).tolist(),'valid_replicates':len(d)} for n,d in diff.items()},
                'protocol_sha256':sha256(PROTOCOL),'predictions_sha256':sha256(OUT/'predictions.csv'),
                'prediction_seconds':prediction_seconds,'total_seconds':time.perf_counter()-start,'runtime':runtime(),
                'verification':'passed: source IDs/labels, sealed member arrays, saved scores, AP, Top-K, original comparators and weighted AP',
                'limits':['Post-hoc comparator addition, not a new untouched test.', 'Fixed existing models; no RF hyperparameter search on Harvard.',
                          'Unknown assay concentration; not actinomycete-specific or new antibiotic discovery.']}
        save(OUT/'results.json',result);save(ROOT/'reports/harvard_morgan_rf_comparison_v1.json',result)
        save(OUT/'status.json',{'state':'complete','completed':len(ids),'total':len(ids),'seconds':result['total_seconds']})
        print(json.dumps(result['comparison_groups'],indent=2),flush=True)
    except Exception as exc:
        save(OUT/'status.json',{'state':'failed','error':str(exc)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','evaluate']);a=p.parse_args()
    freeze() if a.action=='freeze' else evaluate()
