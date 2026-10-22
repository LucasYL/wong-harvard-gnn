"""Independently inspect saved partitions, inventory arithmetic, and exposure checks."""
import gzip
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/processed/synthemol_dataset_v1'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]


def main():
    manifest=json.loads((DEST/'manifest.json').read_text())
    for name,h in manifest['output_hashes'].items():assert sha(DEST/name)==h
    for name,h in manifest['input_hashes'].items():assert sha(ROOT/name)==h
    assert sha(ROOT/'scripts/prepare_synthemol_dataset_v1.py')==manifest['script_sha256']
    fields=['compound_key','parent_smiles','connectivity_smiles','scaffold']
    new={s:read(DEST/f'{s}.jsonl.gz') for s in ('train','validation','external')}
    old={f'wong_{s}':pq.read_table(ROOT/f'data/processed/wong_v0/{s}.parquet',columns=fields).to_pylist() for s in ('train','validation','test')}
    old.update({f'coadd_{s}':read(ROOT/f'data/processed/coadd_auxiliary_v1/{s}.jsonl.gz') for s in ('train','validation')})
    checks=[]
    for s,rows in new.items():
        assert len(rows)==len({r['compound_key'] for r in rows})==manifest['partitions'][s]['compounds']
        assert all(r['split']==s and r['raw_sheet_rows'] for r in rows)
        assert sum(r['activity'] for r in rows)==manifest['partitions'][s]['author_hits']
        compare={**{f'new_{t}':rs for t,rs in new.items() if t!=s},**old}
        for name,other in compare.items():
            if s=='train' and name in ('wong_train','coadd_train'):continue
            for field in fields:
                shared={r[field] for r in rows}&{r[field] for r in other}
                assert not shared,(s,name,field,len(shared))
                checks.append({'partition':s,'against':name,'field':field,'overlap':0})
    train={'wong':old['wong_train'],'coadd':old['coadd_train'],'synthemol':new['train']}
    inventory={}
    for names in [('wong','coadd'),('wong','synthemol'),('wong','coadd','synthemol')]:
        all_rows=[r for n in names for r in train[n]]
        result={'compound_assay_records':len(all_rows),'unique_parent_inchikeys':len({r['compound_key'] for r in all_rows})}
        name='_plus_'.join(names);assert result==manifest['combined_training_inventory'][name];inventory[name]=result
    assigned={r['source_row'] for rs in new.values() for r in rs}
    review=read(DEST/'review.jsonl.gz');excluded={r['source_row'] for r in review}
    assert not assigned&excluded
    assert len(assigned)+len(excluded)==manifest['source_author_rows']==10658
    result={'verified':True,'dataset_manifest_sha256':sha(DEST/'manifest.json'),'zero_overlap_checks':checks,
            'training_inventory':inventory,'source_accounting':{'assigned':len(assigned),'excluded':len(excluded)},
            'external_predictions_created':False,'old_wong_activity_labels_read':False}
    (ROOT/'reports/synthemol_dataset_verification_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified':True,'zero_overlap_checks':len(checks),'inventory':inventory,'source_accounting':result['source_accounting']},indent=2))


if __name__=='__main__':main()
