"""Prepare versioned continuous auxiliary data; quarantine uncertain identities, protect evaluation."""
import collections,gzip,hashlib,json,math,resource,sys,time
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/coadd_auxiliary_v1.json'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def number(raw):
    try:
        x=float(raw);return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None

def main():
    start=time.perf_counter();config=json.loads(CONFIG.read_text());source=ROOT/config['source_audit']
    raw=ROOT/'data/raw/assay_discovery/CO-ADD_InhibitionData_r03_01-02-2020_CSV.zip'
    prior=json.loads((ROOT/'reports/coadd_primary_audit_v1.json').read_text())
    if digest(raw)!=prior['archive_sha256']:raise ValueError('Raw archive changed')
    records=[json.loads(line) for line in gzip.open(source,'rt')]
    if len(records)!=prior['counts']['rows']:raise ValueError('Source audit row count differs')
    for r in records:
        s=r['source']
        if (s['ASSAY_ID'],s['STRAIN'],s['CONC'])!=(config['assay_id'],config['strain'],config['concentration']):raise ValueError('Assay changed')
    identities=collections.defaultdict(list)
    for r in records:
        if r['structure']['structure_eligible']:identities[r['structure']['parent_inchikey']].append(r)
    repeated={k for k,rs in identities.items() if len(rs)>1}
    ambiguous={k for k,rs in identities.items() if len({r['structure']['parent_smiles'] for r in rs})>1}
    protected={k:set() for k in config['excluded_overlap_keys']};protected_hashes={}
    for split in config['wong_excluded_partitions']:
        path=ROOT/f'data/processed/wong_v0/{split}.parquet';protected_hashes[str(path.relative_to(ROOT))]=digest(path)
        for row in pq.read_table(path,columns=list(protected)).to_pylist():
            for key in protected:protected[key].add(row[key])
    # Configuration fixed before filtering; no activity outcomes choose the exclusions or split.
    dest=ROOT/'data/processed/coadd_auxiliary_v1';dest.mkdir(parents=True,exist_ok=False)
    (dest/'config.json').write_bytes(CONFIG.read_bytes())
    reasons=collections.Counter();splits={s:[] for s in ('train','validation')}
    review=gzip.open(dest/'review_records.jsonl.gz','wt')
    try:
        for r in records:
            st,s=r['structure'],r['source'];key=st.get('parent_inchikey');reason=None
            value,sd,n=number(s['INHIB_AVE']),number(s['INHIB_STD']),number(s['NASSAYS'])
            if not st['structure_eligible']:reason='structure_review'
            elif key in ambiguous:reason='ambiguous_identity'
            elif key in repeated:reason='duplicate_identity_quarantined'
            elif value is None or sd is None or sd<0 or n is None or n<2 or n!=int(n):reason='measurement_summary_qc'
            fields={'compound_key':key,**{k:st.get(k) for k in ('parent_smiles','connectivity_smiles','scaffold')}}
            overlap=[k for k,values in protected.items() if fields[k] in values]
            if reason is None and overlap:reason='wong_evaluation_structure_overlap'
            if reason:
                reasons[reason]+=1
                review.write(json.dumps({'reason':reason,'source_row':r['source_row'],'source_id':s['COADD_ID'],'compound_key':key,'protected_overlap_fields':overlap})+'\n');continue
            h=hashlib.sha256(f"{config['auxiliary_split_seed']}:{st['scaffold']}".encode()).hexdigest()
            split='validation' if int(h[:16],16)/2**64<config['auxiliary_validation_hash_fraction'] else 'train'
            splits[split].append({**fields,'source_row':r['source_row'],'source_id':s['COADD_ID'],'assay_id':s['ASSAY_ID'],'strain':s['STRAIN'],'concentration':s['CONC'],'inhibition_percent':value,'inhibition_sd':sd,'n_assays':int(n),'out_of_range_inhibition':not 0<=value<=100,'np_status':'unknown','actinomycete_status':'unknown'})
    finally:review.close()
    if any(not rows for rows in splits.values()):raise ValueError('Empty auxiliary partition')
    for field in protected:
        train={r[field] for r in splits['train']};val={r[field] for r in splits['validation']}
        if train&val or train&protected[field] or val&protected[field]:raise ValueError(f'Leakage in {field}')
    for split,rows in splits.items():
        with gzip.open(dest/f'{split}.jsonl.gz','wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    accepted=sum(map(len,splits.values()))
    if accepted+sum(reasons.values())!=len(records):raise ValueError('Accounting mismatch')
    report={'complete':True,'source_rows':len(records),'accepted_compounds':accepted,'excluded_rows_by_first_reason':dict(reasons),'split_counts':{k:len(v) for k,v in splits.items()},'retained_out_of_range_rows':sum(r['out_of_range_inhibition'] for rows in splits.values() for r in rows),'response_range':{k:[min(r['inhibition_percent'] for r in v),max(r['inhibition_percent'] for r in v)] for k,v in splits.items()},'protected_wong_partitions':config['wong_excluded_partitions'],'structure_overlap_checks_passed':True,'wong_activity_columns_read':False,'continuous_auxiliary_targets_prepared':True,'author_binary_labels_reconstructed':False,'model_training_started':False,'seconds':time.perf_counter()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),'config_sha256':digest(CONFIG),'source_audit_sha256':digest(source),'source_archive_sha256':digest(raw),'protected_files_sha256':protected_hashes,'script_sha256':digest(Path(__file__)),'output_sha256':{p.name:digest(p) for p in dest.iterdir() if p.is_file()},'limitations':['Conservative duplicate exclusion loses some valid repeated measurements; review records retained.','Auxiliary response is continuous inhibition, not MIC or an author-defined hit label.','Out-of-range responses retained; robust loss and scaling must be specified before training.','No plate-level QC reconstruction, independent generalization evidence or actinomycete-specific validation.','Publication or redistribution terms remain to be confirmed; raw data remain local.']}
    (dest/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports/coadd_auxiliary_preparation_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
