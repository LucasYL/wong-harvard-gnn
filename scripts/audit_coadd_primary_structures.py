"""Audit the dominant single-dose CO-ADD assay without assigning activity labels."""
import csv,gzip,hashlib,io,json,math,resource,sys,time,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import pyarrow.parquet as pq
from rdkit import RDLogger
from antibacterial.chemistry import standardize
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'data/raw/assay_discovery/CO-ADD_InhibitionData_r03_01-02-2020_CSV.zip'
EXPECTED='bd57000d373e1176940bfe9462bdaa21982612de776945dec167fa9ddcf864d2'
def number(s):
    try:
        v=float(s);return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None

def main():
    start=time.perf_counter();RDLogger.DisableLog('rdApp.warning')
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==EXPECTED
    overlap={};source_identity=hashlib.sha256((ROOT/'src/antibacterial/chemistry.py').read_bytes()).hexdigest()
    for split in ('train','validation','test'):
        rows=pq.read_table(ROOT/f'data/processed/wong_v0/{split}.parquet',columns=['compound_key','connectivity_smiles','scaffold']).to_pylist()
        overlap[split]={k:{r[k] for r in rows} for k in ('compound_key','connectivity_smiles','scaffold')}
    dest=ROOT/'data/processed/coadd_primary_audit_v1';dest.mkdir(parents=True,exist_ok=True)
    counts=Counter();flags=Counter();ids=set();parents=defaultdict(lambda:{'rows':0,'smiles':set(),'raw_values':set()});matched={s:Counter() for s in overlap}
    output=dest/'gp020_32ugml_records.jsonl.gz'
    with gzip.open(output,'wt',encoding='utf-8') as out,zipfile.ZipFile(ARCHIVE) as z:
        if len(z.infolist())!=1:raise ValueError('Unexpected archive layout')
        with z.open(z.namelist()[0]) as f:
            for line,r in enumerate(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig')),2):
                if None in r or any(v is None for v in r.values()):raise ValueError('Malformed CSV')
                if (r['ORGANISM'],r['ASSAY_ID'],r['STRAIN'],r['CONC'])!=('Staphylococcus aureus','GP_020','ATCC 43300; MRSA','32 ug/mL'):continue
                counts['rows']+=1;ids.add(r['COADD_ID']);st=standardize(r['SMILES']);flags.update(st['structure_flags'])
                ave,sd,n=number(r['INHIB_AVE']),number(r['INHIB_STD']),number(r['NASSAYS'])
                qc=[]
                if ave is None:qc.append('missing_nonfinite_inhibition')
                if sd is None or sd<0:qc.append('missing_or_invalid_standard_deviation')
                if n is None or n<2 or n!=int(n):qc.append('fewer_than_two_or_invalid_replicate_count')
                if ave is not None and not 0<=ave<=100:counts['out_of_range_retained']+=1
                counts.update(qc);counts['structure_eligible_rows']+=st['structure_eligible']
                matches={s:{'inchikey':st.get('parent_inchikey') in vals['compound_key'],'connectivity':st.get('connectivity_smiles') in vals['connectivity_smiles'],'scaffold':st.get('scaffold') not in (None,'ACYCLIC') and st.get('scaffold') in vals['scaffold']} for s,vals in overlap.items()}
                for s,v in matches.items():matched[s].update({k:int(x) for k,x in v.items()})
                if st['structure_eligible']:
                    g=parents[st['parent_inchikey']];g['rows']+=1;g['smiles'].add(st['parent_smiles']);g['raw_values'].add(r['INHIB_AVE'])
                if not qc and st['structure_eligible']:counts['structure_and_summary_qc_pass_rows']+=1
                out.write(json.dumps({'source_row':line,'source':r,'structure':st,'summary_qc_flags':qc,'wong_overlap':matches},ensure_ascii=False)+'\n')
                if counts['rows']%10000==0:print(json.dumps({'processed':counts['rows'],'seconds':time.perf_counter()-start}),flush=True)
    if counts['rows']!=76887:raise ValueError('Row count differs from source inventory')
    report={'source_archive':ARCHIVE.name,'archive_sha256':EXPECTED,'filter':{'organism':'Staphylococcus aureus','assay':'GP_020','strain':'ATCC 43300; MRSA','concentration':'32 ug/mL'},'counts':dict(counts),'distinct_source_ids':len(ids),'structure_flags':dict(flags),'unique_eligible_inchikeys':len(parents),'duplicate_identity_groups':sum(g['rows']>1 for g in parents.values()),'duplicate_identity_groups_with_different_inhibition':sum(len(g['raw_values'])>1 for g in parents.values()),'same_inchikey_multiple_parent_smiles_groups':sum(len(g['smiles'])>1 for g in parents.values()),'wong_overlap_rows':{k:dict(v) for k,v in matched.items()},'new_training_labels_created':False,'wong_activity_labels_read':False,'source_standardizer_sha256':source_identity,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'output_bytes':output.stat().st_size,'seconds':time.perf_counter()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),'limitations':['Summary QC is not equivalent to author-defined hit status or plate-level QC.','Duplicate identity groups and different parent forms require adjudication before modeling.','Overlap counts are source rows including structures requiring review, not final eligible unique molecules.','Negative and above-100 inhibition retained without clipping.','No new independent test, standardized target labels, license clearance, or origin annotations established.']}
    assert source_identity==hashlib.sha256((ROOT/'src/antibacterial/chemistry.py').read_bytes()).hexdigest()
    (ROOT/'reports/coadd_primary_audit_v1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
