"""Join measurement leads to structures/producers and audit declared data exposure.

No model predictions, new labels, averaging across assays, or split assignment.
"""
import collections
import csv
import gzip
import json
import math
import time
from pathlib import Path
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from antibacterial.chemistry import standardize
from antibacterial.common import sha256, runtime

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/actinomycete_discovery'
DEST = ROOT / 'data/processed/npass_domain_v1'
KEYS = ['compound_key', 'parent_smiles', 'connectivity_smiles', 'scaffold']


def read_tsv(path):
    with path.open() as f:
        for i, row in enumerate(csv.DictReader(f, delimiter='\t'), 1):
            assert None not in row and all(v is not None for v in row.values()), (path, i)
            yield i, row


def write_jsonl(path, rows):
    with gzip.open(path, 'wt') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def protected_sets():
    output, hashes = {}, {}
    for folder, splits, ext, hashfield in [
        ('wong_v0', ['train', 'validation', 'test'], 'parquet', 'output_hashes'),
        ('coadd_auxiliary_v1', ['train', 'validation'], 'jsonl.gz', 'output_sha256'),
        ('synthemol_dataset_v1', ['train', 'validation', 'external'], 'jsonl.gz', 'output_hashes')]:
        manifest = json.loads((ROOT / 'data/processed' / folder / 'manifest.json').read_text())
        for split in splits:
            p = ROOT / 'data/processed' / folder / f'{split}.{ext}'
            hashes[str(p.relative_to(ROOT))] = sha256(p)
            assert hashes[str(p.relative_to(ROOT))] == manifest[hashfield][p.name]
            if ext == 'parquet':
                rows = pq.read_table(p, columns=KEYS).to_pylist()
            else:
                with gzip.open(p, 'rt') as f:
                    rows = [{k: r[k] for k in KEYS} for r in map(json.loads, f)]
            output[f'{folder}/{split}'] = {k: {r[k] for r in rows} for k in KEYS}
    return output, hashes


def main():
    start = time.perf_counter()
    RDLogger.DisableLog('rdApp.warning')
    DEST.mkdir(exist_ok=True)
    download = json.loads((RAW / 'npass_join_downloads_v1.json').read_text())
    for name, meta in download.items():
        assert (RAW / name).stat().st_size == meta['bytes'] and sha256(RAW / name) == meta['sha256']
    small = json.loads((RAW / 'npass_small_files_manifest_v1.json').read_text())
    for name, meta in small.items():
        assert sha256(RAW / name) == meta['sha256']
    activity_path = ROOT / 'data/processed/npass_inventory_v1/s_aureus_organism_records.jsonl.gz'
    inventory = json.loads((ROOT / 'reports/npass_activity_inventory_v1.json').read_text())
    assert sha256(activity_path) == inventory['output_sha256']
    with gzip.open(activity_path, 'rt') as f:
        activities = list(map(json.loads, f))
    ids = {r['np_id'] for r in activities}
    assert len(ids) == 4394 and len(activities) == 34972
    rn_ids = {r['np_id'] for r in activities if r['target_id'] == 'NPT1101'}
    structures = collections.defaultdict(list)
    for i, row in read_tsv(RAW / 'NPASS3.0_naturalproducts_structure.txt'):
        if row['np_id'] in ids:
            structures[row['np_id']].append({'source_record': i, **row})
    source_structure_rows = i
    species = {}
    for _, row in read_tsv(RAW / 'NPASS3.0_species_info.txt'):
        assert row['org_id'] not in species
        species[row['org_id']] = row
    origins = collections.defaultdict(list)
    for i, row in read_tsv(RAW / 'NPASS3.0_naturalproducts_species_pair.txt'):
        if row['np_id'] in ids:
            origins[row['np_id']].append({'source_record': i, **row, 'producer': species.get(row['org_id'])})
    source_origin_rows = i
    atlas = collections.defaultdict(list)
    npatlas_path = RAW / 'NPAtlas_download_v2024_09.tsv'
    assert sha256(npatlas_path) == json.loads((RAW / 'npatlas_manifest_v2024_09.json').read_text())['sha256']
    for _, row in read_tsv(npatlas_path):
        atlas[row['compound_inchikey']].append({k: row[k] for k in (
            'npaid', 'compound_name', 'genus', 'origin_species', 'origin_type', 'original_reference_doi', 'npatlas_url')})
    protected, protected_hashes = protected_sets()
    compounds = []
    for i, np_id in enumerate(sorted(ids), 1):
        candidates = structures[np_id]
        r = {'np_id': np_id, 'has_rn4220_measurement': np_id in rn_ids,
             'structure_source_records': candidates, 'producer_links': origins[np_id],
             'intended_use': 'evidence_lead_pending_assay_and_origin_review'}
        if len(candidates) != 1:
            r['structure'] = {'structure_eligible': False, 'structure_flags': ['missing_or_nonunique_structure_id']}
            r['npatlas_exact_raw_key_matches'] = []
        else:
            raw = candidates[0]
            r['structure'] = standardize(raw['SMILES'])
            mol = Chem.MolFromSmiles(raw['SMILES'])
            computed = Chem.MolToInchiKey(mol) if mol is not None else None
            r['published_raw_key_matches_smiles'] = computed == raw['InChIKey']
            if not r['published_raw_key_matches_smiles']:
                r['structure']['structure_flags'].append('published_key_smiles_mismatch')
                r['structure']['structure_eligible'] = False
            r['npatlas_exact_raw_key_matches'] = atlas.get(computed, []) if computed else []
        st = r['structure']
        values = {**st, 'compound_key': st.get('parent_inchikey')}
        r['overlap'] = {split: {k: values.get(k) in sets[k] for k in KEYS} for split, sets in protected.items()}
        r['no_declared_training_identity_or_connectivity_overlap'] = not any(
            any(v[k] for k in KEYS[:3]) for name, v in r['overlap'].items() if name.endswith('/train'))
        r['no_declared_training_scaffold_overlap'] = not any(
            any(v.values()) for name, v in r['overlap'].items() if name.endswith('/train'))
        r['no_any_declared_partition_overlap'] = not any(any(v.values()) for v in r['overlap'].values())
        r['npass_streptomyces_link'] = any(x['producer'] and x['producer']['genus_name'] == 'Streptomyces' for x in r['producer_links'])
        r['npatlas_streptomyces_link'] = any(x['genus'] == 'Streptomyces' for x in r['npatlas_exact_raw_key_matches'])
        r['streptomyces_origin_lead'] = r['npass_streptomyces_link'] or r['npatlas_streptomyces_link']
        r['origin_status'] = 'database_link_requires_original_evidence_review' if r['producer_links'] or r['npatlas_exact_raw_key_matches'] else 'unknown'
        compounds.append(r)
        if i % 500 == 0:
            print(f'Structures standardized and linked {i}/{len(ids)}', flush=True)
    index = {r['np_id']: r for r in compounds}
    # Exact equality of original measurement columns is only a duplicate flag, not license to average.
    original_columns = [k for k in activities[0] if k not in ('source_record', 'target_name', 'intended_use')]
    exact = collections.Counter(tuple(r[k] for k in original_columns) for r in activities)
    ambiguous = collections.defaultdict(set)
    refs = collections.defaultdict(list)
    for r in activities:
        sig = tuple(r[k] for k in ('np_id', 'target_id', 'ref_id_type', 'ref_id', 'activity_type', 'activity_units', 'assay_organism', 'assay_strain', 'assay_tissue', 'assay_cell_type'))
        ambiguous[sig].add((r['activity_relation'], r['activity_value']))
        refs[(r['ref_id_type'], r['ref_id'])].append(r)
    reference_inventory = []
    for (kind, ref), rows in refs.items():
        unique_ids = {r['np_id'] for r in rows}
        leads = [index[i] for i in unique_ids if index[i]['structure']['structure_eligible'] and index[i]['streptomyces_origin_lead']]
        strict = [r['np_id'] for r in leads if r['no_any_declared_partition_overlap']]
        reference_inventory.append({'ref_id_type': kind, 'ref_id': ref, 'records': len(rows),
            'np_ids': sorted(unique_ids), 'rn4220_np_ids': sorted({r['np_id'] for r in rows if r['target_id'] == 'NPT1101'}),
            'streptomyces_eligible_np_ids': sorted(r['np_id'] for r in leads),
            'streptomyces_no_any_partition_overlap_np_ids': sorted(strict),
            'activity_types': dict(collections.Counter(r['activity_type'] for r in rows)),
            'review_status': 'not_an_assay_id; original_protocol_and_strain_variants_not_reconstructed'})
    write_jsonl(DEST / 'compound_evidence.jsonl.gz', compounds)
    write_jsonl(DEST / 'reference_inventory.jsonl.gz', sorted(reference_inventory, key=lambda r: (-len(r['streptomyces_no_any_partition_overlap_np_ids']), r['ref_id'])))
    write_jsonl(DEST / 'rn4220_compound_evidence.jsonl.gz', [r for r in compounds if r['has_rn4220_measurement']])
    write_jsonl(DEST / 'measurement_context_ambiguities.jsonl.gz', [{'context_fields': list(k), 'distinct_relation_value_pairs': sorted(v)} for k, v in ambiguous.items() if len(v) > 1])
    genus = {}
    for row in compounds:
        for link in row['producer_links']:
            p = link['producer']
            if p and p['superkingdom_name'] == 'Bacteria' and p['genus_tax_id'].isdigit():
                genus[p['genus_tax_id']] = {'tax_id': p['genus_tax_id'], 'genus_name': p['genus_name'], 'family_name': p['family_name']}
    (DEST / 'bacterial_genus_taxonomy_queue.json').write_text(json.dumps(list(genus.values()), indent=2) + '\n')
    def profile(rows):
        eligible = [r for r in rows if r['structure']['structure_eligible']]
        strep = [r for r in eligible if r['streptomyces_origin_lead']]
        return {'np_ids': len(rows), 'eligible_structure_ids': len(eligible),
            'unique_eligible_parent_keys': len({r['structure']['parent_inchikey'] for r in eligible}),
            'with_npass_producer_links': sum(bool(r['producer_links']) for r in rows),
            'with_exact_raw_key_npatlas_links': sum(bool(r['npatlas_exact_raw_key_matches']) for r in rows),
            'eligible_streptomyces_origin_lead_ids': len(strep),
            'strep_no_training_identity_or_connectivity_overlap_ids': sum(r['no_declared_training_identity_or_connectivity_overlap'] for r in strep),
            'strep_no_training_scaffold_overlap_ids': sum(r['no_declared_training_scaffold_overlap'] for r in strep),
            'strep_no_any_declared_partition_overlap_ids': sum(r['no_any_declared_partition_overlap'] for r in strep),
            'all_eligible_no_any_partition_overlap_ids': sum(r['no_any_declared_partition_overlap'] for r in eligible)}
    report = {'complete': True, 'source_structure_rows': source_structure_rows, 'source_producer_link_rows': source_origin_rows,
        'all_s_aureus': profile(compounds), 'rn4220': profile([r for r in compounds if r['has_rn4220_measurement']]),
        'structure_flags': dict(collections.Counter(f for r in compounds for f in r['structure']['structure_flags'])),
        'producer_links_retained': sum(len(r['producer_links']) for r in compounds),
        'producer_link_orphans': sum(x['producer'] is None for r in compounds for x in r['producer_links']),
        'measurement_exact_duplicate_excess': sum(v - 1 for v in exact.values()),
        'same_recorded_context_multiple_values_groups': sum(len(v) > 1 for v in ambiguous.values()),
        'bacterial_genus_taxonomy_queue': len(genus),
        'protected_file_hashes': protected_hashes, 'input_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in [activity_path, npatlas_path, RAW/'NPASS3.0_species_info.txt', *[RAW/n for n in download]]},
        'output_hashes': {p.name: sha256(p) for p in DEST.iterdir() if p.is_file()},
        'script_sha256': sha256(Path(__file__)), 'chemistry_code_sha256': sha256(ROOT/'src/antibacterial/chemistry.py'),
        'seconds': time.perf_counter() - start, 'runtime': runtime(),
        'predictions_generated': False, 'new_binary_labels_assigned': False,
        'limitations': ['Streptomyces is only a narrow genus lead, not the complete actinomycete group.',
            'Producer links may be database or secondary evidence; no original isolation adjudication yet.',
            'NPAtlas matches are exact published raw InChIKeys only; no salt or stereochemistry rescue.',
            'Same recorded context with multiple values signals missing conditions or duplicates, not proven measurement disagreement.',
            'Reference is not assay ID. Missing medium/time/strain-variant fields prevent clean pooling.',
            'No-overlap flags include ACYCLIC and declared data only, not unreviewed pretraining data.']}
    (ROOT/'reports/npass_domain_linkage_v1.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('all_s_aureus', 'rn4220', 'structure_flags', 'producer_link_orphans', 'bacterial_genus_taxonomy_queue', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
