"""Audit the staged matrix against the sealed sources before atomic publication."""
import os
import re
import tempfile
from report import MATRIX,SNAP,GUIDE
from score import *

def normalize(text,location):
    def convert(m):
        path=(location.parent/m[1]).resolve(strict=True)
        assert path.is_relative_to(ROOT) and path.is_file(),path
        return ']('+str(path.relative_to(ROOT))+')'
    return re.sub(r'\]\(([^)]+)\)',convert,text)

def main():
    pending=(EV/'matrix_for_publication.md').read_text()
    snap=SNAP.read_text()
    assert normalize(pending,MATRIX)==normalize(snap,SNAP)
    gate=read(EV/'preflight.json')
    for file,h in gate['hashes'].items():
        assert sha(ROOT/file)==h
    for file,h in manifest(P5/'phase5.sha256').items():
        assert sha(P5/file)==h
    for file in ['phase6_evaluation.md','experimental_setting.md']:
        assert sha(GUIDE/file)==sha(EV/file)
    def cells(key):
        ln=next(line for line in snap.splitlines() if line.startswith('| '+key+' |'))
        return [c.strip() for c in ln.strip('|').split('|')]
    def check_interval(cell,x,dd=5):
        vals=list(map(float,re.findall(r'[+-]?\d+\.\d+',cell)))
        assert len(vals)==3 and np.allclose(vals,[x['mean'],x['lo'],x['hi']],rtol=0,atol=.50001*10**-dd),(cell,x)
    data=read(OUT/'summary.json')
    for m in METHODS:
        row=cells(m)
        for i,k in enumerate(METRICS):
            check_interval(row[i+1],data['metrics'][k][m])
    for v in data['comparisons'][:4]:
        row=cells(v['first']+'−'+v['second'])
        assert int(row[1])==v['n']
        check_interval(row[2],v)
        assert abs(float(row[3])-v['p'])<=.0000051 and abs(float(row[4])-v['holm'])<=.0000051
    cal=read(P5/'base/calibration.json')
    for row in readcsv(P5/'analysis/role_weights.csv'):
        role=row['role']
        cells_r=cells(role.title())
        for k in ['brier','T0','T1','delta_pp']:
            assert math.isclose(float(row[k]),cal['roles'][role][k],abs_tol=1e-12)
        assert int(cells_r[2])==int(row['n_users'])
        assert abs(float(cells_r[1])-cal['full_C_heads'][role+'.pJ']['temperature'])<.0000051
        assert abs(float(cells_r[3])-float(row['brier']))<.0000051
        for i,k in [(4,'T0'),(5,'T1'),(6,'delta_pp')]:
            val=float(row[k])*(100 if k!='delta_pp' else 1)
            assert abs(float(cells_r[i].strip('%'))-val)<.000501
        check_interval(cells_r[7],next(r for r in data['comparisons'] if r['first']=='B4-minus-'+role))
    for row in readcsv(P5/'analysis/correlated_error_pairs.csv'):
        orig=next(p for p in cal['pairs'] if (p['a'],p['b'])==(row['a'],row['b']))
        cells_r=cells(row['a'].title()+' × '+row['b'].title())
        for i,k,pt in [(2,'rho0',orig['0']['rho']),(4,'rho1',orig['1']['rho']),(5,'rho',orig['rho'])]:
            assert abs(float(row[k])-pt)<1e-12
            check_interval(cells_r[i],dict(mean=pt,lo=float(row[k+'_lo']),hi=float(row[k+'_hi'])),4)
        for i,k in [(1,'0'),(3,'1')]:
            assert cells_r[i]==f"{orig[k]['n_users']}/{orig[k]['n_jobs']}"
        assert cells_r[7]==row['bootstrap_degenerate0']+' / '+row['bootstrap_degenerate1']
    ledger=readcsv(OUT/'cost_ledger.csv')
    costs=read(OUT/'cost_summary.json')
    assert math.isclose(sum(float(r['gpu_hours']) for r in ledger if r['scope']=='主线'),costs['main_gpu_hours'])
    assert sum(int(r['first_windows'] or 0) for r in ledger if r['scope']=='主线')==14336
    assert all(Path(r['source']).is_file() for r in ledger)
    assert sum(float(r['gpu_hours']) for r in ledger if r['stage'] in ['Phase 5','Phase 6'])==0
    frows=[r for r in readcsv(OUT/'per_user.csv') if r['method']=='F']
    assert len(frows)==256
    csvout('F_per_user.csv',frows)
    write(EV/'F_receipt.json',dict(frozen_rankings_sha256=gate['hashes'][str((P5/'base/rankings.jsonl').relative_to(ROOT))],F_order_fingerprint=gate['F_orders_sha256'],F_per_user_sha256=sha(OUT/'F_per_user.csv'),summary_sha256=sha(OUT/'summary.json')))
    assert sha(MATRIX)==sha(EV/'acceptance_result_matrix.md') or MATRIX.read_text()==pending
    with tempfile.NamedTemporaryFile('w',dir=MATRIX.parent,delete=False) as f:
        f.write(pending)
        tmp=Path(f.name)
    tmp.chmod(MATRIX.stat().st_mode & 0o777)
    os.replace(tmp,MATRIX)
    assert normalize(MATRIX.read_text(),MATRIX)==normalize(SNAP.read_text(),SNAP)
    write(RUN/'acceptance.json',dict(status='passed',completed_utc=now(),preflight=gate['at'],opened_T=read(EV/'score_receipt.json')['label_open_utc'],counts=data['denominators'],statistics=read(EV/'statistics_verified.json'),label_reads_by_scorer=1,phase5_manifest_files=len(manifest(P5/'phase5.sha256')),role_rows=4,correlation_rows=6,phase6_gpu_h=0,matrix_sha256=sha(MATRIX),snapshot_sha256=sha(SNAP),snapshot_rebased_links=True,independent_checks=['50 method intervals, 11 difference intervals','four raw permutation p-values and Holm family','cost ledger reconciliation','calibration points, intervals, weights, role-removal differences','both result matrices resolve to identical content and working links'],optional='未运行（本次执行至 Phase 6）'))
if __name__=='__main__':
    main()

