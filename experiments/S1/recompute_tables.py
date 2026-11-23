"""Recompute the IDD-AW tables from the distributed native confusion matrices.

Usage after extraction: python recompute_tables.py
Or pass the path to Supplementary Data S1.zip. Requires Python 3 and NumPy.
"""
from pathlib import Path
from io import BytesIO
import hashlib, json, sys, zipfile
import numpy as np

def verify(read):
    def js(name):return json.loads(read(name))
    def packet(name):
        with np.load(BytesIO(read(name)),allow_pickle=False) as f:return {k:f[k] for k in f.files}
    def statistics(counts,groups):
        classes=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,17,18]
        group_ids=np.unique(groups)
        total=np.stack([counts[groups==g].sum(0) for g in group_ids])
        support=total.sum(-1)
        if not np.all(support==support[:,:,:1,:1]):raise ValueError('Different labels across methods or seeds')
        hit=np.diagonal(total,axis1=-2,axis2=-1);union=total.sum(-2)+total.sum(-1)-hit
        sufficient=np.stack((hit[...,classes],union[...,classes]),axis=-1)
        def measure(m):
            x=np.tensordot(m,sufficient,axes=([-1],[0]))
            h,u=x[...,0],x[...,1]
            return np.divide(100.*h,u,out=np.zeros_like(h,dtype=float),where=u>0).mean(axis=(-1,-2))
        n=len(group_ids)
        draws=np.random.default_rng(100481).multinomial(n,np.full(n,1/n),size=5000)
        point=measure(np.ones(n,dtype=np.int64))
        boot=np.concatenate([measure(chunk) for chunk in np.array_split(draws,80)])
        return point,boot
    def check(g,report,mode,other,point,boot):
        out={}
        for k,metric in enumerate(('full','boundary')):
            base=point[k,0];control=point[k,other];delta=base-control
            ci=np.quantile(boot[:,k,0]-boot[:,k,other],[.025,.975])
            m=report[metric]
            expected=[m[key] for key in (('raw','control','delta_pp') if mode=='guide' else ('ours','reference','delta'))]
            expected+=m['paired_ci95_pp' if mode=='guide' else 'ci95']
            np.testing.assert_allclose([base,control,delta,*ci],expected,rtol=0,atol=1e-10,err_msg=g)
            out[metric]=[float(base),float(control),float(delta),*[float(x) for x in ci]]
        return out
    for name,expected in js('SHA256.json').items():
        if hashlib.sha256(read(name)).hexdigest()!=expected:raise ValueError('Changed entry: '+name)
    summary=js('idd/spatial-summary.json')['cohorts'];operators=js('idd/operator-comparisons.json')['metrics']
    outputs={};ids={};mixed=None;registered=js('tables/printed.json')
    def cell(stat):
        return (f'{stat[2]:+.3f}\n[{stat[3]:+.3f}, {stat[4]:+.3f}]').replace('-','−')
    mfmt=lambda x:f'{x:.3f}'
    table2={}
    for cohort in ('fog','mixed'):
        f=packet('idd/'+cohort+'-counts.npz')
        ids[cohort]=set(f['groups']); case=summary[cohort]
        assert list(f['seeds'])==[101,211,307] and len(f['ids'])==case['frames'] and len(ids[cohort])==case['drives']
        assert np.all(f['counts'][:,1]<=f['counts'][:,0])
        point,boot=statistics(f['counts'],f['groups'])
        for i,name in enumerate(f['arms'].tolist()[1:],1):
            vals=check(cohort+'/'+name,case['contrasts']['raw_minus_'+name],'guide',i,point,boot)
            table2[cohort,name]=[cell(vals[m]) for m in ('boundary','full')]
        if cohort=='mixed':mixed=f
    assert not (ids['fog'] & ids['mixed'])
    for row,name in enumerate(('author','flat','shift','dual'),1):
        assert registered['2'][row][1:]==table2['fog',name]+table2['mixed',name]
    ops=packet('idd/published-operator-counts.npz')
    assert ops['ids'].tolist()==mixed['ids'].tolist()
    merged=np.concatenate((mixed['counts'][:,:,0:1],ops['counts']),axis=2)
    point,boot=statistics(merged,mixed['groups'])
    for i,key in enumerate(ops['arms'].tolist(),1):
        vals=check(key,operators[key],'operator',i,point,boot)
        b,f=vals['boundary'],vals['full']
        assert registered['3'][i][1:]==[mfmt(b[1]),cell(b).replace('\n',' '),mfmt(f[1]),cell(f).replace('\n',' ')]
    print('Verified all paired differences and drive intervals in current Tables 3 and 4 for the 24 fog and 121 mixed-weather drives.')
if __name__=='__main__':
    if len(sys.argv)>1:
        with zipfile.ZipFile(sys.argv[1]) as z:verify(z.read)
    else:
        root=Path(__file__).resolve().parent
        verify(lambda name:(root/name).read_bytes())
