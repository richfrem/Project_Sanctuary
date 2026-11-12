"""MISSION-074 frozen analysis. usage: analyze.py heldout_dir  -> prints JSON verdict inputs.
Reads <dir>/{p050,p035,c035}_L{L}.jsonl (output of tilings)."""
import json, math, random, sys, os
LS = [64, 128, 256, 512, 1024]
def load(path): return [json.loads(l) for l in open(path)]
def ols(xs, ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
def slope_max(data, rs=None):
    xs=[];ys=[]
    for L in LS:
        v=[r['max'] for r in data[L]]
        if rs: v=[rs.choice(v) for _ in v]
        xs.append(math.log(L)); ys.append(math.log(sum(v)/len(v)))
    return ols(xs,ys)
def alpha_tail(rows, rs=None):
    r = [rs.choice(rows) for _ in rows] if rs else rows
    xs=[];ys=[]
    for b in range(6):
        m=sum(x['ge'][b] for x in r)/len(r)/ (1024*1024)
        xs.append(math.log(128*2**b)); ys.append(math.log(m))
    return -ols(xs,ys)
def ci(f, B=2000, seed=74):
    rs=random.Random(seed); v=sorted(f(rs) for _ in range(B)); return [v[int(.025*B)], v[int(.975*B)-1]]
def main(d):
    out={}; sums_ok=True; n_rows=0
    D={}
    for tag in ('p050','p035','c035'):
        D[tag]={L:load(os.path.join(d,f'{tag}_L{L}.jsonl')) for L in LS}
        for L in LS:
            for r in D[tag][L]:
                n_rows+=1; sums_ok &= (r['sum']==2*L*L and r['L']==L)
                sums_ok &= all(r['ge'][i]>=r['ge'][i+1] for i in range(5))
    out['sum_rule_and_monotone_ok']=sums_ok; out['rows']=n_rows
    out['samples']={t:{L:len(D[t][L]) for L in LS} for t in D}
    out['mean_max']={t:{L:sum(r['max'] for r in D[t][L])/len(D[t][L]) for L in LS} for t in D}
    h1=slope_max(D['p050']); out['H1']={'slope':h1,'ci95':ci(lambda rs:slope_max(D['p050'],rs)),'interval':[1.72,1.78],'pass':1.72<=h1<=1.78}
    a=alpha_tail(D['p050'][1024]); out['H2']={'alpha':a,'ci95':ci(lambda rs:alpha_tail(D['p050'][1024],rs)),'interval':[1.08,1.20],'pass':1.08<=a<=1.20}
    h3a=slope_max(D['p035']); out['H3a']={'slope':h3a,'ci95':ci(lambda rs:slope_max(D['p035'],rs)),'interval':[1.70,1.80],'pass':1.70<=h3a<=1.80}
    h3b=slope_max(D['c035']); out['H3b']={'slope':h3b,'ci95':ci(lambda rs:slope_max(D['c035'],rs)),'limit_below':1.2,'pass':h3b<1.2}
    out['all_hold']=bool(sums_ok and out['H1']['pass'] and out['H2']['pass'] and out['H3a']['pass'] and out['H3b']['pass'])
    print(json.dumps(out,indent=1))
main(sys.argv[1])
