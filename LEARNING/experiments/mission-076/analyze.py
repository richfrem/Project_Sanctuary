"""MISSION-076 analysis. Usage: analyze.py DIR   (DIR has base.csv, lr_<phi>.csv, nn_<phi>.csv)
t*(config) = first t where P_config(t)/P_base(t) >= RATIO (log-log interpolation between checkpoints).
Exponent a = -slope of log t* vs log phi (OLS). Bootstrap over batches (resampling batches, seed 76)."""
import csv, glob, math, os, sys, json, random
RATIO=2.0; WIN=(100,1000); PC=0.6447001
def load(f):
    d={}
    for r in csv.DictReader(open(f)):
        b=int(r["batch"]); d.setdefault(b,{})[int(r["t"])]=(int(r["alive"]),int(r["runs"]))
    return d
def pool(d,batches):
    ts=sorted(next(iter(d.values())))
    return {t:sum(d[b][t][0] for b in batches)/sum(d[b][t][1] for b in batches) for t in ts}
def slope(xs,ys):
    n=len(xs);mx=sum(xs)/n;my=sum(ys)/n
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
def delta(P):
    pts=[(math.log(t),math.log(v)) for t,v in P.items() if WIN[0]<=t<=WIN[1] and v>0]
    return -slope([a for a,_ in pts],[b for _,b in pts])
def tstar(P,P0):
    prev=None
    for t in sorted(P):
        if P0.get(t,0)<=0: continue
        r=P[t]/P0[t]
        if r>=RATIO:
            if prev is None: return float(t)
            (t0,r0)=prev
            f=(math.log(RATIO)-math.log(r0))/(math.log(r)-math.log(r0)) if r>r0 else 1
            return math.exp(math.log(t0)+f*(math.log(t)-math.log(t0)))
        prev=(t,r)
    return None   # censored
def fit(phis,ts):
    pts=[(math.log(p),math.log(t)) for p,t in zip(phis,ts) if t]
    return -slope([a for a,_ in pts],[b for _,b in pts]) if len(pts)>=3 else None
def run(dr):
    base=load(os.path.join(dr,"base.csv")); B=sorted(base)
    cfgs={}
    for kind in("lr","nn"):
        for f in sorted(glob.glob(os.path.join(dr,f"{kind}_*.csv"))):
            cfgs.setdefault(kind,{})[float(os.path.basename(f)[len(kind)+1:-4])]=load(f)
    rng=random.Random(76); out={"delta_base":None,"boot":None}
    P0=pool(base,B); out["delta_base"]=delta(P0)
    res={}
    for kind,dd in cfgs.items():
        res[kind]={}
        for phi,d in dd.items():
            res[kind][phi]=tstar(pool(d,sorted(d)),P0)
    out["tstar"]={k:{str(p):v for p,v in sorted(r.items())} for k,r in res.items()}
    out["a"]={k:fit(sorted(r),[r[p] for p in sorted(r)]) for k,r in res.items()}
    # bootstrap over batches
    bs={"delta":[], "lr":[], "nn":[]}
    for _ in range(300):
        bb=[rng.choice(B) for _ in B]; P0b=pool(base,bb); bs["delta"].append(delta(P0b))
        for kind,dd in cfgs.items():
            phis=sorted(dd); ts=[tstar(pool(dd[p],[rng.choice(sorted(dd[p])) for _ in dd[p]]),P0b) for p in phis]
            a=fit(phis,ts)
            if a is not None: bs[kind].append(a)
    def ci(v): 
        v=sorted(v); return (v[int(.025*len(v))],v[int(.975*len(v))]) if v else None
    out["boot"]={k:ci(v) for k,v in bs.items()}
    print(json.dumps(out,indent=1))
    return out
if __name__=="__main__": run(sys.argv[1])
