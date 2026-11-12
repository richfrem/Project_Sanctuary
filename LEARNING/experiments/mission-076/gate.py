"""MISSION-076 gate G: independent pure-Python implementation of the same process, compared with sim.c
by a two-sample binomial z test at several checkpoints and parameter sets (development seeds)."""
import random, subprocess, math, sys, csv, io
CONFIGS=[(0.6447001,0.0),(0.6447001,0.05),(0.62,0.2),(0.70,0.0)]
RUNS=20000; TMAX=60; NCAP=2000; L=1<<20
CHK=[1,3,7,15,30,60]
def py(p,phi,seed):
    rnd=random.Random(seed); alive={t:0 for t in CHK}
    for _ in range(RUNS):
        cur={0}; t=0; est=False
        while t<TMAX and cur and not est:
            nxt=set()
            for x in cur:
                if rnd.random()<p: nxt.add((x+1)%L)
                if rnd.random()<p: nxt.add((x-1)%L)
                if phi>0 and rnd.random()<phi: nxt.add(int(rnd.random()*L))
            cur=nxt; t+=1
            if t in alive and cur: alive[t]+=1
            if len(cur)>=NCAP: est=True
        if est:
            for c in CHK:
                if c>t: alive[c]+=1
    return alive
def c(p,phi,seed):
    out=subprocess.run(["./sim","--p",str(p),"--phi",str(phi),"--runs",str(RUNS),"--tmax",str(TMAX),"--ncap",str(NCAP),"--L",str(L),"--seed",str(seed)],capture_output=True,text=True,check=True).stdout
    d={}
    for r in csv.DictReader(io.StringIO(out)): d[int(r["t"])]=int(r["alive"])
    return d
worst=0
for i,(p,phi) in enumerate(CONFIGS):
    a=py(p,phi,900+i); b=c(p,phi,800+i)
    for t in CHK:
        if t not in b: continue
        x,y=a[t],b[t]; pool=(x+y)/(2*RUNS); se=math.sqrt(2*RUNS*pool*(1-pool)/RUNS**2*RUNS/RUNS) if 0<pool<1 else 1e-9
        se=math.sqrt(2*pool*(1-pool)/RUNS) if 0<pool<1 else 1e-9
        z=(x/RUNS-y/RUNS)/se; worst=max(worst,abs(z))
        print(f"p={p} phi={phi} t={t}: py={x} c={y} z={z:+.2f}")
print("WORST |z| =",round(worst,2)); sys.exit(0 if worst<4 else 1)
