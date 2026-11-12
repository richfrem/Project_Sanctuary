"""MISSION-074 gate G: independent loop tracer vs C union-find. Walks arcs on explicit
half-integer midpoint coordinates (different from the C index scheme)."""
import subprocess, sys, json, random
def trace(L, bits):
    # tile (i,j) occupies [i,i+1]x[j,j+1]; midpoints in doubled coords: N=(2i,2j+1) S=(2i+2,2j+1) W=(2i+1,2j) E=(2i+1,2j+2), mod 2L
    adj = {}
    def add(a, b):
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    M = 2 * L
    for i in range(L):
        for j in range(L):
            N=((2*i)%M,(2*j+1)%M); S=((2*i+2)%M,(2*j+1)%M); W=((2*i+1)%M,(2*j)%M); E=((2*i+1)%M,(2*j+2)%M)
            if bits[i*L+j]=='0': add(N,W); add(S,E)
            else: add(N,E); add(S,W)
    assert all(len(v)==2 for v in adj.values()) and len(adj)==2*L*L
    seen=set(); out=[]
    for st in adj:
        if st in seen: continue
        n=0; prev=None; cur=st
        while True:
            seen.add(cur); n+=1
            a,b=adj[cur]
            nx = a if (a!=prev or a==b) else b
            if prev is not None and a==prev and b==prev: nx=a
            prev, cur = cur, nx
            if cur==st: break
        out.append(n)
    return sorted(out)
def main():
    rnd=random.Random(74); bad=0; N=200; tot=0
    for k in range(N):
        L=rnd.randint(4,12); p=rnd.choice([0.0,0.2,0.35,0.5,0.8,1.0]); sd=rnd.randint(1,10**6)
        mode=['checker'] if rnd.random()<0.5 else []
        r=subprocess.run(['./tilings','--dump',str(L),str(p),str(sd)]+mode,capture_output=True,text=True,check=True).stdout.split('\n')
        bits=r[0]; cl=sorted(map(int,r[1].split())); pl=trace(L,bits)
        if cl!=pl or sum(pl)!=2*L*L: bad+=1; print('MISMATCH',L,p,sd,cl[:5],pl[:5])
        tot+=1
    print(json.dumps({'gate':'G','tilings':tot,'mismatches':bad,'pass':bad==0}))
    sys.exit(1 if bad else 0)
main()
