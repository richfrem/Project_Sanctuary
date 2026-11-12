"""Exploratory, unfrozen: render a random Truchet tiling (L=40, p=0.5, seed 74) with its largest loops coloured."""
import subprocess, math
L=40
bits=subprocess.run(['./tilings','--dump',str(L),'0.5','74'],capture_output=True,text=True,check=True).stdout.split('\n')[0]
M=2*L; adj={}; arc={}
def add(a,b,seg):
    adj.setdefault(a,[]).append(b); adj.setdefault(b,[]).append(a); arc[(a,b)]=arc[(b,a)]=seg
for i in range(L):
    for j in range(L):
        N=(2*i%M,(2*j+1)%M); S=((2*i+2)%M,(2*j+1)%M); W=((2*i+1)%M,2*j%M); E=((2*i+1)%M,(2*j+2)%M)
        if bits[i*L+j]=='0': add(N,W,(i,j,'NW')); add(S,E,(i,j,'SE'))
        else: add(N,E,(i,j,'NE')); add(S,W,(i,j,'SW'))
comp={}; loops=[]
for st in adj:
    if st in comp: continue
    cyc=[];prev=None;cur=st
    while True:
        comp[cur]=len(loops);cyc.append(cur);a,b=adj[cur];nx=a if a!=prev else b;prev,cur=cur,nx
        if cur==st:break
    loops.append(cyc)
order=sorted(range(len(loops)),key=lambda k:-len(loops[k]))
colors=['#d1495b','#edae49','#00798c','#30638e','#66a182','#8d5a97']
rank={k:r for r,k in enumerate(order[:6])}
S=14; paths=[[] for _ in range(7)]
def arcpath(i,j,c):
    x,y=j*S,i*S; r=S/2
    pts={'NW':((x+r,y),(x,y+r),0),'NE':((x+r,y),(x+S,y+r),1),'SW':((x+r,y+S),(x,y+r),1),'SE':((x+r,y+S),(x+S,y+r),0)}[c]
    (a,b,sw)=pts; return f"M{a[0]},{a[1]}A{r},{r} 0 0 {sw} {b[0]},{b[1]}"
for (a,b),seg in arc.items():
    if a<b:
        k=comp[a]; idx=rank.get(k,6); paths[idx].append(arcpath(*seg))
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {L*S} {L*S}" width="{L*S}" height="{L*S}"><rect width="100%" height="100%" fill="#faf8f3"/>']
for idx in (6,0,1,2,3,4,5):
    col='#b8b4aa' if idx==6 else colors[idx]; w=1.2 if idx==6 else 2.4
    svg.append(f'<path d="{" ".join(paths[idx])}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linecap="round"/>')
svg.append('</svg>')
open('truchet_L40.svg','w').write('\n'.join(svg))
print([len(loops[k]) for k in order[:6]],len(loops))
