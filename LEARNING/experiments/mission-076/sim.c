/* MISSION-076: bond directed percolation (1+1) on a ring with optional long-range
 * "cross-reference" links. Each active site x at generation t activates x-1 and x+1
 * at t+1 independently with prob p (nearest-neighbour recruitment) and, with prob phi,
 * one uniformly random site anywhere on the ring (long-range recruitment).
 * Start: one active site. A run is "established" (counted alive forever) once the number of
 * active sites reaches ncap. Output: batch,t,alive,runs for checkpoints t (log grid).
 * Build: cc -O2 -o sim sim.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static uint64_t s[4];
static inline uint64_t rotl(uint64_t x, int k){return (x<<k)|(x>>(64-k));}
static inline uint64_t nxt(void){
  uint64_t r=s[0]+s[3], t=s[1]<<17;
  s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3]; s[2]^=t; s[3]=rotl(s[3],45);
  return r;
}
static inline double u01(void){return (nxt()>>11)*(1.0/9007199254740992.0);}
static void seedrng(uint64_t z){
  for(int i=0;i<4;i++){z+=0x9e3779b97f4a7c15ULL; uint64_t x=z;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; s[i]=x^(x>>31);}
}
int main(int argc,char**argv){
  double p=0.6447001, phi=0.0; long runs=1000, batches=1; long tmax=1000, ncap=2000, L=1<<20; uint64_t seed=1;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--p"))p=atof(argv[++i]);
    else if(!strcmp(argv[i],"--phi"))phi=atof(argv[++i]);
    else if(!strcmp(argv[i],"--runs"))runs=atol(argv[++i]);
    else if(!strcmp(argv[i],"--batches"))batches=atol(argv[++i]);
    else if(!strcmp(argv[i],"--tmax"))tmax=atol(argv[++i]);
    else if(!strcmp(argv[i],"--ncap"))ncap=atol(argv[++i]);
    else if(!strcmp(argv[i],"--L"))L=atol(argv[++i]);
    else if(!strcmp(argv[i],"--seed"))seed=strtoull(argv[++i],0,10);
    else {fprintf(stderr,"unknown arg %s\n",argv[i]);return 2;}
  }
  long ncp=0; long *cp=malloc(sizeof(long)*4096); double g=1.0;
  for(long t=1;t<=tmax;){cp[ncp++]=t; g*=1.12; long n=(long)llround(g); t = n>t? n : t+1;}
  if(cp[ncp-1]!=tmax) cp[ncp++]=tmax;
  uint8_t *mark=calloc(L,1); int64_t *cur=malloc(sizeof(int64_t)*(ncap*3+16)), *nx=malloc(sizeof(int64_t)*(ncap*3+16));
  printf("batch,t,alive,runs\n");
  for(long b=0;b<batches;b++){
    seedrng(seed*1000003ULL+b);
    long *alive=calloc(ncp,sizeof(long));
    for(long r=0;r<runs;r++){
      long nc=1; cur[0]=0; long k=0; long t=0; int est=0;
      while(k<ncp){
        /* advance one generation */
        if(!est && nc>0){
          long nn=0;
          for(long i=0;i<nc;i++){
            int64_t x=cur[i];
            if(u01()<p){int64_t y=(x+1)%L; if(!mark[y]){mark[y]=1;nx[nn++]=y;}}
            if(u01()<p){int64_t y=(x-1+L)%L; if(!mark[y]){mark[y]=1;nx[nn++]=y;}}
            if(phi>0 && u01()<phi){int64_t y=(int64_t)(u01()*L); if(!mark[y]){mark[y]=1;nx[nn++]=y;}}
          }
          for(long i=0;i<nn;i++)mark[nx[i]]=0;
          int64_t *tmp=cur;cur=nx;nx=tmp; nc=nn; t++;
          if(nc>=ncap) est=1;
        } else if(nc==0){ break; }
        else { t=cp[ncp-1]; /* established: alive at all remaining checkpoints */ }
        while(k<ncp && cp[k]<=t){ if(nc>0||est) alive[k]++; k++; }
        if(est) { while(k<ncp){alive[k]++;k++;} }
      }
    }
    for(long j=0;j<ncp;j++)printf("%ld,%ld,%ld,%ld\n",b,cp[j],alive[j],runs);
    free(alive);
  }
  return 0;
}
