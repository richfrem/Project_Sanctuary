/* MISSION-074: random Truchet quarter-arc tilings on an LxL torus.
 * usage: tilings L p nsamples seed        -> one JSON line per sample
  *        tilings L p nsamples seed [checker]   (checker: P(A)=p on even i+j, 1-p on odd)
 *        tilings --dump L p seed [checker]          -> tile bits, then sorted loop lengths (gate input)
 * Nodes = tile-edge midpoints: H(i,j)=top edge of tile (i,j) idx i*L+j; V(i,j)=left edge idx L*L+i*L+j.
 * Orientation A joins N-W and S-E; B joins N-E and S-W. Loop length = number of arcs = nodes in the cycle. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t s[4];
static uint64_t rotl(uint64_t x,int k){return (x<<k)|(x>>(64-k));}
static uint64_t nxt(void){uint64_t r=rotl(s[1]*5,7)*9,t=s[1]<<17;s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rotl(s[3],45);return r;}
static void seed(uint64_t x){for(int i=0;i<4;i++){x+=0x9e3779b97f4a7c15ULL;uint64_t z=x;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;s[i]=z^(z>>31);}}
static double u01(void){return (nxt()>>11)*(1.0/9007199254740992.0);}
static int checker=0;
static unsigned char draw(int idx,int L,double p){double q=(checker&&(((idx/L)+(idx%L))&1))?1.0-p:p;return (u01()<q)?0:1;}
static int *par,*sz;
static int find(int a){while(par[a]!=a){par[a]=par[par[a]];a=par[a];}return a;}
static void uni(int a,int b){a=find(a);b=find(b);if(a==b)return;if(sz[a]<sz[b]){int t=a;a=b;b=t;}par[b]=a;sz[a]+=sz[b];}
static int cmp(const void*a,const void*b){int x=*(const int*)a,y=*(const int*)b;return (x>y)-(x<y);}
static void build(int L,const unsigned char*t){
  int n=L*L; for(int i=0;i<2*n;i++){par[i]=i;sz[i]=1;}
  for(int i=0;i<L;i++)for(int j=0;j<L;j++){
    int N=i*L+j,S=((i+1)%L)*L+j,W=n+i*L+j,E=n+i*L+(j+1)%L;
    if(t[i*L+j]==0){uni(N,W);uni(S,E);}else{uni(N,E);uni(S,W);}}
}
int main(int argc,char**argv){
  if(argc>=2&&!strcmp(argv[1],"--dump")){
    int L=atoi(argv[2]);double p=atof(argv[3]);seed(strtoull(argv[4],0,10));if(argc>5&&!strcmp(argv[5],"checker"))checker=1;
    int n=L*L;par=malloc(2*n*sizeof(int));sz=malloc(2*n*sizeof(int));unsigned char*t=malloc(n);
    for(int i=0;i<n;i++)t[i]=draw(i,L,p);
    build(L,t);
    for(int i=0;i<n;i++)putchar('0'+t[i]);putchar('\n');
    int*v=malloc(2*n*sizeof(int));int m=0;for(int i=0;i<2*n;i++)if(par[i]==i)v[m++]=sz[i];
    qsort(v,m,sizeof(int),cmp);for(int i=0;i<m;i++)printf("%d%c",v[i],i+1<m?' ':'\n');return 0;}
  if(argc<5){fprintf(stderr,"usage: tilings L p nsamples seed\n");return 1;}
  int L=atoi(argv[1]);double p=atof(argv[2]);int ns=atoi(argv[3]);uint64_t sd=strtoull(argv[4],0,10);if(argc>5&&!strcmp(argv[5],"checker"))checker=1;
  int n=L*L;par=malloc(2*n*sizeof(int));sz=malloc(2*n*sizeof(int));unsigned char*t=malloc(n);
  for(int k=0;k<ns;k++){
    seed(sd*1000003ULL+k);
    for(int i=0;i<n;i++)t[i]=draw(i,L,p);
    build(L,t);
    long sum=0;int mx=0,nl=0;long ge[6]={0};
    for(int i=0;i<2*n;i++)if(par[i]==i){int z=sz[i];sum+=z;nl++;if(z>mx)mx=z;
      for(int b=0;b<6;b++)if(z>=(128<<b))ge[b]++;}
    printf("{\"L\":%d,\"p\":%g,\"seed\":%llu,\"k\":%d,\"loops\":%d,\"max\":%d,\"sum\":%ld,\"ge\":[%ld,%ld,%ld,%ld,%ld,%ld]}\n",
      L,p,(unsigned long long)sd,k,nl,mx,sum,ge[0],ge[1],ge[2],ge[3],ge[4],ge[5]);}
  return 0;}
