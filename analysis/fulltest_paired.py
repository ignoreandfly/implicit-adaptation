"""Paired McNemar tests for results/fulltest (each method vs its greedy baseline and vs Instruct greedy).
Needs the gitignored test.generations.jsonl files; run from the repo root: python -m analysis.fulltest_paired"""
import json, os
from math import comb
R='results/fulltest/'
def load(m,n): return [json.loads(l) for l in open(f'{R}{m}/{n}/test.generations.jsonl')]
def ch(rs): return [bool(r['paths'][r['chosen']]['score']['lenient_correct']) for r in rs]
def mc(x,y):
    b=sum(i and not j for i,j in zip(x,y)); c=sum(j and not i for i,j in zip(x,y)); N=b+c
    return min(1,2*sum(comb(N,k) for k in range(min(b,c)+1))/2**N) if N else 1
I=ch(load('olmo2-1b-instruct','greedy'))
for m,bn in (('olmo2-1b-sft','greedy'),('olmo2-1b-base','greedy_cot')):
    rb=load(m,bn); b=ch(rb); A=sum(I)/len(I); B=sum(b)/len(b); out=[]
    for n in sorted(os.listdir(R+m)):
        rs=load(m,n); assert all(x['question']==y['question'] for x,y in zip(rs,rb))
        v={n:ch(rs)}
        if 'cot_decoding' in n: v={n+':max':ch(rs), n+':agg':[r['agg_answer']==r['answer'] for r in rs]}
        if 'power_smc' in n: v[n+':maj']=[bool(r['paths'][r['majority']]['score']['lenient_correct']) for r in rs]
        for k,x in v.items(): out.append((sum(x)/len(x),k,mc(x,b),mc(x,I)))
    print(f'{m}: baseline {bn} {B:.3f}, Instruct greedy {A:.3f}')
    for a,k,p,q in sorted(out,reverse=True): print(f'  {k:24s} {a:.3f}  vs base {a-B:+.3f} p={p:.2g}  gap closed {100*(a-B)/(A-B):.0f}%  vs Instruct {a-A:+.3f} p={q:.2g}')
for m in ('olmo2-1b-sft','olmo2-1b-base'):
    s=json.load(open(f'{R}{m}/sc_n10/test.selection.json')); print(m,{k:round(s[k],3) for k in s if isinstance(s[k],float) and k in ('majority','oracle','eba','borda0.5:self_certainty','borda1.0:self_certainty') or 'dtr' in k and isinstance(s[k],float)})
