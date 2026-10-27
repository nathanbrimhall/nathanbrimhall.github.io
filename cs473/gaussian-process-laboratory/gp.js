(function(root){
 'use strict';
 function rng(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
 const normal=r=>Math.sqrt(-2*Math.log(Math.max(r(),1e-12)))*Math.cos(2*Math.PI*r());
 function kernel(x,z,p){const r=Math.abs(x-z),a=p.signal**2;if(p.type==='matern'){const d=Math.sqrt(3)*r/p.length;return a*(1+d)*Math.exp(-d)}if(p.type==='periodic')return a*Math.exp(-2*Math.sin(Math.PI*(x-z)/p.period)**2/p.length**2);if(p.type==='linear')return a*(1+(x-p.center)*(z-p.center)/p.length**2);return a*Math.exp(-.5*(r/p.length)**2)}
 function cholesky(A){const n=A.length,L=Array.from({length:n},()=>Array(n).fill(0));for(let i=0;i<n;i++)for(let j=0;j<=i;j++){let s=A[i][j];for(let k=0;k<j;k++)s-=L[i][k]*L[j][k];if(i===j){if(!(s>0)||!Number.isFinite(s))throw Error('Covariance could not be factorized');L[i][j]=Math.sqrt(s)}else L[i][j]=s/L[j][j]}return L}
 function lower(L,b){const x=[];for(let i=0;i<b.length;i++){let s=b[i];for(let j=0;j<i;j++)s-=L[i][j]*x[j];x[i]=s/L[i][i]}return x}
 function upper(L,b){const x=Array(b.length);for(let i=b.length-1;i>=0;i--){let s=b[i];for(let j=i+1;j<b.length;j++)s-=L[j][i]*x[j];x[i]=s/L[i][i]}return x}
 const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
 function factor(A,scale){let jitter=Math.max(1e-12,scale*1e-9);for(let i=0;i<10;i++){try{return{L:cholesky(A.map((row,j)=>row.map((v,k)=>v+(j===k?jitter:0)))),jitter}}catch(e){jitter*=10}}throw Error('Try a shorter length scale or nonzero observation noise.')}
 function fit(data,p){const K=data.map(d=>data.map(e=>kernel(d.x,e.x,p))),C=K.map((row,i)=>row.map((v,j)=>v+(i===j?p.noise**2:0))),{L,jitter}=factor(C,p.signal**2),residual=data.map(d=>d.y-p.mean),alpha=upper(L,lower(L,residual)),logEvidence=-.5*dot(residual,alpha)-L.reduce((s,row,i)=>s+Math.log(row[i]),0)-data.length/2*Math.log(2*Math.PI);return{data,p,K,C,L,jitter,residual,alpha,logEvidence}}
 function predict(model,x){const {data,p,L,alpha}=model,k=data.map(d=>kernel(d.x,x,p)),v=lower(L,k),prior=kernel(x,x,p),reduction=dot(v,v),variance=Math.max(0,prior-reduction);return{x,k,v,mean:p.mean+dot(k,alpha),variance,observationVariance:variance+p.noise**2,prior,reduction}}
 function sample(model,xs,posterior,seed=884){const p=model.p,values=xs.map(x=>posterior?predict(model,x):{mean:p.mean,v:null}),C=xs.map((x,i)=>xs.map((z,j)=>kernel(x,z,p)-(posterior?dot(values[i].v,values[j].v):0))),{L}=factor(C,p.signal**2),r=rng(seed);return Array.from({length:3},()=>{const g=xs.map(()=>normal(r));return xs.map((_,i)=>values[i].mean+L[i].slice(0,i+1).reduce((s,v,j)=>s+v*g[j],0))})}
 function generate(cfg,n,noisy,seed=473){const r=rng(seed),rows=[];for(let i=0;i<n;i++){// Nested, distinct inputs; keep their measurement errors stable as N changes.
 const x=cfg.train[0]+(cfg.train[1]-cfg.train[0])*r(),truth=cfg.truth(x),error=cfg.sensor*normal(r);rows.push({id:i,x,truth,y:truth+(noisy?error:0),error})}return rows.sort((a,b)=>a.x-b.x)}
 const api={kernel,cholesky,lower,upper,dot,fit,predict,sample,generate,rng};if(typeof module!=='undefined')module.exports=api;else root.GPLab=api;
})(typeof window!=='undefined'?window:globalThis);
