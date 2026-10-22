(function(root){
 'use strict';
 function rng(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
 const normal=r=>Math.sqrt(-2*Math.log(Math.max(1e-12,r())))*Math.cos(2*Math.PI*r());
 function kernel(u,type){if(type==='uniform')return Math.abs(u)<=1?.5:0;if(type==='epanechnikov')return Math.abs(u)<=1?.75*(1-u*u):0;return Math.exp(-.5*u*u)/Math.sqrt(2*Math.PI)}
 function generate(mode,n,seed=473){const r=rng(seed),all=[];for(let i=0;i<n;i++){if(mode==='density'){const group=r()<.55;all.push({id:i,x:(group?21.5:29)+normal(r)*(group?1.8:2.3)})}else if(mode==='classification'){const c=i%2;let x;do{x=(c?6.4:3.4)+normal(r)*1.15}while(x<0||x>10);all.push({id:i,x,y:c})}else{const x=20*r();all.push({id:i,x,y:Math.max(2,truth(x)+normal(r)*3)})}}return all}
 const truth=x=>12+2*x+8*Math.sin(x/3.8);
 function estimate(data,x,h,type,mode,knn=false,k=5){const order=data.map(d=>({...d,u:(x-d.x)/h})).sort((a,b)=>Math.abs(a.x-x)-Math.abs(b.x-x)||a.id-b.id),ids=new Set(order.slice(0,Math.min(k,data.length)).map(d=>d.id));const rows=data.map(d=>({...d,u:(x-d.x)/h,raw:knn?+ids.has(d.id):kernel((x-d.x)/h,type)}));const sum=rows.reduce((s,d)=>s+d.raw,0);rows.forEach(d=>{d.weight=sum?d.raw/sum:0;d.contribution=mode==='density'?d.raw/(data.length*h):sum?d.weight*d.y:0});const numerator=rows.reduce((s,d)=>s+d.raw*(mode==='density'?1:d.y),0),value=mode==='density'?sum/(data.length*h):sum?numerator/sum:null;return{rows,sum,numerator,value,active:rows.filter(d=>d.raw>0).length,radius:knn?Math.abs(order[Math.min(k,data.length)-1].x-x):h,k:Math.min(k,data.length)}}
 const api={kernel,generate,estimate,truth};if(typeof module!=='undefined')module.exports=api;else root.KernelMath=api;
})(typeof window!=='undefined'?window:globalThis);
