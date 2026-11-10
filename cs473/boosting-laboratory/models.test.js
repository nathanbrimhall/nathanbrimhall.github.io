'use strict';
const assert=require('node:assert/strict');
const M=require('./models.js');
const near=(a,b,tol=1e-9)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
for(const noisy of [false,true])for(const flip of [false,true])for(const eta of [.1,.5,1,2]){
 const data=M.makeData(noisy,flip),validation=M.makeData(noisy,false,true);
 const trajectories=Object.fromEntries(['reg','ada','logit'].map(k=>[k,M.build(k,data,validation,eta,12)]));
 // A separately changed validation set must have no effect on any fitted learner.
 for(const k of Object.keys(trajectories)){
   const other=validation.map(d=>({...d,x:10-d.x,demand:d.demand+100,t:1-d.t,y:-d.y}));
   const refit=M.build(k,data,other,eta,12);
   assert.deepEqual(trajectories[k].models,refit.models);
 }
 const reg=trajectories.reg;near(reg.initial,M.mean(data.map(d=>d.demand)));
 for(const r of reg.rounds){for(const side of [true,false]){let ids=data.map((d,i)=>({d,i})).filter(q=>(q.d.x<=r.tree.threshold)===side);if(ids.length)near(r.tree.values[side?0:1],M.mean(ids.map(q=>q.d.demand-r.before[q.i])))}for(let i=0;i<data.length;i++)near(r.after[i],r.before[i]+eta*r.correction[i]);}
 // Manual first-round AdaBoost error, coefficient, score and normalization.
 const ada=trajectories.ada;for(const r of ada.rounds){near(r.tree.error,Math.min(...r.tree.candidates.map(c=>c.error)));near(r.tree.error,data.reduce((s,d,i)=>s+r.weights[i]*(M.classValue(r.tree,d.x)!==d.y),0));near(r.alpha,.5*Math.log((1-Math.max(r.tree.error,1e-12))/Math.max(r.tree.error,1e-12)));const raw=r.weights.map((w,i)=>w*Math.exp(-eta*r.alpha*data[i].y*r.correction[i])),z=raw.reduce((a,b)=>a+b,0);for(let i=0;i<data.length;i++){near(r.after[i],r.before[i]+eta*r.alpha*r.correction[i]);near(r.afterWeights[i],raw[i]/z)}near(r.afterWeights.reduce((a,b)=>a+b,0),1)}
 const logit=trajectories.logit;for(let n=0;n<logit.rounds.length;n++){const r=logit.rounds[n];for(const side of [true,false]){const ids=data.map((d,i)=>({d,i})).filter(q=>(q.d.x<=r.tree.threshold)===side);if(ids.length){let num=ids.reduce((s,q)=>s+r.weights[q.i]*r.targets[q.i],0),den=ids.reduce((s,q)=>s+r.weights[q.i],0);near(r.tree.values[side?0:1],num/den)}}for(let i=0;i<data.length;i++){near(r.grad[i],M.sigmoid(r.before[i])-data[i].t);near(r.after[i],r.before[i]+r.step*r.correction[i])}assert.ok(logit.history[n+1].train.loss<=logit.history[n].train.loss+1e-12)}
}
// Direct one-round hand-sized examples: constant initialization, exact leaf means, and class error.
const simple=[{x:1,demand:1,t:0,y:-1},{x:2,demand:3,t:0,y:-1},{x:3,demand:5,t:1,y:1},{x:4,demand:7,t:1,y:1}];
const regression=M.build('reg',simple,simple,.5,1),r=regression.rounds[0];near(regression.initial,4);near(r.tree.threshold,2.5);near(r.tree.values[0],-2);near(r.tree.values[1],2);near(r.after[0],3);
const ada=M.build('ada',simple,simple,.5,1).rounds[0];near(ada.tree.error,0);near(ada.alpha,.5*Math.log((1-1e-12)/1e-12));near(ada.afterWeights[0],.25);
const logit=M.build('logit',simple,simple,.5,1).rounds[0];near(logit.p[0],.5);near(logit.weights[0],.25);near(logit.targets[0],-2);near(logit.tree.values[0],-2);near(logit.after[0],-1);
console.log('Passed: regression leaves, weighted stump optimum, AdaBoost updates, LogitBoost leaf and loss, manual examples, validation independence.');
