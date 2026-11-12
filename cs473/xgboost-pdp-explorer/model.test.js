'use strict';
const assert=require('node:assert/strict'),m=require('./model.js');
const near=(a,b,t=1e-9)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
// Hand calculation: baseline 20/6; G_L = 6, H_L = 3, G_R = -6, H_R = 3.
const d=[0,2,4,6].map((y,i)=>({id:`B${i+1}`,x:[i+1,10*(i+1)],y})),a=[1,2,1,2],base=m.weightedMean(d,a);near(base,20/6);
let r=m.fitRound(d,a,base,[],1,2,.5),c=m.candidate(d,r.g,r.h,0,2.5,1,2);near(c.L.G,6);near(c.R.G,-6);near(c.L.H,3);near(c.R.H,3);near(c.left,-1.5);near(c.right,1.5);near(c.gain,7);near(r.best.gain,7);assert.equal(r.tree.split,true);near(r.after[0],base-.75);near(r.after[3],base+.75);
assert.equal(m.candidate(d,r.g,r.h,0,0,1,2),null);assert.equal(m.candidate(d,r.g,[0,0,1,2],0,2.5,1,2),null);
let noSplit=m.fitRound(d,a,base,[],1,100,.5);assert.equal(noSplit.tree.split,false);near(noSplit.tree.weight,0);assert.ok(noSplit.best.gain<0);
for(const lambda of [0,1,4]){let v=m.candidate(d,r.g,r.h,0,2.5,lambda,0);near(v.left,-6/(3+lambda));near(v.right,6/(3+lambda));near(v.gain,36/(3+lambda));}
let shrink=m.fitRound(d,a,base,[],1,2,.25);near(shrink.tree.left,r.tree.left);near(shrink.after[0],base-.375);
const building=m.data,w=building.map((_,i)=>i===2?3:1),b=m.weightedMean(building,w);let trees=[];for(let i=0;i<4;i++){let step=m.fitRound(building,w,b,trees,1,0,.5);for(let j=0;j<building.length;j++){near(step.g[j],w[j]*(step.before[j]-building[j].y));near(step.h[j],w[j])}assert.equal(step.best.gain,Math.max(...step.candidates.map(q=>q.gain)));trees.push(step.tree)}
assert.ok(trees.some(t=>t.split&&t.feature===0));assert.ok(trees.some(t=>t.split&&t.feature===1));
for(let f=0;f<2;f++)for(const z of [f?200:3,f?550:16,f?800:29]){let result=m.pdp(building,b,trees,f,z);let ice=building.map(row=>{let x=row.x.slice();x[f]=z;return m.predict(b,trees,x)});near(result.average,ice.reduce((p,q)=>p+q,0)/building.length);assert.deepEqual(result.values,ice);}
// Training weights are not inputs to PDP; changing the weight array without refitting cannot change its mean.
const old=m.pdp(building,b,trees,0,16),copy=JSON.stringify(trees);w[2]=.25;assert.deepEqual(m.pdp(building,b,trees,0,16),old);assert.equal(JSON.stringify(trees),copy);
console.log('Passed: manual weighted baseline, gradients, leaf weights, gain, split selection, penalties, shrinkage, no-split behavior, fixed PDP and ICE averaging.');
