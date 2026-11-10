/* Deterministic educational models. Only training data are passed to fitters. */
'use strict';
const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
const softplus = x => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const logisticLoss = (t, f) => softplus(f) - t * f;
function makeData(noisy, flip, validation = false) {
  const n = validation ? 12 : 10, offset = validation ? .31 : 0;
  return Array.from({length:n}, (_, i) => {
    const x = (i + .5 + offset) * 10 / n;
    const noise = Math.sin((i + 1) * 19.31 + (validation ? 2.8 : .4));
    const demand = 11 + .12 * (x - 5) ** 2 + 1.7 * Math.sin(.8 * x) + (noisy ? 1.05 : .25) * noise;
    const prob = sigmoid(-1.2 + 2.2 * Math.sin(.72 * x) + .25 * (x - 5));
    const u = (Math.sin((i + 1) * 127.31 + (validation ? 17 : 0)) * 43758.5453 % 1 + 1) % 1;
    let t = noisy ? +(u < prob) : +(prob >= .5);
    if (flip && !validation && i === 7) t = 1 - t;
    return {id:(validation?'V':'T')+(i+1), x, demand, t, y:2*t-1};
  });
}
function regressionStump(data, targets, weights) {
  const n=data.length, candidates=[];
  // A constant tree is a valid one-leaf option; every other candidate has two nonempty leaves.
  const thresholds=[-Infinity,...data.slice(0,-1).map((d,i)=>(d.x+data[i+1].x)/2)];
  for(const threshold of thresholds){
    const groups=[[],[]];data.forEach((d,i)=>groups[d.x<=threshold?0:1].push(i));
    const values=groups.map(ids=>{const den=ids.reduce((s,i)=>s+weights[i],0);return den?ids.reduce((s,i)=>s+weights[i]*targets[i],0)/den:0});
    const sse=data.reduce((s,d,i)=>s+weights[i]*(targets[i]-values[d.x<=threshold?0:1])**2,0);
    candidates.push({threshold,values,sse});
  }
  candidates.sort((a,b)=>a.sse-b.sse);return {...candidates[0], candidates};
}
const stumpValue=(tree,x)=>tree.values[x<=tree.threshold?0:1];
function classStump(data,weights){
  const thresholds=[-Infinity,...data.slice(0,-1).map((d,i)=>(d.x+data[i+1].x)/2)],candidates=[];
  for(const threshold of thresholds)for(const left of [-1,1]){
    const predict=x=>x<=threshold?left:-left;
    const error=data.reduce((s,d,i)=>s+weights[i]*(predict(d.x)!==d.y),0);
    candidates.push({threshold,left,error});
  }
  candidates.sort((a,b)=>a.error-b.error);return {...candidates[0],candidates};
}
const classValue=(s,x)=>x<=s.threshold?s.left:-s.left;
function normaliseLogs(logs){const max=Math.max(...logs),a=logs.map(x=>Math.exp(x-max)),z=a.reduce((s,x)=>s+x,0);return a.map(x=>x/z)}
function build(kind,data,validation,eta,maxRounds){
  const initial=kind==='reg'?mean(data.map(d=>d.demand)):kind==='ada'?0:Math.log(mean(data.map(d=>d.t))/(1-mean(data.map(d=>d.t))));
  const models=[],rounds=[],history=[];let stop='';
  const score=(x, upto=models.length)=>initial+models.slice(0,upto).reduce((s,m)=>s+m.step*(kind==='ada'?classValue(m.tree,x):stumpValue(m.tree,x)),0);
  const metrics=(set,upto)=>{const f=set.map(d=>score(d.x,upto));if(kind==='reg')return {loss:mean(set.map((d,i)=>.5*(d.demand-f[i])**2))};const prob=f.map(v=>sigmoid(kind==='ada'?2*v:v));return {loss:mean(set.map((d,i)=>logisticLoss(d.t,kind==='ada'?2*f[i]:f[i]))),accuracy:mean(set.map((d,i)=>+(+(f[i]>=0)===d.t))),objective:kind==='ada'?set.reduce((s,d,i)=>s+Math.exp(-d.y*f[i]),0):mean(set.map((d,i)=>logisticLoss(d.t,f[i]))),prob}};
  history.push({train:metrics(data,0),validation:metrics(validation,0)});
  for(let round=0;round<maxRounds;round++){
    const before=data.map(d=>score(d.x));let tree,step=eta,alpha=null,weights,targets,grad,p,afterWeights=null,warning='';
    if(kind==='reg'){
      targets=data.map((d,i)=>d.demand-before[i]);weights=data.map(()=>1);tree=regressionStump(data,targets,weights);
    } else if(kind==='ada'){
      weights=normaliseLogs(data.map((d,i)=>-d.y*before[i]));tree=classStump(data,weights);
      if(tree.error>=.5-1e-12){stop='No stump has weighted error below 0.5; fitting stops.';break}
      const safe=Math.max(tree.error,1e-12);alpha=.5*Math.log((1-safe)/safe);step=eta*alpha;
      if(tree.error<1e-12)warning='Perfect stump: ε is capped at 10⁻¹² to display a finite coefficient, then fitting stops.';
      afterWeights=normaliseLogs(data.map((d,i)=>Math.log(weights[i])-step*d.y*classValue(tree,d.x)));
    } else {
      p=before.map(sigmoid);grad=p.map((v,i)=>v-data[i].t);weights=p.map(v=>Math.max(1e-10,v*(1-v)));targets=weights.map((w,i)=>-grad[i]/w);tree=regressionStump(data,targets,weights);
      const old=mean(data.map((d,i)=>logisticLoss(d.t,before[i])));let candidate=mean(data.map((d,i)=>logisticLoss(d.t,before[i]+step*stumpValue(tree,d.x))));
      while(candidate>old+1e-12 && step>eta/1048576){step/=2;candidate=mean(data.map((d,i)=>logisticLoss(d.t,before[i]+step*stumpValue(tree,d.x))));}
      if(candidate>old+1e-12 || !Number.isFinite(candidate) || candidate>=old-1e-14){stop='No improving logistic-loss step was found; fitting stops.';break}
      if(step<eta)warning='The requested step increased logistic loss, so it was halved until the original loss did not increase.';
    }
    const correction=data.map(d=>kind==='ada'?classValue(tree,d.x):stumpValue(tree,d.x));
    const after=before.map((f,i)=>f+step*correction[i]);
    models.push({tree,step});rounds.push({before,after,tree,step,alpha,weights,afterWeights,targets,grad,p,correction,warning});
    history.push({train:metrics(data,models.length),validation:metrics(validation,models.length)});
    if(kind==='ada'&&tree.error<1e-12){stop=warning;break}
  }
  return {kind,data,validation,initial,models,rounds,history,score,stop};
}
if(typeof module!=='undefined')module.exports={sigmoid,softplus,mean,logisticLoss,makeData,regressionStump,stumpValue,classStump,classValue,normaliseLogs,build};
