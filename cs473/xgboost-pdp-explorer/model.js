'use strict';
// Values and noise are fabricated; x[0] is °C, x[1] is building area in m².
const data=[[2,180],[5,370],[7,240],[9,550],[12,320],[15,640],[18,400],[21,710],[24,500],[27,820],[30,600]].map(([temp,size],i)=>({id:`B${i+1}`,x:[temp,size],y:Math.round((16+.045*size+.075*(temp-18)**2+[1,-2,2,0,-1,2,-2,1,-1,1,-2][i])*10)/10}));
const featureNames=['Temperature','Building size'],units=['°C','m²'];
const sum=a=>a.reduce((x,y)=>x+y,0),weightedMean=(d,a)=>sum(d.map((v,i)=>v.y*a[i]))/sum(a);
const predict=(base,trees,x)=>base+sum(trees.map(t=>t.eta*leaf(t,x)));
const leaf=(t,x)=>t.split?(x[t.feature]<=t.threshold?t.left:t.right):t.weight;
function stats(indices,g,h){return {n:indices.length,G:sum(indices.map(i=>g[i])),H:sum(indices.map(i=>h[i]))}}
function thresholdOptions(d,feature){const vals=[...new Set(d.map(q=>q.x[feature]))].sort((a,b)=>a-b);return vals.slice(0,-1).map((x,i)=>(x+vals[i+1])/2)}
function candidate(d,g,h,feature,threshold,lambda,gamma){let l=[],r=[];d.forEach((row,i)=>(row.x[feature]<=threshold?l:r).push(i));let L=stats(l,g,h),R=stats(r,g,h),P=stats([...l,...r],g,h);if(!L.n||!R.n||L.H<=0||R.H<=0)return null;let left=-L.G/(L.H+lambda),right=-R.G/(R.H+lambda),gain=.5*(L.G**2/(L.H+lambda)+R.G**2/(R.H+lambda)-P.G**2/(P.H+lambda))-gamma;return {feature,threshold,L,R,P,left,right,gain,leftIds:l,rightIds:r}}
function fitRound(d,a,base,trees,lambda,gamma,eta){let before=d.map(row=>predict(base,trees,row.x)),g=d.map((row,i)=>a[i]*(before[i]-row.y)),h=a.slice(),parent=stats(d.map((_,i)=>i),g,h),candidates=[];for(let f=0;f<2;f++)for(const threshold of thresholdOptions(d,f)){let c=candidate(d,g,h,f,threshold,lambda,gamma);if(c)candidates.push(c)}candidates.sort((u,v)=>v.gain-u.gain||u.feature-v.feature||u.threshold-v.threshold);const best=candidates[0],tree=best&&best.gain>1e-12?{split:true,feature:best.feature,threshold:best.threshold,left:best.left,right:best.right,eta}:{split:false,weight:-parent.G/(parent.H+lambda),eta};let after=d.map((row,i)=>before[i]+eta*leaf(tree,row.x));let loss=f=>sum(d.map((row,i)=>a[i]*.5*(row.y-f[i])**2));return {before,g,h,parent,candidates,best,tree,after,lossBefore:loss(before),lossAfter:loss(after),lambda,gamma,eta}}
function pdp(d,base,trees,feature,z){let values=d.map(q=>{let x=q.x.slice();x[feature]=z;return predict(base,trees,x)});return {values,average:sum(values)/values.length}}
if(typeof module!=='undefined')module.exports={data,featureNames,units,sum,weightedMean,predict,leaf,stats,thresholdOptions,candidate,fitRound,pdp};
