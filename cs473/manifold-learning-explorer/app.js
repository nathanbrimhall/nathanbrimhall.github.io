function jacobi(input){let a=input.map(r=>r.slice()),n=a.length,v=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));for(let z=0;z<100*n*n;z++){let p=0,q=1,best=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(a[i][j])>best){best=Math.abs(a[i][j]);p=i;q=j}if(best<1e-12)break;let app=a[p][p],aqq=a[q][q],apq=a[p][q],tau=(aqq-app)/(2*apq),t=(tau===0?1:Math.sign(tau))/(Math.abs(tau)+Math.sqrt(1+tau*tau)),c=1/Math.sqrt(1+t*t),s=t*c;for(let k=0;k<n;k++){if(k!==p&&k!==q){let akp=a[k][p],akq=a[k][q];a[k][p]=a[p][k]=c*akp-s*akq;a[k][q]=a[q][k]=s*akp+c*akq}let vip=v[k][p],viq=v[k][q];v[k][p]=c*vip-s*viq;v[k][q]=s*vip+c*viq}a[p][p]=app-t*apq;a[q][q]=aqq+t*apq;a[p][q]=a[q][p]=0}
 let order=Array.from({length:n},(_,i)=>i).sort((i,j)=>a[j][j]-a[i][i]),vals=order.map(i=>Math.max(0,a[i][i])),vec=order.map(i=>{let col=v.map(r=>r[i]),k=col.findIndex(x=>Math.abs(x)>1e-12);if(k>=0&&col[k]<0)col=col.map(x=>-x);return col});return {values:vals,vectors:vec,cov:input}}

"use strict";
const $ = id => document.getElementById(id);
const labels = ["See the shape", "Compare two points", "Build a neighbor graph", "Find a graph path", "Compare embeddings"];
const explanations = [
  "The rolled sheet has two underlying coordinates even though each sensor measurement has three values. The second view shows the known coordinates used to generate this simulation.",
  "The dashed line goes directly through 3D space. Points on neighboring layers can have a short straight-line distance but be far apart along the sheet.",
  "Each point connects to its k nearest neighbors. Small k may leave the graph disconnected. Large k can connect different layers and create shortcuts.",
  "The orange path follows graph edges and minimizes their total length. It estimates travel along the sheet when neighbor links follow the surface.",
  "PCA and classical MDS use the same Euclidean geometry. Isomap first replaces straight-line distances with shortest graph-path distances, then applies MDS."
];
let step = 0, points = [], graph, distances, paths, pcaMap, mdsMap, isoMap;
const count = 40;
function makePoints() {
  points = [];
  for (let i=0;i<10;i++) for (let j=0;j<4;j++) {
    const t=4.8+i*.72, h=j*2.5+(i%2)*.15;
    const flat=$('shape').value==='flat';
    points.push({xyz:flat?[i*1.7,h,0]:[t*Math.cos(t),h,t*Math.sin(t)], truth:[i,h], color:`hsl(${240-i*20},70%,43%)`});
  }
  $('a').innerHTML=points.map((_,i)=>`<option value="${i}">Point ${i+1}</option>`).join('');
  $('b').innerHTML=$('a').innerHTML; $('a').value=4; $('b').value=36;
  step=0; compute();
}
function euclid(a,b){return Math.hypot(...a.map((x,i)=>x-b[i]));}
function embedding(d) {
  const n=d.length, sq=d.map(row=>row.map(x=>x*x));
  const means=sq.map(row=>row.reduce((a,b)=>a+b,0)/n), total=means.reduce((a,b)=>a+b,0)/n;
  const matrix=sq.map((row,i)=>row.map((x,j)=>-.5*(x-means[i]-means[j]+total)));
  const e=jacobi(matrix);
  return matrix.map((_,i)=>[e.vectors[0][i]*Math.sqrt(e.values[0]),e.vectors[1][i]*Math.sqrt(e.values[1])]);
}
function compute(){
  const n=points.length,k=+$('k').value;
  distances=points.map(a=>points.map(b=>euclid(a.xyz,b.xyz)));
  graph=Array.from({length:n},()=>Array(n).fill(Infinity));
  for(let i=0;i<n;i++){graph[i][i]=0;let near=Array.from({length:n},(_,j)=>j).filter(j=>j!==i).sort((a,b)=>distances[i][a]-distances[i][b]);for(const j of near.slice(0,k))graph[i][j]=graph[j][i]=distances[i][j];}
  paths=graph.map((row,i)=>row.map((d,j)=>Number.isFinite(d)&&i!==j?j:-1));
  const shortest=graph.map(row=>row.slice());
  for(let k=0;k<n;k++)for(let i=0;i<n;i++)for(let j=0;j<n;j++)if(shortest[i][k]+shortest[k][j]<shortest[i][j]){shortest[i][j]=shortest[i][k]+shortest[k][j];paths[i][j]=paths[i][k];}
  const mean=[0,1,2].map(j=>points.reduce((sum,p)=>sum+p.xyz[j],0)/n);
  const centered=points.map(p=>p.xyz.map((x,j)=>x-mean[j]));
  const covariance=mean.map((_,i)=>mean.map((_,j)=>centered.reduce((sum,p)=>sum+p[i]*p[j],0)/(n-1)));
  const pc=jacobi(covariance);
  pcaMap=centered.map(row=>pc.vectors.slice(0,2).map(v=>row.reduce((sum,x,j)=>sum+x*v[j],0)));
  mdsMap=embedding(distances);
  isoMap=shortest.every(row=>row.every(Number.isFinite))?embedding(shortest):null;
  graph.shortest=shortest;render();
}
function path(a,b){if(paths[a][b]<0)return [];let result=[a];while(a!==b&&result.length<=count){a=paths[a][b];result.push(a)}return result;}
function draw(id,coords,links=false){
  const xs=coords.map(p=>p[0]),ys=coords.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),scale=Math.min(420/(xmax-xmin||1),270/(ymax-ymin||1));
  const X=x=>250+(x-(xmin+xmax)/2)*scale,Y=y=>170-(y-(ymin+ymax)/2)*scale;
  let s='<rect width="500" height="350" fill="white"/>';
  const seg=(i,j,color,width,dash='')=>`<line x1="${X(coords[i][0])}" y1="${Y(coords[i][1])}" x2="${X(coords[j][0])}" y2="${Y(coords[j][1])}" stroke="${color}" stroke-width="${width}" ${dash}/>`;
  const a=+$('a').value,b=+$('b').value;
  if(links&&step>=2)for(let i=0;i<count;i++)for(let j=i+1;j<count;j++)if(Number.isFinite(graph[i][j]))s+=seg(i,j,(i===a||j===a||i===b||j===b)?'#178358':'#c8d4df',1);
  if(links&&step>=1)s+=seg(a,b,'#cc4534',2,'stroke-dasharray="5 4"');
  if(links&&step>=3){let p=path(a,b);for(let i=1;i<p.length;i++)s+=seg(p[i-1],p[i],'#e6761e',3)}
  points.forEach((p,i)=>{s+=`<circle cx="${X(coords[i][0])}" cy="${Y(coords[i][1])}" r="${i===a||i===b?7:4}" fill="${p.color}" stroke="${i===a?'#000':i===b?'#a028b0':'white'}" stroke-width="${i===a||i===b?3:1}"><title>Point ${i+1}</title></circle>`});$(id).innerHTML=s;
}
function render(){
  const a=+$('a').value,b=+$('b').value,theta=+$('angle').value*Math.PI/180;
  const projected=points.map(p=>[p.xyz[0]*Math.cos(theta)+p.xyz[2]*Math.sin(theta),p.xyz[1]+.3*(-p.xyz[0]*Math.sin(theta)+p.xyz[2]*Math.cos(theta))]);
  $('step-title').textContent=`Step ${step+1} · ${labels[step]}`;$('steps').innerHTML=labels.map((x,i)=>`<li ${i===step?'aria-current="step"':''}>${i+1} · ${x}</li>`).join('');$('explain').textContent=explanations[step];
  const length=graph.shortest[a][b];$('distance').textContent=step>=1?`Straight-line distance: ${distances[a][b].toFixed(2)}. ${step>=3?(Number.isFinite(length)?`Shortest graph path: ${length.toFixed(2)} (${Math.max(0,path(a,b).length-1)} edges).`:'No graph path connects these points.'):'Continue to compare a graph path.'}`:'';
  $('angle-out').textContent=$('angle').value+'°';$('k-out').textContent=$('k').value;$('back').disabled=step===0;$('next').disabled=step===4;$('embeddings').hidden=step<4;
  draw('cloud',projected,true);draw('truth',points.map(p=>p.truth));draw('pca',pcaMap);draw('mds',mdsMap);if(isoMap){draw('isomap',isoMap);$('iso-status').textContent='Graph connected. Isomap uses all shortest-path distances.'}else{$('isomap').innerHTML='<text x="250" y="175" text-anchor="middle">Graph disconnected — increase k</text>';$('iso-status').textContent='Isomap requires finite paths between all points.'}
}
$('next').onclick=()=>{step=Math.min(4,step+1);render()};$('back').onclick=()=>{step=Math.max(0,step-1);render()};$('reset').onclick=()=>{step=0;render()};$('shape').onchange=makePoints;$('k').oninput=compute;for(const id of ['a','b','angle'])$(id).oninput=render;
const quiz=[['What is the intrinsic dimension of the ideal sheet?','Two','Two coordinates locate a point on the sheet.'],['What does MDS take as input?','Pairwise distances','MDS arranges points using a distance or dissimilarity table.'],['What does a graph path estimate?','Distance along the sheet','Short neighbor links can trace the surface.'],['Why can Isomap unfold the roll?','It uses graph distances','It applies MDS to shortest-path distances.'],['What can happen when k is too large?','Links can shortcut between layers','Large neighborhoods may connect separate folds.']];
$('quiz').innerHTML=quiz.map((q,i)=>`<article class="quiz-item"><h3>${i+1}. ${q[0]}</h3><button data-q="${i}">${q[1]}</button><button data-q="${i}">Always three dimensions</button><p id="feedback${i}" aria-live="polite"></p></article>`).join('');document.querySelectorAll('[data-q]').forEach(button=>button.onclick=()=>{let i=+button.dataset.q; $(`feedback${i}`).textContent=(button.textContent===quiz[i][1]?'Correct. ':'Try again. ')+quiz[i][2]});
makePoints();
