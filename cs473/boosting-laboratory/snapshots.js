'use strict';
// Each stage owns a deep-frozen copy containing only information available so far.
const stageFields = {
  reg:[[],['targets'],['targets','candidates'],['targets','tree','correction'],['targets','tree','correction','step'],['targets','tree','correction','step','after'],['targets','tree','correction','step','after','committed']],
  ada:[['weights'],['weights','candidates'],['weights','tree','correction'],['weights','tree','correction','alpha','step'],['weights','tree','correction','alpha','step','after'],['weights','tree','correction','alpha','step','after','multipliers'],['weights','tree','correction','alpha','step','after','multipliers','afterWeights','committed']],
  logit:[['p'],['p','grad'],['p','grad','weights'],['p','grad','weights','targets'],['p','grad','weights','targets','tree','correction','candidates'],['p','grad','weights','targets','tree','correction','step','after','warning'],['p','grad','weights','targets','tree','correction','step','after','warning','updatedP'],['p','grad','weights','targets','tree','correction','step','after','warning','updatedP','committed']]
};
function frozenSnapshot(kind,round,data,stage){
 const allowed=stageFields[kind][stage],out={kind,stage,before:round.before.slice()};
 for(const key of allowed){
  if(key==='candidates')out.candidates=round.tree.candidates.map(c=>structuredClone(c));
  else if(key==='multipliers')out.multipliers=data.map((d,i)=>Math.exp(-round.step*d.y*round.correction[i]));
  else if(key==='updatedP')out.updatedP=round.after.map(x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x)));
  else if(key==='committed')out.committed=true;
  else out[key]=structuredClone(round[key]);
 }
 // The fitted tree object contains candidates; the best candidate must not disclose
 // future fields through references and every snapshot is independent.
 function freeze(o){if(o&&typeof o==='object'&&!Object.isFrozen(o)){Object.values(o).forEach(freeze);Object.freeze(o)}return o}
 return freeze(out);
}
if(typeof module!=='undefined')module.exports={stageFields,frozenSnapshot};
