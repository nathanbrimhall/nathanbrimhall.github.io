'use strict';
const assert=require('node:assert/strict'),m=require('./models.js'),{stageFields,frozenSnapshot}=require('./snapshots.js');
const data=m.makeData(true,false),validation=m.makeData(true,false,true);
assert.equal(data.length,10);
for(const kind of ['reg','ada','logit']){
 const tr=m.build(kind,data,validation,.5,8);
 for(const round of tr.rounds){const stages=stageFields[kind].map((_,i)=>frozenSnapshot(kind,round,data,i));
  stages.forEach((s,i)=>{assert.ok(Object.isFrozen(s));assert.ok(Object.isFrozen(s.before));assert.deepEqual(s.before,round.before);assert.equal(s.stage,i);assert.deepEqual(Object.keys(s).sort(),['kind','stage','before',...stageFields[kind][i]].sort());if(s.after)assert.deepEqual(s.after,round.after);if(s.weights)assert.deepEqual(s.weights,round.weights)});
  assert.equal(stages[0].after,undefined);assert.equal(stages[0].tree,undefined);assert.equal(stages[0].targets,undefined);
  assert.ok(stages.at(-1).committed);if(kind==='ada')assert.deepEqual(stages.at(-1).afterWeights,round.afterWeights);
 }
}
assert.notDeepEqual(m.makeData(false,false).map(x=>x.t),m.makeData(true,false).map(x=>x.t));
console.log('Passed: frozen stage snapshots, delayed future values, original fitted states, and distinct data settings.');
