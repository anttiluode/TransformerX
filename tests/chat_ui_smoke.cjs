/* Optional DOM/HTTP smoke: npm install --prefix /tmp/chat-test jsdom
 * NODE_PATH=/tmp/chat-test/node_modules node tests/chat_ui_smoke.cjs
 * Start the test-only server below in a second terminal first:
 * python -c "import sys,tempfile;sys.path.insert(0,'tests');from test_chat_app import TestEngine;from transformerx.chat import ChatService,make_server;from transformerx.chat_store import ChatStore;make_server(ChatService(ChatStore(tempfile.mkdtemp()),TestEngine(),context_tokens=2000),7865).serve_forever()"
 * Inference is simulated. This tests UI behavior, not model quality or rendering.
 */
const {JSDOM}=require('jsdom');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const base='http://127.0.0.1:7865';
async function main(){
  const html=fs.readFileSync('transformerx/chat_ui.html','utf8');
  let available=true;
  const dom=new JSDOM(html,{url:base,runScripts:'dangerously',beforeParse(window){
    window.fetch=async(path,options={})=>{
      if(path==='/api/status') return {ok:true,json:async()=>({model:'test-qwen',device:'cpu',context_tokens:2000,loop_available:available})};
      const response=await fetch(base+path,{...options,headers:{...options.headers,Origin:base}});
      return response;
    };
    window.confirm=()=>true;
  }});
  const w=dom.window, d=w.document;
  async function wait(test){for(let i=0;i<200;i++){if(test())return;await new Promise(r=>setTimeout(r,10));}throw Error('UI did not finish: '+d.querySelector('#error').textContent);}
  await wait(()=>!d.querySelector('#send').disabled);
  d.querySelector('#new').click();await wait(()=>!d.querySelector('#new').disabled);
  d.querySelector('#notes').value='Our project is Cedar.';d.querySelector('#save-notes').click();
  await wait(()=>!d.querySelector('#save-notes').disabled);
  d.querySelector('#input').value='Hello <img src=x onerror="window.injected=1">';
  d.querySelector('#compose').dispatchEvent(new w.Event('submit',{cancelable:true}));
  await wait(()=>!d.querySelector('#send').disabled);
  assert.equal(d.querySelectorAll('.message').length,2,d.querySelector('#error').textContent);
  assert.equal(d.querySelectorAll('.message img').length,0);
  assert.equal(w.injected,undefined);
  // Reproduce a server restart without the controller while this tab stays open.
  d.querySelector('#loop').checked=true;available=false;d.querySelector('#reload').click();
  await wait(()=>!d.querySelector('#reload').disabled);
  assert.equal(d.querySelector('#loop').disabled,true);
  assert.equal(d.querySelector('#loop').checked,false,'Unavailable controller must clear stale checked toggle');
  assert.equal(d.querySelector('#notes').value,'Our project is Cedar.');
  assert.equal(d.querySelectorAll('.message').length,2);
  d.querySelector('#new').click();await wait(()=>!d.querySelector('#new').disabled);
  assert.equal(d.querySelectorAll('.message').length,0);
  assert.equal(d.querySelector('#notes').value,'');
  const old=[...d.querySelectorAll('.session')].find(b=>b.textContent.includes('Hello'));
  old.click();await wait(()=>!d.querySelector('#send').disabled);
  assert.equal(d.querySelectorAll('.message').length,2);
  d.querySelector('#delete').click();await wait(()=>!d.querySelector('#delete').disabled);
  assert.equal(d.querySelectorAll('.message').length,0);
  assert.equal(d.querySelector('#error').textContent,'');
  dom.window.close();
  console.log('PASS: DOM+HTTP send, notes, reload, loop-availability restart, session switch, delete, inert HTML. Simulated inference.');
}
main().catch(e=>{console.error(e);process.exit(1)});
