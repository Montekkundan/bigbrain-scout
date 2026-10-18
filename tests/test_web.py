"""Small DOM-stub regressions for request invalidation and verification recovery."""

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r'''
const fs = require("fs"), vm = require("vm"), assert = require("assert/strict");
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    disabled:false, hidden:false, textContent:"", className:"", href:"", tabIndex:0,
    events:{}, attributes:{}, children:[], value:"",
    classList:{values:new Set(),add(v){this.values.add(v)},remove(v){this.values.delete(v)}},
    addEventListener(name,handler){this.events[name]=handler},
    setAttribute(name,value){this.attributes[name]=value},
    removeAttribute(name){delete this.attributes[name];if(name==="href")this.href=""},
    append(child){this.children.push(child)},replaceChildren(){this.children=[]}
  });
  return elements.get(id);
}
const replies = [];
const context = vm.createContext({document:{getElementById:element,createElement:()=>element(Symbol())},
  fetch:async()=>{const reply=replies.shift();return {ok:reply.ok,json:async()=>reply.data}},
  atob:s=>Buffer.from(s,"base64").toString("binary"),console});
// Do not run asynchronous configuration bootstrap in this isolated unit test.
const source = fs.readFileSync("bigbrain_scout/web/app.js","utf8").replace(/start\(\);\s*$/, "");
vm.runInContext(source,context);
vm.runInContext("config={csrf_token:'test'};capsule={id:'test-capsule'};checked={token:'old'};enableDownload()",context);
(async()=>{
  assert.equal(element("download").href,"/api/capsules/test-capsule/download");
  replies.push({ok:false,data:{error:"Another local job is running"}});
  await element("verify").events.click();
  assert.equal(element("download").href,"");
  assert.equal(element("download").attributes["aria-disabled"],"true");
  replies.push({ok:true,data:{verified:true,warning:"Local-only result"}});
  await element("verify").events.click();
  assert.equal(element("download").href,"/api/capsules/test-capsule/download");
  assert.equal(element("download").attributes["aria-disabled"],"false");
  assert.equal(element("download").classList.values.has("disabled"),false);
  assert.equal(element("download").tabIndex,0);
  vm.runInContext("invalidate()",context);
  assert.equal(element("build").disabled,true);
  assert.equal(element("verify").disabled,true);
  assert.equal(element("download").href,"");
  assert.equal(element("plan-details").hidden,true);
  assert.equal(element("manifest-details").hidden,true);
  vm.runInContext(`config={csrf_token:'test',sources:[{id:'section-3954',label:'BigBrain section #3954',description:'Fixed source',defaults:{scale:'4um',roi_mm:[[7.6,9.04,7.6],[8,9.08,8]],max_decoded_bytes:100,max_estimated_working_bytes:200},scales:[{key:'4um',label:'4 µm'}]}]}`, context);
  element("source").value="section-3954";
  vm.runInContext("sourceChanged()",context);
  assert.equal(element("lo-y").value,9.04);
  assert.equal(element("hi-y").value,9.08);
  assert.equal(element("build").textContent,"Fetch & build capsule");
  assert.equal(element("build").disabled,true);
  const canvas=element("preview");
  canvas.getContext=()=>({createImageData:()=>({data:new Uint8Array(4)}),putImageData(){}});
  vm.runInContext("preview({width:1,height:1,pixels_base64:'AA=='},'real_remote_data','section-3954')",context);
  assert.match(element("preview-note").textContent,/#3954/);
  assert.doesNotMatch(element("preview-note").textContent,/#3905/);
  console.log("verification-retry and input-invalidation regressions passed");
})().catch(error=>{console.error(error);process.exitCode=1});
'''


class BrowserLogicTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is optional for the browser-logic regression")
    def test_verification_retry_and_input_invalidation(self):
        result = subprocess.run(["node", "-e", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("regressions passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
