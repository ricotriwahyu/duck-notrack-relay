"""Duck.ai VQD, challenge JS, and anti-fingerprint management.

Isolates all low-level anti-bot mechanisms:
- Webdriver and navigator mask script
- window.fetch request hook for capturing SSE and rewriting outgoing payloads
- Challenge warmup and readiness checks
- Desktop Chrome user-agents and headers
"""
from __future__ import annotations

import logging
import random
from typing import Dict, Optional

logger = logging.getLogger("zexrtw.challenge")

# Real desktop Chrome UAs (avoid HeadlessChrome markers)
REAL_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
]

DEFAULT_UA = REAL_USER_AGENTS[0]


def get_random_ua() -> str:
    return random.choice(REAL_USER_AGENTS)


# Injected into every new browser context before any page script executes
CHALLENGE_INIT_SCRIPT = r"""
try {
  Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
  delete Object.getPrototypeOf(navigator).webdriver;
} catch(e){}

if (!window.chrome) {
  window.chrome = {
    app: { isInstalled: false },
    runtime: { PlatformArch: { X86_64: "x86-64" }, PlatformOs: { WIN: "win" } },
    loadTimes: function(){},
    csi: function(){}
  };
}

try {
  const fakePlugins = [
    { name: "Chrome PDF Plugin", filename: "internal-pdf-viewer", description: "Portable Document Format" },
    { name: "Chrome PDF Viewer", filename: "mhjfbmdgcfjbbpaeojofohoefgiehjai", description: "" },
    { name: "Native Client", filename: "internal-nacl-plugin", description: "" }
  ];
  Object.defineProperty(navigator, 'plugins', { get: () => fakePlugins });
  Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en', 'id'] });
} catch(e){}

try {
  const origQuery = window.navigator.permissions.query;
  window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications' ?
      Promise.resolve({ state: Notification.permission }) :
      origQuery(parameters)
  );
} catch(e){}

window.__duckai = {chunks:[], done:false, err:null, rewrite:null, rewriteApplied:false};
(function(){
  const O = window.fetch;
  window.fetch = async function(u,o){
    o = o || {};
    if(typeof u === 'string' && u.includes('/duckchat/v1/chat')){
      const D = window.__duckai;
      D.chunks = []; D.done = false; D.err = null;
      const rw = D.rewrite; D.rewrite = null;
      if(rw && typeof o.body === 'string'){
        try{
          const j = JSON.parse(o.body);
          if(rw.model) j.model = rw.model;
          if(rw.reasoningEffort) j.reasoningEffort = rw.reasoningEffort;
          if(rw.canUseTools !== undefined) j.canUseTools = rw.canUseTools;
          if(rw.toolChoice){
            j.metadata = Object.assign({}, j.metadata);
            j.metadata.toolChoice = Object.assign({}, j.metadata.toolChoice, rw.toolChoice);
          }
          o = Object.assign({}, o, {body: JSON.stringify(j)});
          D.rewriteApplied = true;
        }catch(e){}
      }
      const r = await O.call(this, u, o);
      try{
        const reader = r.clone().body.getReader();
        const dec = new TextDecoder();
        (async function pump(){
          try{
            while(true){
              const {value,done} = await reader.read();
              if(done){ D.chunks.push(dec.decode()); D.done=true; break; }
              const s = dec.decode(value,{stream:true});
              D.chunks.push(s);
              if(s.indexOf('"action":"error"') !== -1) D.err = s;
            }
          }catch(e){ D.err = String(e); D.done = true; }
        })();
      }catch(e){ D.err = String(e); D.done = true; }
      return r;
    }
    return O.apply(this, arguments);
  };
})();
"""

# Reset capture buffer for next turn
JS_RESET_CAPTURE = r"""
(rw) => { window.__duckai = {chunks:[], done:false, err:null, rewrite: rw, rewriteApplied:false}; }
"""

# Poll new streamed chunks since index n
JS_POLL_CHUNKS = r"""
(n) => {
  const D = window.__duckai || {chunks:[],done:false,err:null};
  const text = D.chunks.slice(n).join('');
  return {n: D.chunks.length, text, done: D.done, err: D.err, rewriteApplied: !!D.rewriteApplied};
}
"""
