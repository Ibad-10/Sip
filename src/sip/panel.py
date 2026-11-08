"""Local control panel (stdlib only): live camera feeds, map, state, safety, events, and buttons
for live edits during the demo. Served on http://127.0.0.1:8765 by sip.app --panel."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sip control panel</title>
<style>
:root{--bg:#0f1216;--card:#171b21;--line:#262c35;--ink:#e8ecf1;--mut:#8b96a5;--acc:#4fb3ff;--ok:#3ecf8e;--warn:#ffb020;--bad:#ff5d5d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}
header h1{font-size:16px;margin:0;font-weight:650}#state{padding:4px 10px;border-radius:999px;background:#1e2a36;color:var(--acc);font-weight:600}
#safety{padding:4px 10px;border-radius:999px;font-weight:600}.proceed{background:#173126;color:var(--ok)}.pause{background:#33290f;color:var(--warn)}
.retreat,.ask,.halt{background:#3a1717;color:var(--bad)}#net{margin-left:auto;color:var(--mut)}
main{display:grid;gap:12px;padding:12px;grid-template-columns:repeat(12,1fr)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px;min-width:0}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:0 0 8px}
.cams{grid-column:span 8;display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.cams figure{margin:0}.cams img{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:8px;background:#000;display:block}
.cams figcaption{color:var(--mut);font-size:12px;margin-top:4px}
.map{grid-column:span 4}canvas{width:100%;aspect-ratio:4/3;background:#0b0e12;border-radius:8px;display:block}
.ctl{grid-column:span 5}.log{grid-column:span 7}
.btns{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
button{background:#202733;color:var(--ink);border:1px solid #2e3746;border-radius:8px;padding:8px 10px;font:inherit;cursor:pointer}
button:hover{border-color:var(--acc)}button.say{background:#13273a}button.danger{background:#3a1717;border-color:#5a2424}
input{flex:1;min-width:0;background:#0b0e12;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px}
.row{display:flex;gap:6px}#events{height:230px;overflow:auto;font:12px/1.5 ui-monospace,Consolas,monospace;margin:0;white-space:pre-wrap}
.k{color:var(--mut)}.alert{color:var(--bad)}.safetyk{color:var(--warn)}.plan{color:var(--acc)}.say{color:var(--ok)}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-top:8px}.stat{background:#11151a;border-radius:8px;padding:6px 8px}
.stat b{display:block;font-size:16px}.stat span{color:var(--mut);font-size:11px}
#alert{display:none;grid-column:span 12;background:#3a1717;border:1px solid #5a2424;color:#ffd5d5}
@media (max-width:900px){.cams,.map,.ctl,.log{grid-column:span 12}.cams{grid-template-columns:1fr 1fr}}
</style></head><body>
<header><h1>Sip &middot; home care robot</h1><span id="state">-</span><span id="safety">-</span><span id="net">-</span></header>
<main>
<div class="card" id="alert"></div>
<section class="card cams"><figure><img id="c_head" alt="Robot head camera"><figcaption>Robot sees (head camera)</figcaption></figure>
<figure><img id="c_wrist" alt="Robot wrist camera"><figcaption>Robot sees (wrist camera)</figcaption></figure>
<figure><img id="c_person" alt="Person's view"><figcaption>Person sees</figcaption></figure></section>
<section class="card map"><h2>Map &middot; route &middot; lidar &middot; memory</h2><canvas id="map" width="640" height="480"></canvas>
<div class="stats"><div class="stat"><b id="dist">-</b><span>straw to mouth</span></div><div class="stat"><b id="hz">-</b><span>control loop</span></div>
<div class="stat"><b id="water">-</b><span>water in glass</span></div><div class="stat"><b id="t">-</b><span>sim time</span></div></div></section>
<section class="card ctl"><h2>Person says</h2><div class="btns">
<button class="say" data-c="I'm thirsty, can I have some water?">Water please</button><button class="say" data-c="ready">Ready</button>
<button class="say" data-c="that's enough">That's enough</button><button class="say danger" data-c="stop">Stop</button><button class="say" data-c="go">Go</button></div>
<div class="row"><input id="txt" placeholder="Type what the person says" aria-label="What the person says"><button id="send">Say</button></div>
<h2 style="margin-top:12px">Live changes</h2><div class="btns">
<button data-c="/chair">Drop chair in hall</button><button data-c="/glass">Move glass to bedroom</button><button data-c="/turn">Turn head away</button>
<button data-c="/hand">Raise / lower hand</button><button data-c="/mouth">Close / open mouth</button><button data-c="/bed">Person to bed</button></div>
<h2 style="margin-top:12px">View</h2><div class="btns"><button data-c="/lidar">Lidar</button><button data-c="/path">Route</button><button data-c="/reset">Reset</button></div></section>
<section class="card log"><h2>What it's thinking</h2><pre id="events"></pre></section>
</main>
<script>
const $=id=>document.getElementById(id);
function cmd(c){fetch('/cmd',{method:'POST',body:c})}
document.querySelectorAll('button[data-c]').forEach(b=>b.onclick=()=>cmd(b.dataset.c));
$('send').onclick=()=>{if($('txt').value){cmd($('txt').value);$('txt').value=''}};
$('txt').onkeydown=e=>{if(e.key==='Enter')$('send').click()};
let n=0;function cams(){n++;for(const k of ['head','wrist','person'])$('c_'+k).src='/frame/'+k+'.jpg?'+n}
setInterval(cams,300);
const ctx=$('map').getContext('2d');let occ=null;
function W(x){return (x+4)/8*640}function H(y){return 480-(y+3)/6*480}
function draw(s){ctx.fillStyle='#0b0e12';ctx.fillRect(0,0,640,480);
 if(s.occ){occ=s.occ}if(occ){ctx.fillStyle='#3a4250';const g=occ;for(let i=0;i<g.nx;i++)for(let j=0;j<g.ny;j++)if(g.d[i*g.ny+j]==='1')ctx.fillRect(i*640/g.nx,480-(j+1)*480/g.ny,640/g.nx+.5,480/g.ny+.5)}
 ctx.fillStyle='rgba(255,90,70,.9)';for(const p of s.lidar||[])ctx.fillRect(W(p[0])-1.5,H(p[1])-1.5,3,3);
 if(s.path&&s.path.length){ctx.strokeStyle='#3ecf8e';ctx.lineWidth=3;ctx.beginPath();s.path.forEach((p,i)=>i?ctx.lineTo(W(p[0]),H(p[1])):ctx.moveTo(W(p[0]),H(p[1])));ctx.stroke()}
 for(const m of s.memory||[]){ctx.fillStyle=`rgba(255,210,0,${.3+.7*m.c})`;ctx.beginPath();ctx.arc(W(m.x),H(m.y),7,0,7);ctx.fill();ctx.fillStyle='#ffd34d';ctx.font='12px system-ui';ctx.fillText(m.label+' '+Math.round(m.c*100)+'%',W(m.x)+9,H(m.y)+4)}
 if(s.person){ctx.fillStyle='#c58aff';ctx.beginPath();ctx.arc(W(s.person[0]),H(s.person[1]),8,0,7);ctx.fill()}
 const r=s.robot;if(r){ctx.save();ctx.translate(W(r[0]),H(r[1]));ctx.rotate(-r[2]);ctx.fillStyle='#4fb3ff';ctx.beginPath();ctx.arc(0,0,13,0,7);ctx.fill();ctx.fillStyle='#fff';ctx.fillRect(0,-2,15,4);ctx.restore()}}
async function poll(){try{const s=await (await fetch('/state')).json();
 $('state').textContent=s.state;const d=s.decision||{};$('safety').textContent=d.action?d.action+' - '+d.reason:'-';$('safety').className=d.action||'';
 $('net').textContent=s.online?'online (Gemini available)':'offline - all on device';
 $('dist').textContent=s.dist<1?(s.dist*100).toFixed(1)+' cm':'-';$('hz').textContent=s.hz.toFixed(0)+' Hz';
 $('water').textContent=Math.round(s.water*100)+'%';$('t').textContent=s.t.toFixed(1)+' s';
 const ev=$('events');const atBottom=ev.scrollHeight-ev.scrollTop-ev.clientHeight<20;
 ev.innerHTML=s.events.map(e=>`<span class="k">${e.t.toFixed(1)}s</span> <span class="${e.kind==='safety'?'safetyk':e.kind}">[${e.kind}]</span> ${e.text.replace(/</g,'&lt;')}`).join('\n');
 if(atBottom)ev.scrollTop=ev.scrollHeight;
 if(s.alert){$('alert').style.display='block';$('alert').innerHTML='<b>ALERT for carer:</b> '+s.alert.replace(/</g,'&lt;')+'<br><img style="max-width:100%;margin-top:8px;border-radius:8px" src="/frame/alert.jpg?'+Math.floor(s.t)+'" alt="Evidence snapshot">'}else $('alert').style.display='none';
 draw(s)}catch(e){}}
setInterval(poll,250);
</script></body></html>"""


class Panel:
    def __init__(self, on_command, port: int = 8765):
        self.on_command = on_command
        self.frames: dict[str, bytes] = {}
        self.state: dict = {}
        self.lock = threading.Lock()
        panel = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if self.path.startswith("/frame/"):
                    name = self.path.split("/")[2].split(".")[0]
                    with panel.lock:
                        img = panel.frames.get(name)
                    return self._send(200, img, "image/jpeg") if img else self._send(204, b"", "text/plain")
                if self.path.startswith("/state"):
                    with panel.lock:
                        body = json.dumps(panel.state).encode()
                    return self._send(200, body, "application/json")
                self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path == "/cmd":
                    n = int(self.headers.get("Content-Length", "0") or 0)
                    text = self.rfile.read(min(n, 2000)).decode("utf-8", "replace")
                    panel.on_command(text)
                    return self._send(204, b"", "text/plain")
                self._send(404, b"not found", "text/plain")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.url = f"http://127.0.0.1:{port}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def set_frame(self, name: str, jpg: bytes) -> None:
        with self.lock:
            self.frames[name] = jpg

    def set_state(self, state: dict) -> None:
        with self.lock:
            self.state = state

    def close(self) -> None:
        self.server.shutdown()
