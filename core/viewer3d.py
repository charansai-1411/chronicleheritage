"""Interactive 3D damage visualisation (Three.js inside a Streamlit component).

Image → 3D mapping strategy (approximate, by design)
----------------------------------------------------
A YOLO box lives in 2D image coordinates; it says nothing about depth. We use a
*manually defined spatial correspondence*:

1. Every image is tagged with the facade it shows (north / east / south / west
   elevation, or `top` for nadir satellite imagery).
2. Each GLB has a small calibration file `static/models3d/<slug>.json`
   (`north_deg`: rotation that brings the model's north to -Z; `frame`: how much
   of the image width/height the monument occupies). Defaults apply if absent.
3. We place a virtual orthographic camera looking at that facade, so the image
   plane is the facade's bounding-box face. The box centre (u, v) becomes a
   point on that plane and a ray is cast *into* the mesh; the first surface hit
   is where the marker is drawn.

This assumes a roughly frontal, monument-filling photograph, so every marker is
labelled "approximate location". Precise localisation would need camera pose
estimation (e.g. COLMAP/photogrammetry registration), which the calibration file
is designed to accept later (`camera` entries), but is not claimed here.
"""
from __future__ import annotations

import html
import json

import streamlit.components.v1 as components

from core.damage_reference import DAMAGE
from core.monuments import Monument

_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--ink:#0B1F3A;--muted:#5B6578;--line:#E3E6EC;--paper:#FBFAF7}
*{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden;font-family:'Inter Tight',system-ui,sans-serif;color:var(--ink);background:radial-gradient(120% 90% at 50% 20%,#FFFFFF 0%,#F2F1EC 70%,#E9E7E0 100%)}
#c{position:absolute;inset:0}
.hud{position:absolute;z-index:3;background:rgba(255,255,255,.86);backdrop-filter:blur(10px);border:1px solid var(--line);border-radius:14px;box-shadow:0 10px 30px -18px rgba(11,31,58,.35)}
#top{left:14px;top:14px;padding:10px 12px;display:flex;gap:8px;align-items:center;font-size:12px}
#top .mono{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);letter-spacing:.04em}
.btn{border:1px solid var(--line);background:#fff;color:var(--ink);border-radius:999px;padding:6px 12px;font:500 12px 'Inter Tight';cursor:pointer}
.btn.on{background:var(--ink);color:#fff;border-color:var(--ink)}
#legend{left:14px;bottom:14px;padding:10px 12px;font-size:11.5px;max-width:250px}
#legend b{display:block;font:500 10px 'JetBrains Mono';letter-spacing:.12em;color:var(--muted);margin-bottom:6px;text-transform:uppercase}
.row{display:flex;align-items:center;gap:8px;margin:3px 0}.sw{width:10px;height:10px;border-radius:3px;flex:none}
.cnt{margin-left:auto;font-family:'JetBrains Mono';color:var(--muted)}
#panel{right:14px;top:14px;width:290px;padding:16px;display:none;font-size:12.5px;line-height:1.5}
#panel h4{margin:2px 0 2px;font-size:16px;font-weight:600}
#panel .k{font:500 10px 'JetBrains Mono';letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-top:10px}
#panel ul{margin:4px 0 0;padding-left:16px}#panel .x{position:absolute;right:12px;top:10px;cursor:pointer;color:var(--muted)}
.pill{display:inline-block;font:500 10.5px 'JetBrains Mono';padding:3px 8px;border-radius:999px;background:#FFF4E5;color:#8A5300;border:1px solid #F3D9B1}
#note{right:14px;bottom:14px;padding:8px 12px;font-size:11px;color:var(--muted);max-width:330px}
#load{position:absolute;inset:0;display:grid;place-items:center;font:500 12px 'JetBrains Mono';color:var(--muted);letter-spacing:.1em}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div id="load">LOADING DIGITAL TWIN…</div><div id="c"></div>
<div id="top" class="hud"><span class="mono" id="cnt"></span>
  <button class="btn on" id="bDmg">Damage view</button><button class="btn" id="bNorm">Normal view</button>
  <button class="btn" id="bSpin">Auto-rotate</button></div>
<div id="legend" class="hud"></div>
<div id="panel" class="hud"><span class="x" id="px">✕</span><div id="pc"></div></div>
<div id="note" class="hud"></div>
<script type="module">
import * as THREE from "three";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {RoomEnvironment} from "three/addons/environments/RoomEnvironment.js";
const CFG=__CFG__;
const el=document.getElementById('c');
const R=new THREE.WebGLRenderer({antialias:true,alpha:true});R.setPixelRatio(Math.min(devicePixelRatio,2));
R.setSize(innerWidth,innerHeight);R.shadowMap.enabled=true;R.toneMapping=THREE.ACESFilmicToneMapping;R.toneMappingExposure=.78;R.outputColorSpace=THREE.SRGBColorSpace;el.appendChild(R.domElement);
const scene=new THREE.Scene();const pm=new THREE.PMREMGenerator(R);scene.environment=pm.fromScene(new RoomEnvironment(),.04).texture;
const cam=new THREE.PerspectiveCamera(35,innerWidth/innerHeight,.01,1e6);
const sun=new THREE.DirectionalLight(0xffffff,1.6);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);scene.add(sun);
scene.add(new THREE.HemisphereLight(0xffffff,0xd8d2c4,.6));
const ctl=new OrbitControls(cam,R.domElement);ctl.enableDamping=true;ctl.screenSpacePanning=true;
const world=new THREE.Group();scene.add(world);const markers=new THREE.Group();scene.add(markers);
const legend=document.getElementById('legend'),panel=document.getElementById('panel'),pc=document.getElementById('pc');
document.getElementById('cnt').textContent=CFG.dets.length+' MARKER'+(CFG.dets.length===1?'':'S')+' · DRAG · SCROLL · RIGHT-DRAG';
document.getElementById('note').innerHTML=CFG.note;
// legend
const counts={};CFG.dets.forEach(d=>counts[d.label]=(counts[d.label]||0)+1);
legend.innerHTML='<b>Damage legend</b>'+Object.entries(CFG.classes).map(([k,v])=>`<div class="row"><span class="sw" style="background:${v.color}"></span>${k}<span class="cnt">${counts[k]||0}</span></div>`).join('');
function geomFor(kind,r){switch(kind){case'diamond':return new THREE.OctahedronGeometry(r);case'square':return new THREE.BoxGeometry(r*1.4,r*1.4,r*1.4);
 case'x':case'cross':return new THREE.TetrahedronGeometry(r*1.2);case'triangle-up':return new THREE.ConeGeometry(r,r*2,4);case'circle-open':return new THREE.TorusGeometry(r*.8,r*.3,10,24);default:return new THREE.SphereGeometry(r,24,24);}}
function placeholder(){const g=new THREE.Group();const m=new THREE.MeshStandardMaterial({color:0xD9D2C3,roughness:.9});
 const base=new THREE.Mesh(new THREE.BoxGeometry(30,4,30),m);base.position.y=2;g.add(base);
 const body=new THREE.Mesh(new THREE.BoxGeometry(22,18,22),m);body.position.y=13;g.add(body);
 const dome=new THREE.Mesh(new THREE.SphereGeometry(8,32,16,0,Math.PI*2,0,Math.PI/2),m);dome.position.y=22;g.add(dome);
 [[-10,-10],[10,-10],[-10,10],[10,10]].forEach(([x,z])=>{const t=new THREE.Mesh(new THREE.CylinderGeometry(1.6,1.8,30,16),m);t.position.set(x,15,z);g.add(t)});
 g.traverse(o=>{if(o.isMesh){o.castShadow=o.receiveShadow=true}});return g;}
let box,size,center,meshes=[];
function frame(root){
 root.rotation.y=THREE.MathUtils.degToRad(CFG.calib.north_deg||0);world.add(root);world.updateMatrixWorld(true);
 box=new THREE.Box3().setFromObject(world);size=box.getSize(new THREE.Vector3());center=box.getCenter(new THREE.Vector3());
 world.traverse(o=>{if(o.isMesh){o.castShadow=o.receiveShadow=true;meshes.push(o)}});
 const S=Math.max(size.x,size.y,size.z);
 const ground=new THREE.Mesh(new THREE.CircleGeometry(S*1.4,96),new THREE.ShadowMaterial({opacity:.18}));ground.rotation.x=-Math.PI/2;ground.position.y=box.min.y;ground.receiveShadow=true;scene.add(ground);
 const ring=new THREE.Mesh(new THREE.RingGeometry(S*.92,S*.925,128),new THREE.MeshBasicMaterial({color:0x0B1F3A,transparent:true,opacity:.25}));ring.rotation.x=-Math.PI/2;ring.position.y=box.min.y+S*.001;scene.add(ring);
 // compass
 const mk=(t,x,z)=>{const c=document.createElement('canvas');c.width=c.height=64;const g=c.getContext('2d');g.font='600 34px Inter Tight';g.fillStyle='#0B1F3A';g.textAlign='center';g.textBaseline='middle';g.fillText(t,32,34);
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true,opacity:.6}));s.scale.set(S*.07,S*.07,1);s.position.set(center.x+x,box.min.y+S*.02,center.z+z);scene.add(s)};
 mk('N',0,-S*.98);mk('S',0,S*.98);mk('E',S*.98,0);mk('W',-S*.98,0);
 sun.position.set(center.x+S,center.y+S*1.6,center.z+S*.8);sun.target.position.copy(center);scene.add(sun.target);
 const sc=sun.shadow.camera;sc.left=sc.bottom=-S;sc.right=sc.top=S;sc.near=.01;sc.far=S*6;sc.updateProjectionMatrix();
 const rad=box.getBoundingSphere(new THREE.Sphere()).radius;const half=THREE.MathUtils.degToRad(cam.fov/2);const dist=Math.max(rad/Math.sin(half)*.78,(size.y/2)/Math.tan(half)*1.7);
 const dir=new THREE.Vector3(...(CFG.calib.view||[.62,.38,.69])).normalize();ctl.target.copy(center);cam.position.copy(center).addScaledVector(dir,dist);cam.near=S/1000;cam.far=S*50;cam.updateProjectionMatrix();ctl.update();
 placeMarkers(S);document.getElementById('load').remove();}
const DIRS={north:[0,0,1],south:[0,0,-1],east:[-1,0,0],west:[1,0,0],top:[0,-1,0]}; // viewing direction (camera looks along this)
const rc=new THREE.Raycaster();const pulses=[];
function placeMarkers(S){
 const fr=CFG.calib.frame||{w:1,h:1};
 CFG.dets.forEach((d,i)=>{
  const f=DIRS[d.facade]?d.facade:'south';const fwd=new THREE.Vector3(...DIRS[f]);
  const up=f==='top'?new THREE.Vector3(0,0,-1):new THREE.Vector3(0,1,0);const right=new THREE.Vector3().crossVectors(fwd,up).normalize();
  // extent of the facade plane in world units
  const wExt=Math.abs(right.x)*size.x+Math.abs(right.z)*size.z+Math.abs(right.y)*size.y;const hExt=Math.abs(up.x)*size.x+Math.abs(up.y)*size.y+Math.abs(up.z)*size.z;
  const du=(d.u-.5)/(fr.w||1),dv=(.5-d.v)/(fr.h||1);
  const p=center.clone().addScaledVector(right,du*wExt).addScaledVector(up,dv*hExt).addScaledVector(fwd,-S*2);
  rc.set(p,fwd);const hit=rc.intersectObjects(meshes,false)[0];
  const pos=hit?hit.point.clone().addScaledVector(fwd,-S*.006):center.clone().addScaledVector(right,du*wExt).addScaledVector(up,dv*hExt).addScaledVector(fwd,-((Math.abs(fwd.x)*size.x+Math.abs(fwd.y)*size.y+Math.abs(fwd.z)*size.z)/2));
  const c=CFG.classes[d.label]||{color:'#0B1F3A',marker:'circle'};const col=new THREE.Color(c.color);const r=S*.016;
  const m=new THREE.Mesh(geomFor(c.marker,r),new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:.55,roughness:.35,metalness:.1}));
  m.position.copy(pos);m.userData={i,d,hit:!!hit};markers.add(m);
  const halo=new THREE.Mesh(new THREE.SphereGeometry(r*(1.6+3*Math.sqrt(d.area||.01)),24,24),new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:.18,depthWrite:false}));
  halo.position.copy(pos);markers.add(halo);pulses.push(halo);});}
function showPanel(o){const d=o.userData.d;const c=CFG.classes[d.label]||{};
 pc.innerHTML=`<span class="pill">APPROXIMATE LOCATION</span><h4 style="margin-top:8px">${d.label}</h4>
 <div style="color:var(--muted)">Detection #${o.userData.i+1} · confidence <b style="color:var(--ink)">${(d.confidence*100).toFixed(1)}%</b> · ${d.facade} view</div>
 <div class="k">What it is</div><div>${c.what||''}</div>
 <div class="k">Possible causes (reference)</div><ul>${(c.typical_causes||[]).slice(0,4).map(x=>`<li>${x}</li>`).join('')}</ul>
 ${d.ai?`<div class="k">AI hypothesis · Gemini</div><div>${d.ai}</div>`:''}
 <div class="k">Mapping</div><div style="color:var(--muted)">${o.userData.hit?'Projected from the image onto the model surface':'No surface hit — shown on the facade plane'} (facade projection; not photogrammetric).</div>`;
 panel.style.display='block';}
document.getElementById('px').onclick=()=>panel.style.display='none';
const ray=new THREE.Raycaster(),mouse=new THREE.Vector2();
R.domElement.addEventListener('click',e=>{const b=R.domElement.getBoundingClientRect();mouse.set(((e.clientX-b.left)/b.width)*2-1,-((e.clientY-b.top)/b.height)*2+1);
 ray.setFromCamera(mouse,cam);const h=ray.intersectObjects(markers.children.filter(m=>m.userData.d),false)[0];if(h&&markers.visible)showPanel(h.object);});
const bD=document.getElementById('bDmg'),bN=document.getElementById('bNorm'),bS=document.getElementById('bSpin');
bD.onclick=()=>{markers.visible=true;bD.classList.add('on');bN.classList.remove('on')};
bN.onclick=()=>{markers.visible=false;panel.style.display='none';bN.classList.add('on');bD.classList.remove('on')};
bS.onclick=()=>{ctl.autoRotate=!ctl.autoRotate;bS.classList.toggle('on')};ctl.autoRotateSpeed=.8;
if(CFG.glb){new GLTFLoader().load(CFG.glb,g=>frame(g.scene),undefined,e=>{document.getElementById('load').textContent='COULD NOT LOAD 3D MODEL — SHOWING PLACEHOLDER';frame(placeholder())});}
else frame(placeholder());
const clk=new THREE.Clock();(function loop(){requestAnimationFrame(loop);const t=clk.getElapsedTime();pulses.forEach((p,i)=>{const s=1+.18*Math.sin(t*2.4+i);p.scale.setScalar(s)});ctl.update();R.render(scene,cam)})();
addEventListener('resize',()=>{R.setSize(innerWidth,innerHeight);cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix()});
</script></body></html>"""


def model_info(m: Monument) -> dict:
    """Provenance stored in the GLB's asset.extras (title, author, license, source)."""
    if not m.has_3d:
        return {}
    import struct
    try:
        with open(m.glb_file, "rb") as f:
            head = f.read(20)
            n = struct.unpack("<I", head[12:16])[0]
            js = json.loads(f.read(n))
        return js.get("asset", {}).get("extras", {}) or {}
    except Exception:
        return {}


def provenance_html(m: Monument) -> str:
    info = model_info(m)
    if not info:
        return ""
    if info.get("source") == "parametric":
        return ("<b>Parametric approximation</b> — massing model generated from documented dimensions "
                "(scripts/build_models.py); proportions and layout, not a survey or scan.")
    src = info.get("source", "")
    return (f"<b>3D model:</b> “{html.escape(info.get('title', m.name))}” by "
            f"{html.escape(info.get('author', 'unknown').split(' (')[0])} · {html.escape(info.get('license', '').split(' (')[0])}"
            + (f' · <a href="{html.escape(src)}" target="_blank">source</a>' if src.startswith("http") else ""))


def calibration(m: Monument) -> dict:
    p = m.glb_file.with_suffix(".json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"north_deg": 0, "frame": {"w": 1.0, "h": 1.0}}


def render(m: Monument, dets: list[dict], height: int = 620, ai_notes: dict | None = None) -> None:
    """dets: [{label, confidence, u, v, facade, area}] — area = box area / image area."""
    ai_notes = ai_notes or {}
    payload = [{**{k: d[k] for k in ("label", "confidence", "u", "v")}, "facade": d.get("facade") or "south",
                "area": d.get("area", 0.01), "ai": ai_notes.get(d["label"], "")} for d in dets]
    if m.has_3d:
        note = (provenance_html(m) + "<br>Markers are <b>approximate</b>: each 2D box is projected "
                "onto the model from the facade the photo shows.")
    else:
        note = (f"<b>No 3D model for {m.name} yet</b> — showing placeholder geometry. Add "
                f"<code>static/models3d/{m.slug}.glb</code> to enable the digital twin.")
    cfg = {"glb": m.glb_url, "dets": payload, "calib": calibration(m), "note": note,
           "classes": {k: {kk: v[kk] for kk in ("color", "marker", "what", "typical_causes")} for k, v in DAMAGE.items()}}
    components.html(_HTML.replace("__CFG__", json.dumps(cfg)), height=height, scrolling=False)
