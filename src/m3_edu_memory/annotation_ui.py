from __future__ import annotations


def annotation_ui_html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>手写数学教师标注台</title>
  <style>
    :root { color-scheme: light; --ink:#1f2937; --muted:#64748b; --line:#dbe3ec;
      --blue:#2563eb; --green:#15803d; --red:#dc2626; --panel:#f8fafc; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:"Microsoft YaHei",system-ui,sans-serif; color:var(--ink); background:#eef2f7; }
    header { background:#fff; border-bottom:1px solid var(--line); padding:16px 24px; display:flex;
      gap:18px; align-items:center; flex-wrap:wrap; position:sticky; top:0; z-index:5; }
    h1 { font-size:20px; margin:0 auto 0 0; }
    label { font-size:14px; color:var(--muted); }
    select,input,textarea,button { font:inherit; }
    select,input,textarea { border:1px solid #bcc8d6; border-radius:8px; padding:9px 10px; background:#fff; }
    button { border:0; border-radius:8px; padding:10px 16px; cursor:pointer; font-weight:600; }
    button.primary { background:var(--green); color:#fff; }
    button.secondary { background:var(--blue); color:#fff; }
    button.ghost { background:#e8eef7; color:var(--ink); }
    button:disabled { opacity:.45; cursor:not-allowed; }
    main { display:grid; grid-template-columns:minmax(520px,1.45fr) minmax(360px,.75fr); gap:18px;
      max-width:1500px; margin:18px auto; padding:0 18px 30px; }
    .card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:18px; box-shadow:0 2px 8px #1e293b0a; }
    .image-wrap { position:relative; display:inline-block; max-width:100%; background:#f1f5f9; line-height:0; }
    #attemptImage { display:block; max-width:100%; max-height:72vh; object-fit:contain; }
    #bboxCanvas { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; }
    .question { white-space:pre-wrap; line-height:1.55; padding:12px; background:var(--panel); border-radius:8px; margin-bottom:14px; }
    .field { display:grid; gap:6px; margin-bottom:13px; }
    .field input,.field textarea,.field select { width:100%; }
    textarea { resize:vertical; min-height:78px; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .actions,.navigation { display:flex; gap:9px; align-items:center; flex-wrap:wrap; }
    .actions { margin-top:18px; }
    .navigation { justify-content:center; margin-top:14px; }
    .badge { display:inline-block; border-radius:999px; padding:4px 10px; background:#e2e8f0; font-size:13px; }
    .progress { font-size:14px; color:var(--muted); }
    .hint { color:var(--muted); font-size:13px; line-height:1.5; }
    #message { min-height:24px; margin-top:10px; font-size:14px; }
    #message.error { color:var(--red); } #message.ok { color:var(--green); }
    #bboxText { font-family:Consolas,monospace; }
    .disabled { opacity:.5; pointer-events:none; }
    @media(max-width:900px){ main{grid-template-columns:1fr}.image-wrap{display:block} }
  </style>
</head>
<body>
<header>
  <h1>手写数学教师标注台</h1>
  <label>评测集 <input id="suite" value="fermat-vlm-pilot-100" size="24"></label>
  <label>查看范围
    <select id="filter">
      <option value="unannotated">待标注</option><option value="draft">草稿</option>
      <option value="approved">已批准</option><option value="rejected">已拒绝</option>
    </select>
  </label>
  <button class="ghost" id="load">载入</button>
  <span class="progress" id="progress">尚未载入</span>
</header>
<main>
  <section class="card">
    <div class="question" id="question">请载入评测集。</div>
    <div class="image-wrap" id="imageWrap">
      <img id="attemptImage" alt="学生手写数学作答">
      <canvas id="bboxCanvas"></canvas>
    </div>
    <p class="hint">在图片上按住鼠标拖动，框选第一处导致答案错误的区域。重新拖动即可替换。</p>
    <div class="navigation">
      <button class="ghost" id="previous">上一题</button>
      <span id="position" class="badge">0 / 0</span>
      <button class="ghost" id="next">下一题</button>
    </div>
  </section>
  <aside class="card">
    <div class="field"><label>标注教师</label><input id="annotator" value="teacher-1"></div>
    <div class="field"><label>这份作答是否存在数学错误？</label>
      <select id="hasError"><option value="">尚未判断</option><option value="true">有错误</option><option value="false">没有错误</option></select>
    </div>
    <div id="errorFields">
      <div class="row">
        <div class="field"><label>首错步骤编号（从 0 开始）</label><input id="firstStep" type="number" min="0"></div>
        <div class="field"><label>错误框坐标</label><input id="bboxText" readonly placeholder="请在图片上框选"></div>
      </div>
      <div class="field"><label>知识点（用逗号分隔）</label><input id="knowledge" placeholder="例如：chain rule, derivative"></div>
      <div class="field"><label>正确最终答案</label><textarea id="answer" placeholder="填写经过核对的最终答案"></textarea></div>
    </div>
    <div class="field"><label>教师备注</label><textarea id="notes" placeholder="记录判断依据、难辨字符或复核说明"></textarea></div>
    <div class="actions">
      <button class="secondary" id="saveDraft">保存草稿</button>
      <button class="primary" id="approve">批准标注</button>
      <button class="ghost" id="reject">拒绝此版本</button>
    </div>
    <div id="message"></div>
    <p class="hint">批准后的标注只作为离线模型评测真值，不会覆盖模型原始输出，也不会直接写入学生记忆。</p>
  </aside>
</main>
<script>
const $ = id => document.getElementById(id);
let cases = [], index = 0, bbox = null, drawing = false, start = null;
const img = $('attemptImage'), canvas = $('bboxCanvas'), ctx = canvas.getContext('2d');
const enc = encodeURIComponent;

function setMessage(text, type='') { $('message').textContent=text; $('message').className=type; }
function current() { return cases[index] || null; }
function resizeCanvas() {
  const rect=img.getBoundingClientRect(); canvas.width=Math.max(1,Math.round(rect.width));
  canvas.height=Math.max(1,Math.round(rect.height)); drawBox();
}
function drawBox() {
  ctx.clearRect(0,0,canvas.width,canvas.height); if(!bbox) return;
  ctx.fillStyle='rgba(220,38,38,.16)'; ctx.strokeStyle='#dc2626'; ctx.lineWidth=3;
  const x=bbox[0]*canvas.width,y=bbox[1]*canvas.height,w=(bbox[2]-bbox[0])*canvas.width,h=(bbox[3]-bbox[1])*canvas.height;
  ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h);
}
function pointer(event) { const r=canvas.getBoundingClientRect(); return [Math.max(0,Math.min(r.width,event.clientX-r.left)),Math.max(0,Math.min(r.height,event.clientY-r.top))]; }
canvas.addEventListener('pointerdown',e=>{ if($('hasError').value!=='true')return; drawing=true; start=pointer(e); canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove',e=>{ if(!drawing)return; const p=pointer(e); bbox=[Math.min(start[0],p[0])/canvas.width,Math.min(start[1],p[1])/canvas.height,Math.max(start[0],p[0])/canvas.width,Math.max(start[1],p[1])/canvas.height]; showBBox(); drawBox(); });
canvas.addEventListener('pointerup',()=>{drawing=false;});
function showBBox(){ $('bboxText').value=bbox?bbox.map(v=>v.toFixed(4)).join(', '):''; }
img.addEventListener('load',resizeCanvas); window.addEventListener('resize',resizeCanvas);

async function loadProgress(){ const s=enc($('suite').value.trim()); const r=await fetch(`/v1/evaluation-suites/${s}/progress`); const d=await r.json(); if(!r.ok)throw Error(d.error||'无法获取进度'); const c=d.counts; $('progress').textContent=`待标注 ${c.unannotated} · 草稿 ${c.draft} · 已批准 ${c.approved} · 已拒绝 ${c.rejected}`; }
async function loadCases(){
  setMessage('正在载入…'); const s=enc($('suite').value.trim()),f=enc($('filter').value);
  try { const r=await fetch(`/v1/evaluation-suites/${s}/cases?annotation_status=${f}&limit=500`); const d=await r.json(); if(!r.ok)throw Error(d.error||'载入失败'); cases=d.cases; index=0; render(); await loadProgress(); setMessage(`已载入 ${cases.length} 题`,'ok'); }
  catch(e){cases=[];render();setMessage(e.message,'error');}
}
function render(){
  const c=current(); $('position').textContent=c?`${index+1} / ${cases.length}`:`0 / ${cases.length}`;
  $('previous').disabled=!c||index===0; $('next').disabled=!c||index>=cases.length-1;
  if(!c){$('question').textContent='当前范围没有题目。';img.removeAttribute('src');bbox=null;showBBox();drawBox();return;}
  $('question').textContent=`${c.attempt_id}\n\n${c.orig_q}`;
  img.src=`/v1/attempts/${enc(c.attempt_id)}/image`;
  $('hasError').value=c.has_error===true?'true':c.has_error===false?'false':'';
  $('firstStep').value=c.first_error_step ?? ''; bbox=c.error_bbox || null; showBBox();
  $('knowledge').value=(c.knowledge_points||[]).join(', '); $('answer').value=c.corrected_final_answer||''; $('notes').value=''; toggleErrorFields(); drawBox();
}
function toggleErrorFields(){ $('errorFields').classList.toggle('disabled',$('hasError').value!=='true'); }
$('hasError').addEventListener('change',()=>{ if($('hasError').value!=='true'){bbox=null;showBBox();drawBox();} toggleErrorFields(); });
$('previous').onclick=()=>{if(index>0){index--;render();}}; $('next').onclick=()=>{if(index+1<cases.length){index++;render();}};
$('load').onclick=loadCases; $('filter').onchange=loadCases;

async function submit(status){
  const c=current(); if(!c)return; const value=$('hasError').value;
  const body={annotator:$('annotator').value.trim(),status,has_error:value===''?null:value==='true',notes:$('notes').value,knowledge_points:$('knowledge').value.split(/[,，\n]/).map(x=>x.trim()).filter(Boolean),correction_steps:[]};
  if(value==='true'){body.first_error_step=$('firstStep').value===''?null:Number($('firstStep').value);body.error_bbox=bbox;body.corrected_final_answer=$('answer').value.trim();}
  setMessage('正在保存…');
  try { const s=enc($('suite').value.trim()); const r=await fetch(`/v1/evaluation-suites/${s}/annotations/${enc(c.attempt_id)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const d=await r.json(); if(!r.ok)throw Error(d.error||'保存失败'); setMessage(`已保存 ${status} 版本`,'ok'); await loadProgress(); await loadCases(); }
  catch(e){setMessage(e.message,'error');}
}
$('saveDraft').onclick=()=>submit('draft'); $('approve').onclick=()=>submit('approved'); $('reject').onclick=()=>submit('rejected');
loadCases();
</script>
</body></html>"""
