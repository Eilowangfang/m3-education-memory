from __future__ import annotations


def review_ui_html() -> str:
    return r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>教师修订错题报告</title><style>
body{font-family:'Microsoft YaHei',Segoe UI,sans-serif;margin:0;background:#f4f6f8;color:#1f2933}
main{max-width:1180px;margin:auto;padding:24px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.panel{background:white;border:1px solid #d9e0e7;border-radius:10px;padding:18px;margin-bottom:18px}
label{display:block;font-weight:600;margin-top:12px}input,select,textarea{box-sizing:border-box;width:100%;padding:9px;border:1px solid #b8c2cc;border-radius:5px}
textarea{min-height:86px}button{margin-top:14px;padding:10px 16px;border:0;border-radius:5px;background:#155e75;color:white;cursor:pointer}
img{max-width:100%;border:1px solid #d9e0e7}.hint{color:#52606d}.ok{color:#166534}.error{color:#b91c1c}
@media(max-width:850px){.grid{grid-template-columns:1fr}}</style></head><body><main>
<h1>教师修订错题报告</h1><p class="hint">修改只覆盖查询展示和学生记忆，不改变 FERMAT ground truth，也不覆盖原始 VLM 输出。</p>
<div class="panel"><div class="grid"><div><label>Attempt ID</label><input id="attempt"></div><div><label>模型</label><input id="model"></div></div><button id="load">载入</button><p id="message"></p></div>
<div class="grid"><section class="panel"><h2>原始手写作答</h2><img id="image" alt="手写作答"></section>
<section class="panel"><h2>修改诊断</h2>
<label>教师</label><input id="reviewer" value="teacher-1">
<label>处理方式</label><select id="verdict"><option value="modified">修改 VLM 结果</option><option value="confirmed">确认 VLM 结果</option><option value="rejected">VLM 错判：原作答无错</option></select>
<label>是否有错</label><select id="hasError"><option value="">沿用 VLM</option><option value="true">有错</option><option value="false">无错</option></select>
<label>错误类型</label><select id="errorType"><option value="">沿用 VLM</option><option>conceptual</option><option>assumption</option><option>algebraic_manipulation</option><option>arithmetic</option><option>notation</option><option>transcription</option><option>omitted_step</option><option>presentation</option><option>no_actual_error</option><option>uncertain</option></select>
<label>首错步骤（从 0 开始）</label><input id="step" type="number" min="0">
<label>错误框 x1,y1,x2,y2（0–1）</label><input id="bbox" placeholder="0.1,0.2,0.8,0.5">
<label>知识点（逗号分隔）</label><input id="points">
<label>错因说明</label><textarea id="explanation"></textarea>
<label>教师备注</label><textarea id="notes"></textarea>
<button id="saveReview">保存诊断并重绘审计图</button></section></div>
<section class="panel"><h2>替换订正</h2><p class="hint">仅当 VLM 订正不正确时提交。新版本会成为查询报告中的生效版本，旧版本继续保留。</p>
<label>完整订正过程</label><textarea id="solution"></textarea><label>正确结果</label><input id="finalAnswer"><label>可验证表达式</label><input id="verify" placeholder="1/3 + 1/6 = 1/2"><button id="saveCorrection">保存人工订正版</button></section>
<section class="panel"><h2>当前生效数据</h2><pre id="output"></pre></section>
<script>
const $=id=>document.getElementById(id), q=new URLSearchParams(location.search);
$('attempt').value=q.get('attempt_id')||'';$('model').value=q.get('model')||'';
function valueOrNull(id){const v=$(id).value.trim();return v===''?null:v}
function msg(t,ok=true){$('message').className=ok?'ok':'error';$('message').textContent=t}
async function load(){try{const id=encodeURIComponent($('attempt').value.trim()),m=encodeURIComponent($('model').value.trim());const r=await fetch(`/v1/attempts/${id}?model=${m}`),d=await r.json();if(!r.ok)throw Error(d.error||'载入失败');$('image').src=`/v1/attempts/${id}/image`; $('output').textContent=JSON.stringify(d,null,2);$('points').value=(d.knowledge_points_effective||d.knowledge_points||[]).join(', ');$('explanation').value=d.error_explanation||'';$('step').value=d.first_error_step_effective??'';$('bbox').value=(d.error_bbox_effective||[]).join(',');$('solution').value=d.correction?.corrected_solution||'';$('finalAnswer').value=d.correction?.final_answer||'';msg('已载入当前模型结果和教师覆盖');}catch(e){msg(e.message,false)}}
async function saveReview(){try{const b=$('bbox').value.trim(),p=$('points').value.split(',').map(x=>x.trim()).filter(Boolean),h=$('hasError').value;const body={model:$('model').value.trim(),verdict:$('verdict').value,reviewer:$('reviewer').value.trim(),has_error:h===''?null:h==='true',error_type:valueOrNull('errorType'),first_error_step:valueOrNull('step')===null?null:Number($('step').value),error_bbox:b?b.split(',').map(Number):null,knowledge_points:p.length?p:null,error_explanation:valueOrNull('explanation'),notes:$('notes').value,render:true};const r=await fetch(`/v1/errors/${encodeURIComponent($('attempt').value.trim())}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.error||'保存失败');msg('教师诊断已生效');await load();}catch(e){msg(e.message,false)}}
async function saveCorrection(){try{const body={model:$('model').value.trim(),source:'human',created_by:$('reviewer').value.trim(),corrected_solution:$('solution').value,corrected_steps:[],final_answer:valueOrNull('finalAnswer'),verification_expression:valueOrNull('verify'),activate:true,render:true};const r=await fetch(`/v1/errors/${encodeURIComponent($('attempt').value.trim())}/correction`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.error||'保存失败');msg('人工订正版已生效');await load();}catch(e){msg(e.message,false)}}
$('load').onclick=load;$('saveReview').onclick=saveReview;$('saveCorrection').onclick=saveCorrection;if($('attempt').value&&$('model').value)load();
</script></main></body></html>"""
