#!/usr/bin/env python3
"""Local HTML viewer for question CSV and retrieval CSV. Python standard library only."""
import argparse
import ast
import csv
import html
import json
from pathlib import Path


def load_pairs(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    first = [x.strip().lower() for x in rows[0]]
    if len(first) >= 2 and first[0] in ('id', 'question_id', 'query_id', 'номер', '№'):
        rows = rows[1:]
    return [(r[0].strip(), r[1]) for r in rows if len(r) >= 2]


def parse_chunks(raw):
    try:
        value = ast.literal_eval(raw)
        if not isinstance(value, (list, tuple)):
            return []
        out = []
        for item in value:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                out.append({'source': str(item[0]), 'text': str(item[1])})
        return out
    except (SyntaxError, ValueError, TypeError, MemoryError):
        return [{'source': 'Не удалось разобрать список чанков', 'text': raw}]


def main():
    p = argparse.ArgumentParser(description='Generate a local RAG retrieval evaluation HTML page')
    p.add_argument('questions', type=Path, help='questions.csv: id, question')
    p.add_argument('results', type=Path, help='results.csv: id, list of (source, text)')
    p.add_argument('-o', '--output', type=Path, default=Path('rag_review.html'))
    a = p.parse_args()
    questions = load_pairs(a.questions)
    results = dict(load_pairs(a.results))
    data = [{'id': qid, 'question': question, 'chunks': parse_chunks(results.get(qid, '[]')),
             'missing': qid not in results} for qid, question in questions]
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = r'''<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RAG Review</title>
<style>
:root{font-family:system-ui,sans-serif;color:#e8edf5;background:#101521}*{box-sizing:border-box}body{margin:0}header{padding:20px 5%;background:#1a2232;border-bottom:1px solid #354158}h1{margin:0 0 8px;font-size:24px}main{max-width:1100px;margin:24px auto;padding:0 18px}.toolbar,.actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}button,select,input,textarea{border:1px solid #4b5d78;background:#202b3e;color:#f3f5fa;border-radius:8px;padding:10px 12px;font:inherit}button{cursor:pointer}button:hover{background:#32415a}input[type=search]{min-width:260px;flex:1}.card{background:#1b2434;border:1px solid #38455a;border-radius:13px;padding:20px;margin:14px 0}.question{font-size:21px;line-height:1.5;white-space:pre-wrap}.muted{color:#aab8cc}.rank{display:inline-block;background:#30435d;border-radius:6px;padding:4px 9px;margin-right:8px}details{background:#182231;border:1px solid #3c4c62;border-radius:9px;margin:10px 0;padding:13px}summary{cursor:pointer;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;line-height:1.5}textarea{width:100%;min-height:80px}.selected{outline:2px solid #76baef}.pass.selected{outline-color:#5cdc9a}.partial.selected{outline-color:#f7c46e}.fail.selected{outline-color:#fa8181}.status{font-weight:bold}small{color:#aab8cc}.count{margin-left:auto}
</style></head><body><header><h1>RAG Review · проверка Retrieval</h1><div id="stats" class="muted"></div></header><main>
<div class="toolbar"><input id="search" type="search" placeholder="Поиск по ID или тексту вопроса"><select id="filter"><option value="all">Все вопросы</option><option value="unrated">Без оценки</option><option value="PASS">PASS</option><option value="PARTIAL">PARTIAL</option><option value="FAIL">FAIL</option></select><button id="export">Экспорт оценок CSV</button></div>
<div class="toolbar"><button id="prev">← Предыдущий</button><strong id="counter"></strong><button id="next">Следующий →</button></div>
<section class="card"><div class="muted" id="qid"></div><div class="question" id="question"></div></section>
<section class="card"><h2>Найденные чанки</h2><div id="chunks"></div></section>
<section class="card"><h2>Оценка поиска</h2><div class="actions"><button class="pass" data-rating="PASS">PASS</button><button class="partial" data-rating="PARTIAL">PARTIAL</button><button class="fail" data-rating="FAIL">FAIL</button><button data-rating="">Сбросить оценку</button></div><textarea id="note" placeholder="Комментарий (необязательно)"></textarea><p><small>Оценки сохраняются в этом браузере. Для переноса/резервной копии используй экспорт CSV.</small></p></section>
</main><script>
const data=__PAYLOAD__;const key='rag_review_v1_'+location.pathname;let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}let index=0;let visible=[];
const $=s=>document.querySelector(s);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function save(){try{localStorage.setItem(key,JSON.stringify(state))}catch(e){alert('Браузер не смог сохранить оценку. Экспортируй CSV.')};stats()}
function stats(){let counts={PASS:0,PARTIAL:0,FAIL:0};Object.values(state).forEach(v=>{if(v.rating in counts)counts[v.rating]++});$('#stats').textContent=`Вопросов: ${data.length} · PASS: ${counts.PASS} · PARTIAL: ${counts.PARTIAL} · FAIL: ${counts.FAIL} · Не оценено: ${data.length-counts.PASS-counts.PARTIAL-counts.FAIL}`}
function refresh(){let old=visible[index]?.id;let q=$('#search').value.toLowerCase();let filter=$('#filter').value;visible=data.filter(d=>(d.id.toLowerCase().includes(q)||d.question.toLowerCase().includes(q))&&(filter==='all'||(filter==='unrated'?!state[d.id]?.rating:state[d.id]?.rating===filter)));index=Math.max(0,visible.findIndex(d=>d.id===old));draw()}
function draw(){const d=visible[index];$('#counter').textContent=visible.length?`${index+1} / ${visible.length}`:'0 / 0';$('#prev').disabled=index<=0;$('#next').disabled=index>=visible.length-1;if(!d){$('#qid').textContent='';$('#question').textContent='Нет вопросов';$('#chunks').innerHTML='';$('#note').value='';return}$('#qid').textContent=`Вопрос №${d.id}`;$('#question').textContent=d.question;$('#chunks').innerHTML=d.missing?'<p>Нет результатов для этого ID</p>':(d.chunks.length?d.chunks.map((c,i)=>`<details ${i===0?'open':''}><summary><span class="rank">#${i+1}</span>${esc(c.source)}</summary><pre>${esc(c.text)}</pre></details>`).join(''):'<p>Чанки не найдены</p>');$('#note').value=state[d.id]?.note||'';document.querySelectorAll('[data-rating]').forEach(b=>b.classList.toggle('selected',b.dataset.rating===state[d.id]?.rating));}
function move(n){index=Math.min(Math.max(index+n,0),visible.length-1);draw();window.scrollTo({top:0,behavior:'smooth'})}
$('#search').addEventListener('input',refresh);$('#filter').addEventListener('change',refresh);$('#prev').onclick=()=>move(-1);$('#next').onclick=()=>move(1);
document.querySelectorAll('[data-rating]').forEach(b=>b.onclick=()=>{const d=visible[index];if(!d)return;state[d.id]={...state[d.id],rating:b.dataset.rating,note:$('#note').value};save();if($('#filter').value!=='all')refresh();else draw()});
$('#note').addEventListener('input',()=>{const d=visible[index];if(!d)return;state[d.id]={...state[d.id],note:$('#note').value};save()});
function quoteCSV(x){return '"'+String(x??'').replace(/"/g,'""')+'"'}
$('#export').onclick=()=>{const rows=[['id','question','rating','comment'],...data.map(d=>[d.id,d.question,state[d.id]?.rating||'',state[d.id]?.note||''])];const content='\ufeff'+rows.map(r=>r.map(quoteCSV).join(',')).join('\r\n');const blob=new Blob([content],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='rag_evaluation.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)});stats();refresh();
</script></body></html>'''
    a.output.write_text(page.replace('__PAYLOAD__', payload), encoding='utf-8')
    print(f'Готово: {a.output.resolve()} (вопросов: {len(data)})')


if __name__ == '__main__':
    main()
