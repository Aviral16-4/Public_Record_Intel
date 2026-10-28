"""
Gotham Public Record Intelligence - ADVANCED (timeline, entity network, dedupe, PDF, consent-based 1:1 verification)

Install:
    pip install deepface tf-keras flask feedparser requests trafilatura spacy rapidfuzz python-dateutil
    python -m spacy download en_core_web_sm

Run:
    python app.py        ->  open http://127.0.0.1:5000

Output = reported mentions from public news, NOT verified criminal records.
Use only with the subject's consent / a lawful purpose.
"""
import re, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import feedparser
import requests
from dateutil import parser as dtparse
from flask import Flask, jsonify, request, render_template_string
from rapidfuzz import fuzz

try:
    import trafilatura
except ImportError:
    trafilatura = None
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

app = Flask(__name__)
from collections import Counter


def dedupe(items):
    """Cluster the same story reported by many outlets."""
    out = []
    for r in items:
        for o in out:
            if fuzz.token_set_ratio(r["title"], o["title"]) >= 80:
                o["duplicates"] += 1
                break
        else:
            r["duplicates"] = 1
            out.append(r)
    return out


def extract_entities(text):
    if not nlp:
        return []
    ents = {(e.text.strip(), e.label_) for e in nlp(text[:5000]).ents if e.label_ in ("PERSON", "ORG", "GPE")}
    return sorted(ents)

# ------------------------------------------------------------------ NLP config
CATEGORIES = {
    "Conviction / Sentence": r"\b(convicted|sentenced|jailed|imprisoned|life term|rigorous imprisonment|found guilty)\b",
    "Arrest / Charge": r"\b(arrested|booked|charge[- ]?sheet(ed)?|charged with|remanded|detained|FIR (registered|filed))\b",
    "Fine / Penalty": r"\b(fined|penalty|penalised|penalized|ordered to pay|imposed a fine)\b",
    "Complaint / Allegation": r"\b(complaint|alleged|allegation|accused|petition|lawsuit|sued|case filed)\b",
    "Fraud / Financial": r"\b(fraud|scam|money laundering|embezzle\w*|cheating|forgery|bribery|corruption)\b",
    "Acquittal / Cleared": r"\b(acquitted|cleared of|discharged|exonerated|quashed|bail granted)\b",
}
SEVERITY = {"Conviction / Sentence": 5, "Arrest / Charge": 4, "Fraud / Financial": 4,
            "Fine / Penalty": 3, "Complaint / Allegation": 2, "Acquittal / Cleared": 1}
MONEY = re.compile(r"(?:₹|Rs\.?|INR|\$|USD|€)\s?[\d,]+(?:\.\d+)?(?:\s?(?:lakh|crore|million|billion|k))?", re.I)
TERM = re.compile(r"\b\d+\s?(?:years?|months?)\b(?:\s+(?:of\s+)?(?:rigorous\s+)?imprisonment|\s+in jail)?", re.I)
AGE_PAT = re.compile(r"\b(\d{2})[- ]year[- ]old\b|\baged\s+(\d{2})\b|\bage\s+(\d{2})\b", re.I)


# ------------------------------------------------------------------ collection
def google_news(name, start, end):
    q = (f'"{name}" (arrested OR convicted OR fraud OR court OR fine OR police OR complaint) '
         f'after:{start:%Y-%m-%d} before:{end:%Y-%m-%d}')
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-IN&gl=IN&ceid=IN:en"
    for e in feedparser.parse(url).entries:
        yield {"title": e.get("title", ""), "url": e.get("link", ""),
               "published": e.get("published", ""), "source": e.get("source", {}).get("title", ""),
               "summary": re.sub(r"<[^>]+>", " ", e.get("summary", ""))}


def collect(name, years):
    end, seen, out = datetime.now(timezone.utc), set(), []
    for i in range(years * 4):  # quarterly windows get past the ~100 result cap
        e = end - timedelta(days=91 * i)
        for item in google_news(name, e - timedelta(days=91), e):
            if item["url"] not in seen:
                seen.add(item["url"]); out.append(item)
        time.sleep(0.5)
    return out


def fetch_text(url, fallback):
    if not trafilatura:
        return fallback
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        return trafilatura.extract(r.text) or fallback
    except Exception:
        return fallback


# ------------------------------------------------------------------ analysis
def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def identity_score(name, age, context, text):
    """Name-collision guard: name match + NER + age + context corroboration."""
    score, signals = 0.0, []
    if fuzz.partial_ratio(name.lower(), text.lower()) >= 95:
        score += 0.35; signals.append("full name present")
    if nlp:
        persons = [e.text for e in nlp(text[:5000]).ents if e.label_ == "PERSON"]
        if any(fuzz.token_sort_ratio(name.lower(), p.lower()) > 90 for p in persons):
            score += 0.15; signals.append("NER person match")
    for c in context:
        if c.lower() in text.lower():
            score += 0.2; signals.append(f"context: {c}")
    if age:
        ages = [int(g) for m in AGE_PAT.findall(text) for g in m if g]
        if any(abs(a - age) <= 1 for a in ages):
            score += 0.3; signals.append(f"age {age} matches")
        elif ages:
            score -= 0.3; signals.append(f"age conflict (article says {sorted(set(ages))})")
    return max(0.0, min(score, 1.0)), signals


def analyze(item, name, age, context):
    text = fetch_text(item["url"], f'{item["title"]}. {item["summary"]}')
    conf, signals = identity_score(name, age, context, text)
    last = name.split()[-1].lower()
    pool = [s for s in sentences(text) if last in s.lower()] or sentences(text)[:15]
    cats, hits = set(), []
    for s in pool:
        for cat, pat in CATEGORIES.items():
            if re.search(pat, s, re.I):
                cats.add(cat); hits.append(s[:300])
    try:
        pub = dtparse.parse(item["published"]).date().isoformat()
    except Exception:
        pub = item["published"]
    return {
        "title": item["title"], "url": item["url"], "published": pub, "source": item["source"],
        "categories": sorted(cats), "severity": max([SEVERITY[c] for c in cats], default=0),
        "amounts": sorted(set(MONEY.findall(text)))[:5], "terms": sorted(set(TERM.findall(text)))[:5],
        "evidence": list(dict.fromkeys(hits))[:3],
        "confidence": round(conf, 2), "signals": signals, "entities": extract_entities(text),
    }


# ------------------------------------------------------------------ API
@app.post("/api/search")
def api_search():
    d = request.get_json(force=True)
    name = (d.get("name") or "").strip()
    if len(name.split()) < 2:
        return jsonify(error="Enter the full name (first and last)."), 400
    if not d.get("consent"):
        return jsonify(error="You must confirm consent / lawful purpose."), 400
    try:
        age = int(d["age"]) if d.get("age") else None
    except ValueError:
        return jsonify(error="Age must be a number."), 400
    context = [c.strip() for c in (d.get("context") or "").split(",") if c.strip()]
    years = max(1, min(int(d.get("years") or 5), 10))
    min_conf = float(d.get("min_conf") or 0.4)

    raw = collect(name, years)[:80]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda i: analyze(i, name, age, context), raw))
    flagged = [r for r in results if r["categories"]]
    relevant = sorted([r for r in flagged if r["confidence"] >= min_conf],
                      key=lambda r: (r["severity"], r["published"]), reverse=True)
    review = [r for r in flagged if r["confidence"] < min_conf][:15]
    relevant = dedupe(relevant)
    tl = {}
    for r in relevant:
        try:
            q = f'{r["published"][:4]}-Q{(int(r["published"][5:7]) - 1) // 3 + 1}'
        except ValueError:
            continue
        slot = tl.setdefault(q, {"period": q, "count": 0, "max_sev": 0})
        slot["count"] += 1
        slot["max_sev"] = max(slot["max_sev"], r["severity"])
    ents = Counter((t, l) for r in relevant for t, l in r["entities"]
                   if fuzz.partial_ratio(name.lower(), t.lower()) < 85)
    top_ents = [{"text": t, "label": l, "count": c} for (t, l), c in ents.most_common(14)]
    by_cat = {c: sum(c in r["categories"] for r in relevant) for c in CATEGORIES}
    return jsonify(timeline=sorted(tl.values(), key=lambda x: x["period"]), entities=top_ents,
                   subject=name, age=age, years=years, scanned=len(raw),
                   relevant=relevant, review=review, by_category=by_cat,
                   max_severity=max([r["severity"] for r in relevant], default=0))


# ------------------------------------------------------------------ UI
@app.post("/api/verify")
def api_verify():
    """Consent-based 1:1 check: do two photos supplied by the user show the same person?
    Not an identification/search tool. Images are processed locally and deleted immediately."""
    import os, tempfile
    try:
        from deepface import DeepFace
    except ImportError:
        return jsonify(error="Install first: pip install deepface tf-keras"), 501
    a, b = request.files.get("selfie"), request.files.get("idphoto")
    if not (a and b):
        return jsonify(error="Upload both photos."), 400
    if request.form.get("consent") != "true":
        return jsonify(error="Confirm the person in both photos has consented."), 400
    paths = []
    try:
        for f in (a, b):
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            f.save(t.name); t.close(); paths.append(t.name)
        r = DeepFace.verify(paths[0], paths[1], model_name="Facenet", enforce_detection=True)
        return jsonify(verified=bool(r["verified"]), distance=round(r["distance"], 3),
                       threshold=round(r["threshold"], 3))
    except Exception as e:
        return jsonify(error=f"Could not compare: {str(e)[:150]}"), 400
    finally:
        for p in paths:
            os.unlink(p)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gotham Records Intelligence</title>
<style>
:root{--bg:#07080a;--panel:#111317;--panel2:#171a1f;--line:#262a31;--text:#d9dce1;--mute:#8a909b;--bat:#f5c518;--red:#e04545;--orange:#e8883a;--green:#4caf7d}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(ellipse at top,#14161b 0%,var(--bg) 60%) fixed;color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;min-height:100vh}
header{text-align:center;padding:34px 16px 10px}
.logo{width:110px;filter:drop-shadow(0 0 14px rgba(245,197,24,.45))}
h1{font-family:Impact,"Arial Narrow",sans-serif;letter-spacing:6px;font-weight:400;font-size:2rem;margin:8px 0 2px;color:var(--bat);text-transform:uppercase}
.tag{color:var(--mute);letter-spacing:3px;font-size:.75rem;text-transform:uppercase}
main{max-width:920px;margin:0 auto;padding:18px 16px 60px}
.card{background:var(--panel);border:1px solid var(--line);border-top:2px solid var(--bat);border-radius:6px;padding:20px;box-shadow:0 10px 40px rgba(0,0,0,.5)}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
@media(max-width:640px){.grid,.grid2{grid-template-columns:1fr}}
label{display:block;font-size:.72rem;letter-spacing:2px;text-transform:uppercase;color:var(--mute);margin-bottom:5px}
input[type=text],input[type=number],select{width:100%;padding:11px 12px;background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:4px;font-size:.95rem}
input:focus,select:focus{outline:none;border-color:var(--bat);box-shadow:0 0 0 2px rgba(245,197,24,.15)}
.consent{display:flex;gap:10px;align-items:flex-start;margin-top:16px;font-size:.82rem;color:var(--mute)}
.consent input{margin-top:3px;accent-color:var(--bat)}
button{margin-top:18px;width:100%;padding:13px;background:var(--bat);color:#0a0a0a;border:0;border-radius:4px;font-family:Impact,sans-serif;letter-spacing:4px;font-size:1.05rem;cursor:pointer;text-transform:uppercase}
button:hover{filter:brightness(1.1)}button:disabled{opacity:.5;cursor:wait}
.err{color:var(--red);margin-top:12px;font-size:.9rem}
#loader{display:none;text-align:center;padding:40px;color:var(--bat);letter-spacing:3px;text-transform:uppercase;font-size:.8rem}
.spin{width:38px;height:38px;border:3px solid var(--line);border-top-color:var(--bat);border-radius:50%;margin:0 auto 14px;animation:s 1s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:22px 0 10px}
.stat{background:var(--panel);border:1px solid var(--line);padding:12px;border-radius:4px;text-align:center}
.stat b{display:block;font-size:1.5rem;color:var(--bat)}.stat span{font-size:.68rem;letter-spacing:2px;color:var(--mute);text-transform:uppercase}
h2{font-family:Impact,sans-serif;letter-spacing:4px;font-weight:400;color:var(--bat);font-size:1.15rem;margin:26px 0 10px;text-transform:uppercase}
.item{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--mute);border-radius:4px;padding:14px;margin-bottom:12px}
.item.s5,.item.s4{border-left-color:var(--red)}.item.s3,.item.s2{border-left-color:var(--orange)}.item.s1{border-left-color:var(--green)}
.item a{color:var(--text);text-decoration:none;font-weight:600}.item a:hover{color:var(--bat)}
.meta{font-size:.78rem;color:var(--mute);margin:4px 0 8px}
.pill{display:inline-block;font-size:.68rem;letter-spacing:1px;text-transform:uppercase;padding:3px 8px;border-radius:20px;background:#22262d;border:1px solid var(--line);margin:0 6px 6px 0}
.pill.cat{border-color:var(--bat);color:var(--bat)}
.ev{font-size:.85rem;color:#b5bac4;border-left:2px solid var(--line);padding-left:10px;margin:6px 0}
.conf{font-size:.75rem;color:var(--mute)}
.note{font-size:.78rem;color:var(--mute);background:var(--panel2);border:1px dashed var(--line);padding:12px;border-radius:4px;margin-top:22px}
@media print{body{background:#fff;color:#000}.card,#vcard,.dl,#loader,.logo{display:none}.item,.stat{background:#fff;border-color:#999}.item a,.ev,.meta{color:#000}}
.dl+.dl{margin-left:8px}
.dl{width:auto;padding:8px 16px;margin:0;font-size:.8rem;letter-spacing:2px;background:transparent;color:var(--bat);border:1px solid var(--bat)}
</style></head><body>
<header>
<svg class="logo" viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg"><ellipse cx="100" cy="55" rx="94" ry="50" fill="#f5c518"/><ellipse cx="100" cy="55" rx="88" ry="44" fill="none" stroke="#0a0a0a" stroke-width="2"/>
<path fill="#0a0a0a" d="M100 30c-6 0-9 5-12 5-7 0-14-6-24-6 7 6 9 14 8 20 4-3 9-4 14-1 3 2 5 6 8 8l3-6 3 6c3-2 5-6 8-8 5-3 10-2 14 1-1-6 1-14 8-20-10 0-17 6-24 6-3 0-6-5-12-5z" transform="translate(0,8)"/>
<path fill="#0a0a0a" d="M92 38l3-9 5 5 5-5 3 9z"/></svg>
<h1>Gotham Records Intelligence</h1>
<div class="tag">Public source aggregation &middot; NLP analysis</div>
</header>
<main>
<div class="card">
 <div class="grid">
  <div><label>Full name</label><input type="text" id="name" placeholder="e.g. Firstname Lastname"></div>
  <div><label>Age</label><input type="number" id="age" min="10" max="110" placeholder="e.g. 34"></div>
 </div>
 <div class="grid2">
  <div><label>Context (comma separated)</label><input type="text" id="context" placeholder="city, employer, father's name"></div>
  <div><label>Years to scan</label><select id="years"><option>1</option><option>2</option><option>3</option><option>4</option><option selected>5</option><option>7</option><option>10</option></select></div>
 </div>
 <div class="consent"><input type="checkbox" id="consent"><span>I have the subject's consent or another lawful purpose for this search, and I understand results are unverified media mentions, not official records.</span></div>
 <button id="go">Search the Archives</button>
 <div class="err" id="err"></div>
</div>
<div class="card" id="vcard" style="margin-top:16px">
 <label>Consent-based identity verification (1:1 selfie vs ID photo)</label>
 <div class="grid2"><div><label>Selfie</label><input type="file" id="v1" accept="image/*"></div>
 <div><label>ID photo</label><input type="file" id="v2" accept="image/*"></div></div>
 <div class="consent"><input type="checkbox" id="vc"><span>The person in both photos has consented. Photos are compared locally and deleted immediately.</span></div>
 <button id="vgo">Verify Match</button><div id="vres" style="margin-top:10px;color:var(--bat)"></div>
</div>
<div id="loader"><div class="spin"></div>Scanning public sources&hellip; this can take up to a minute</div>
<div id="out"></div>
</main>
<script>
const $=id=>document.getElementById(id);let last=null;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function card(r){
 return `<div class="item s${r.severity}"><a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.title)}</a>
 <div class="meta">${esc(r.source)} &middot; ${esc(r.published)}${r.duplicates>1?' &middot; '+r.duplicates+' similar reports':''}</div>
 ${r.categories.map(c=>`<span class="pill cat">${esc(c)}</span>`).join('')}
 ${r.amounts.map(a=>`<span class="pill">Amount: ${esc(a)}</span>`).join('')}
 ${r.terms.map(a=>`<span class="pill">Term: ${esc(a)}</span>`).join('')}
 ${r.evidence.map(e=>`<div class="ev">${esc(e)}</div>`).join('')}
 <div class="conf">Identity confidence: ${Math.round(r.confidence*100)}% &mdash; ${esc(r.signals.join('; ')||'no corroborating signals')}</div></div>`;
}
function charts(d){
 let h='';
 if(d.timeline.length){const mx=Math.max(...d.timeline.map(t=>t.count)),W=Math.max(d.timeline.length*46,300),col=['#555','#4caf7d','#e8883a','#e8883a','#e04545','#e04545'];
  h+=`<h2>Incident timeline</h2><div style="overflow-x:auto"><svg width="${W}" height="140">`+d.timeline.map((t,i)=>{const bh=Math.round(t.count/mx*90);
   return `<rect x="${i*46+8}" y="${105-bh}" width="30" height="${bh}" fill="${col[t.max_sev]}"/><text x="${i*46+23}" y="${100-bh}" fill="#d9dce1" font-size="11" text-anchor="middle">${t.count}</text><text x="${i*46+23}" y="122" fill="#8a909b" font-size="9" text-anchor="middle">${esc(t.period)}</text>`}).join('')+`</svg></div>`;}
 if(d.entities.length){const n=d.entities.length,cx=230,cy=170,mc=d.entities[0].count,colr={PERSON:'#f5c518',ORG:'#e8883a',GPE:'#4caf7d'};
  h+=`<h2>Entity network</h2><svg viewBox="0 0 460 340" style="width:100%;max-width:460px">`+d.entities.map((e,i)=>{const a=2*Math.PI*i/n,x=cx+170*Math.cos(a),y=cy+125*Math.sin(a),r=5+8*e.count/mc;
   return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#262a31"/><circle cx="${x}" cy="${y}" r="${r}" fill="${colr[e.label]||'#888'}"/><text x="${x}" y="${y-r-3}" fill="#d9dce1" font-size="10" text-anchor="middle">${esc(e.text.slice(0,18))} (${e.count})</text>`}).join('')+`<circle cx="${cx}" cy="${cy}" r="14" fill="#f5c518"/><text x="${cx}" y="${cy+28}" fill="#f5c518" font-size="11" text-anchor="middle">SUBJECT</text></svg><div class="meta">People (yellow), organizations (orange), places (green)</div>`;}
 return h;
}
$('vgo').onclick=async()=>{const f=new FormData();f.append('selfie',$('v1').files[0]||'');f.append('idphoto',$('v2').files[0]||'');f.append('consent',$('vc').checked);
 $('vres').textContent='Comparing...';const r=await fetch('/api/verify',{method:'POST',body:f});const d=await r.json();
 $('vres').textContent=!r.ok?d.error:(d.verified?'MATCH':'NO MATCH')+' (distance '+d.distance+' / threshold '+d.threshold+')';};
function render(d){
 const cats=Object.entries(d.by_category).filter(([,v])=>v>0);
 $('out').innerHTML=`<div class="stats">
  <div class="stat"><b>${d.scanned}</b><span>Articles scanned</span></div>
  <div class="stat"><b>${d.relevant.length}</b><span>Relevant findings</span></div>
  <div class="stat"><b>${d.max_severity}/5</b><span>Max severity</span></div>
  <div class="stat"><b>${d.review.length}</b><span>Needs review</span></div></div>
  <div>${cats.map(([c,v])=>`<span class="pill cat">${esc(c)}: ${v}</span>`).join('')}</div>
  ${charts(d)}<h2>Findings &mdash; ${esc(d.subject)} (last ${d.years} yrs)</h2>
  ${d.relevant.map(card).join('')||'<div class="note">No high-confidence incident reports found. This does not prove a clean record.</div>'}
  ${d.review.length?`<h2>Low confidence &mdash; verify manually</h2>${d.review.map(card).join('')}`:''}
  <button class="dl" id="dl">Download JSON report</button><button class="dl" onclick="window.print()">Save as PDF</button>
  <div class="note">Results are automated extractions from public news. Name collisions and misreporting are possible. Verify against official sources (court records, police) before drawing any conclusions or taking any action about a person.</div>`;
 $('dl').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(last,null,2)],{type:'application/json'}));a.download='report.json';a.click();};
}
$('go').onclick=async()=>{
 $('err').textContent='';$('out').innerHTML='';$('loader').style.display='block';$('go').disabled=true;
 try{
  const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
   name:$('name').value,age:$('age').value,context:$('context').value,years:$('years').value,consent:$('consent').checked})});
  const d=await r.json();
  if(!r.ok)$('err').textContent=d.error||'Something went wrong.';else{last=d;render(d);}
 }catch(e){$('err').textContent='Request failed: '+e.message;}
 $('loader').style.display='none';$('go').disabled=false;
};
</script></body></html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    app.run(debug=False, port=5000)