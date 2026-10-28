"""
build_dashboard.py

Turns the simulator's output files into a single self-contained web page:
    site/index.html

Reads (same folder):
    weekly_matchup_v1_8_team_summary.csv
    weekly_matchup_v1_8_matchup_win_probabilities.csv   (optional: needs fantasy_matchups.csv)
    weekly_matchup_v1_8_player_summary.csv
    weekly_matchup_v1_8_run_meta.json

Run after weekly_matchup_simulator_v3_1.py. Open site/index.html in a browser to
preview; the hourly GitHub workflow publishes the same folder to GitHub Pages.
"""

from __future__ import annotations

import json
import math
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TAG = "v1_8"
SITE_DIR = os.path.join(BASE, "site")
LEAGUE_TITLE = "League matchup forecast"


def _p(name):
    return os.path.join(BASE, f"weekly_matchup_{TAG}_{name}")


def _num(v, nd=1):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, nd)


def load_payload() -> dict:
    teams = pd.read_csv(_p("team_summary.csv"))
    players = pd.read_csv(_p("player_summary.csv"))
    meta = {}
    if os.path.exists(_p("run_meta.json")):
        with open(_p("run_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    wp = pd.read_csv(_p("matchup_win_probabilities.csv")) if os.path.exists(_p("matchup_win_probabilities.csv")) else pd.DataFrame()

    team_rows = []
    by_name = {}
    for _, r in teams.iterrows():
        name = str(r["fantasy_team"]).strip()
        row = {
            "name": name,
            "final": _num(r["mean"]),
            "p10": _num(r["p10"]),
            "p90": _num(r["p90"]),
            "toDate": _num(r.get("actual_to_date", 0.0)),
            "startsToDate": _num(r.get("starts_to_date"), 0),
            "pppToDate": _num(r.get("ppp_to_date"), 2),
            "startsRest": _num(r.get("starts_rest_mean"), 1),
            "pppRest": _num(r.get("ppp_rest_of_week"), 2),
        }
        team_rows.append(row)
        by_name[name] = row

    matchups = []
    for _, r in wp.iterrows():
        a, b = str(r["team"]).strip(), str(r["opponent"]).strip()
        matchups.append({
            "a": a, "b": b,
            "aWin": _num(r["team_win_prob"], 4), "bWin": _num(r["opponent_win_prob"], 4),
            "tie": _num(r["tie_prob"], 4),
            "margin": _num(r["mean_margin"]), "marginLo": _num(r["margin_p10"]), "marginHi": _num(r["margin_p90"]),
        })

    rosters = {}
    for _, r in players.iterrows():
        team = str(r["fantasy_team"]).strip()
        rosters.setdefault(team, []).append({
            "name": str(r["player_name"]),
            "nba": str(r["nba_team"]),
            "status": str(r["injury_status"]),
            "slot": "" if pd.isna(r["lineup_slot"]) else str(r["lineup_slot"]),
            "games": int(r["games"]) if pd.notna(r["games"]) else 0,
            "proj": _num(r["mean_week_fp_active"]),
            "lo": _num(r["p10_active"]),
            "hi": _num(r["p90_active"]),
        })
    for team in rosters:
        rosters[team].sort(key=lambda x: -(x["proj"] or 0))

    return {"meta": meta, "teams": team_rows, "matchups": matchups, "rosters": rosters,
            "title": LEAGUE_TITLE}


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#EEF0F3; --surface:#FFFFFF; --ink:#17213B; --muted:#5B6478; --line:#D5D9E1;
  --a:#2F45C5; --a-soft:#DCE1FA; --b:#E8A317; --b-soft:#FBEBC6; --flag:#B42318;
  --cond:"Barlow Condensed","Arial Narrow",Arial,sans-serif; --body:"Barlow",system-ui,sans-serif;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#10172A; --surface:#18213A; --ink:#E8ECF5; --muted:#A3ADC2; --line:#2B3654;
         --a:#7B8CF0; --a-soft:#26305A; --b:#F2B640; --b-soft:#4A3A16; --flag:#FF7B6B; }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:400 16px/1.5 var(--body)}
.wrap{max-width:1080px;margin:0 auto;padding:24px 16px 64px}
header{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:12px 24px;margin-bottom:20px}
h1{font:700 clamp(2rem,6vw,3.4rem)/1 var(--cond);letter-spacing:-.01em;margin:0}
.sub{color:var(--muted);margin:6px 0 0;font-size:.95rem}
.follow{display:flex;align-items:center;gap:8px;font-size:.9rem;color:var(--muted)}
select{font:500 1rem var(--body);color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
:focus-visible{outline:3px solid var(--a);outline-offset:2px}
h2{font:600 1.6rem/1.1 var(--cond);margin:40px 0 12px}
.grid{display:grid;grid-template-columns:1fr;gap:14px}
@media (min-width:820px){.grid{grid-template-columns:1fr 1fr}}
.match{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px}
.match.mine{border-color:var(--a);box-shadow:inset 4px 0 0 var(--a)}
.names{display:flex;justify-content:space-between;gap:12px;font:600 1.15rem/1.15 var(--cond)}
.names span:last-child{text-align:right}
.tug{position:relative;display:flex;height:46px;margin:10px 0 8px;border-radius:6px;overflow:hidden}
.tug .sa{background:var(--a);color:#fff;display:flex;align-items:center;padding-left:10px}
.tug .sb{background:var(--b);color:#1B1300;display:flex;align-items:center;justify-content:flex-end;padding-right:10px}
.tug .sa,.tug .sb{font:700 1.5rem/1 var(--cond);white-space:nowrap;min-width:0;transition:flex-basis .6s ease}
.tug .mid{position:absolute;left:50%;top:-4px;bottom:-4px;width:2px;background:var(--ink);opacity:.35}
.stats{display:grid;grid-template-columns:1fr auto 1fr;gap:2px 10px;font-size:.92rem}
.stats .k{color:var(--muted);text-align:center;font-size:.82rem}
.stats .r{text-align:right}
.verdict{margin-top:10px;font-size:.92rem;color:var(--muted)}
table{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:9px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font:600 .95rem var(--cond);color:var(--muted);background:transparent}
tr:last-child td{border-bottom:0}
tr.mine td{background:var(--a-soft)}
.tablewrap{overflow-x:auto;border-radius:10px}
.num{font-variant-numeric:tabular-nums}
details{background:var(--surface);border:1px solid var(--line);border-radius:10px;margin-bottom:10px}
summary{cursor:pointer;padding:12px 14px;font:600 1.15rem var(--cond);display:flex;justify-content:space-between;gap:12px;list-style:none}
summary::-webkit-details-marker{display:none}
summary .meta{font:400 .9rem var(--body);color:var(--muted)}
details[open] summary{border-bottom:1px solid var(--line)}
details table{border:0;border-radius:0}
.flag{color:var(--flag);font-weight:600}
.slot{color:var(--muted)}
.note{color:var(--muted);font-size:.9rem;max-width:72ch}
.empty{background:var(--surface);border:1px dashed var(--line);border-radius:10px;padding:16px;color:var(--muted)}
@media (prefers-reduced-motion: reduce){.tug .sa,.tug .sb{transition:none}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1 id="title"></h1>
      <p class="sub" id="updated"></p>
    </div>
    <label class="follow">My team
      <select id="follow"><option value="">None</option></select>
    </label>
  </header>

  <section aria-labelledby="h-match">
    <h2 id="h-match">Matchups</h2>
    <div class="grid" id="matchups"></div>
  </section>

  <section aria-labelledby="h-teams">
    <h2 id="h-teams">Teams</h2>
    <div class="tablewrap"><table id="teams"></table></div>
    <p class="note">Points per player (PPP) = fantasy points ÷ games played by players in the active lineup. "So far" uses ESPN's actual scores; "rest of week" is the simulation.</p>
  </section>

  <section aria-labelledby="h-rosters">
    <h2 id="h-rosters">Rosters</h2>
    <div id="rosters"></div>
  </section>

  <section aria-labelledby="h-how">
    <h2 id="h-how">How these numbers are made</h2>
    <p class="note" id="how"></p>
  </section>
</div>
<script id="payload" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const $ = s => document.querySelector(s);
const fmt = (v, d=0) => v === null || v === undefined ? '–' : Number(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const pct = v => v === null || v === undefined ? '–' : v >= .995 ? '>99%' : (v > 0 && v < .005) ? '<1%' : Math.round(v*100) + '%';
const side = v => v >= 0 ? `winning by ${fmt(Math.abs(v))}` : `losing by ${fmt(Math.abs(v))}`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const team = Object.fromEntries(D.teams.map(t => [t.name, t]));
let mine = '';
try { mine = localStorage.getItem('myTeam') || ''; } catch (e) {}

function header(){
  const m = D.meta || {};
  $('#title').textContent = m.matchup_period ? `Week ${m.matchup_period} forecast` : (D.title || 'Matchup forecast');
  const when = m.generated_utc ? new Date(m.generated_utc).toLocaleString(undefined,{weekday:'short',hour:'numeric',minute:'2-digit',day:'numeric',month:'short'}) : 'unknown';
  const fd = m.finished_days || [], sd = m.simulated_days || [];
  const span = a => a.length > 2 ? `${a[0]} to ${a[a.length-1]}` : a.join(' and ');
  const done = fd.length ? `${span(fd)} ${fd.length > 1 ? 'are' : 'is'} final` : 'No games are final yet';
  const left = sd.length ? `${span(sd)} simulated` : 'the week is complete';
  $('#updated').textContent = `Updated ${when}. ${done}; ${left}.`;
  $('#how').textContent = `Each run plays out the rest of the week ${fmt(m.sims)} times: every NBA game is simulated possession by possession with injuries, minutes and returns, and every fantasy team sets its best legal lineup each day. Win chance is the share of those runs each team finished ahead. Days already played use ESPN's actual scores.`;
  const sel = $('#follow');
  D.teams.map(t => t.name).sort((a,b)=>a.localeCompare(b)).forEach(n => {
    const o = document.createElement('option'); o.value = n; o.textContent = n; sel.appendChild(o);
  });
  sel.value = mine;
  sel.addEventListener('change', () => { mine = sel.value; try { localStorage.setItem('myTeam', mine); } catch (e) {} render(); });
}

function matchCard(m){
  const ta = team[m.a] || {}, tb = team[m.b] || {};
  const isMine = mine && (m.a === mine || m.b === mine);
  const aw = Math.max(m.aWin || 0, 0), bw = Math.max(m.bWin || 0, 0);
  const total = aw + bw || 1;
  const fav = aw >= bw ? m.a : m.b;
  const favP = Math.max(aw, bw);
  const marg = Math.abs(m.margin || 0);
  const lo = aw >= bw ? m.marginLo : -m.marginHi, hi = aw >= bw ? m.marginHi : -m.marginLo;
  return `<article class="match${isMine ? ' mine' : ''}" aria-label="${esc(m.a)} versus ${esc(m.b)}">
    <div class="names"><span>${esc(m.a)}</span><span>${esc(m.b)}</span></div>
    <div class="tug" role="img" aria-label="${esc(m.a)} ${pct(aw)} to win, ${esc(m.b)} ${pct(bw)}">
      <div class="sa" style="flex-basis:${aw/total*100}%">${aw >= .12 ? pct(aw) : ''}</div>
      <div class="sb" style="flex-basis:${bw/total*100}%">${bw >= .12 ? pct(bw) : ''}</div>
      <div class="mid"></div>
    </div>
    <div class="stats num">
      <span>${fmt(ta.toDate,1)}</span><span class="k">so far</span><span class="r">${fmt(tb.toDate,1)}</span>
      <span>${fmt(ta.final)}</span><span class="k">projected final</span><span class="r">${fmt(tb.final)}</span>
      <span>${fmt(ta.pppToDate,1)} / ${fmt(ta.pppRest,1)}</span><span class="k">PPP so far / rest</span><span class="r">${fmt(tb.pppToDate,1)} / ${fmt(tb.pppRest,1)}</span>
    </div>
    <p class="verdict">${esc(fav)} ${pct(favP)} to win, by about ${fmt(marg)} points. In 80% of runs they finish somewhere between ${side(lo)} and ${side(hi)}.</p>
  </article>`;
}

function render(){
  const ms = [...D.matchups].sort((x,y) => ((y.a===mine||y.b===mine) - (x.a===mine||x.b===mine)));
  $('#matchups').innerHTML = ms.length ? ms.map(matchCard).join('')
    : `<p class="empty">Matchups appear once the ESPN matchup download has run this week.</p>`;

  const rows = [...D.teams].sort((a,b) => (b.final||0) - (a.final||0));
  $('#teams').innerHTML = `<thead><tr><th scope="col">Team</th><th scope="col">Projected final</th><th scope="col">80% range</th><th scope="col">So far</th><th scope="col">Starts so far</th><th scope="col">PPP so far</th><th scope="col">Starts left</th><th scope="col">PPP rest of week</th></tr></thead><tbody>` +
    rows.map(t => `<tr class="${t.name===mine?'mine':''}"><td>${esc(t.name)}</td><td class="num">${fmt(t.final)}</td><td class="num">${fmt(t.p10)}–${fmt(t.p90)}</td><td class="num">${fmt(t.toDate,1)}</td><td class="num">${fmt(t.startsToDate)}</td><td class="num">${fmt(t.pppToDate,2)}</td><td class="num">${fmt(t.startsRest,1)}</td><td class="num">${fmt(t.pppRest,2)}</td></tr>`).join('') + `</tbody>`;

  const names = Object.keys(D.rosters).sort((a,b) => (b===mine) - (a===mine) || a.localeCompare(b));
  $('#rosters').innerHTML = names.map(n => {
    const ps = D.rosters[n];
    const hurt = ps.filter(p => p.status !== 'ACTIVE').length;
    return `<details${n===mine?' open':''}><summary><span>${esc(n)}</span><span class="meta">${ps.length} players${hurt ? `, ${hurt} injured or questionable` : ''}</span></summary>
      <div class="tablewrap"><table><thead><tr><th scope="col">Player</th><th scope="col">NBA</th><th scope="col">Games left</th><th scope="col">Projected FP</th><th scope="col">80% range</th></tr></thead><tbody>${
        ps.map(p => `<tr><td>${esc(p.name)} ${p.status!=='ACTIVE'?`<span class="flag">${esc(p.status.replace(/_/g,' ').toLowerCase())}</span>`:''} ${p.slot==='IR'?'<span class="slot">IR</span>':''}</td><td>${esc(p.nba)}</td><td class="num">${p.games}</td><td class="num">${fmt(p.proj)}</td><td class="num">${fmt(p.lo)}–${fmt(p.hi)}</td></tr>`).join('')
      }</tbody></table></div></details>`;
  }).join('');
}

header();
render();
</script>
</body>
</html>
"""


def main():
    payload = load_payload()
    os.makedirs(SITE_DIR, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("__TITLE__", payload["title"]).replace("__DATA__", data)
    out = os.path.join(SITE_DIR, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Dashboard written: {out} ({len(payload['matchups'])} matchups, {len(payload['teams'])} teams)")


if __name__ == "__main__":
    main()
