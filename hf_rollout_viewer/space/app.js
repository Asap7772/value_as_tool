'use strict';
(() => {
  const CONFIG = window.ROLLOUT_CONFIG || {};
  const query = new URLSearchParams(location.search);
  const DATA = (query.get('data') || CONFIG.dataBase || '../data').replace(/\/$/, '');
  const MODES = ['solution_summary', 'thinking_summary', 'solutions'];
  const MODE_LABELS = {solutions: 'Full solutions', solution_summary: 'Solution summaries', thinking_summary: 'Solution+thinking summaries'};
  const ROLE_LABELS = {
    direct: 'Direct solver', generator: 'Generator', reviser: 'Reviser', verifier: 'Verifier', subagent: 'Subagent',
    value_solver: 'Value-tool solver', value_verifier: 'Value verifier', planner: 'Planner', worker: 'Worker',
    reviewer: 'Reviewer',
  };
  const STATUS_GLYPH = {A: '✓', L: '↻', C: '', P: '⚠', B: '⚠', X: '⚠', F: '⚠'};
  const LONG_TEXT = 60000;
  const VIEWS = ['rollouts', 'summary', 'analysis'];
  const state = {view: 'rollouts', meta: null, bench: null, index: {}, rows: [], problemId: null, problem: null, open: null, pinned: null};
  const cache = new Map();

  const $ = id => document.getElementById(id);
  const node = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  };
  const fmt = n => (n === null || n === undefined ? '–' : Number(n).toLocaleString('en-US'));
  const kfmt = n => (n === null || n === undefined ? '–' : n >= 1000 ? `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k` : String(n));
  const pct = x => `${Math.round(100 * x)}%`;
  const button = (text, handler, cls = 'button') => {
    const b = node('button', cls, text);
    b.type = 'button';
    b.addEventListener('click', handler);
    return b;
  };
  const badge = (text, kind = '') => node('span', `badge ${kind}`, text);
  const lazy = (title, make, {open = false, cls = ''} = {}) => {
    const d = node('details', cls);
    const s = node('summary');
    if (title instanceof Node) s.append(title); else s.textContent = title;
    d.append(s);
    const fill = () => {
      if (d.dataset.loaded) return;
      d.dataset.loaded = '1';
      try { d.append(make()); } catch (error) { d.append(node('p', 'error', `Could not render: ${error.message}`)); }
    };
    d.addEventListener('toggle', () => { if (d.open) fill(); });
    if (open) { d.open = true; fill(); }
    return d;
  };

  async function fetchJSON(path) {
    if (cache.has(path)) {
      const hit = cache.get(path);
      cache.delete(path);
      cache.set(path, hit);
      return hit;
    }
    while (cache.size >= 48) {
      const oldest = [...cache.keys()].find(key => key.startsWith('rollouts/'));
      if (!oldest) break;
      cache.delete(oldest);
    }
    const promise = (async () => {
      const response = await fetch(`${DATA}/${path}`);
      if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      let text;
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
        const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
        text = await new Response(stream).text();
      } else {
        text = new TextDecoder().decode(bytes);
      }
      return JSON.parse(text);
    })();
    cache.set(path, promise);
    promise.catch(() => cache.delete(path));
    return promise;
  }

  // Model text is untrusted: it only ever becomes text nodes; KaTeX renders math with trust disabled.
  function inline(parent, text) {
    const re = /(`[^`\n]+`|\*\*[^*\n]+\*\*)/g;
    let last = 0, m;
    while ((m = re.exec(text))) {
      parent.append(document.createTextNode(text.slice(last, m.index)));
      const token = m[0];
      parent.append(token[0] === '`' ? node('code', '', token.slice(1, -1)) : node('strong', '', token.slice(2, -2)));
      last = re.lastIndex;
    }
    parent.append(document.createTextNode(text.slice(last)));
  }

  function renderMath(el, source) {
    if (!window.renderMathInElement) return;
    const singleDollar = !/\$\s*\d[\d,.]*(?:\s+(?:and|then|in|for|per|each|to|worth|more|less|dollars?|a|an)\b|[,;]\s*\$\d)/i.test(source);
    try {
      window.renderMathInElement(el, {
        delimiters: [{left: '$$', right: '$$', display: true}, {left: '\\[', right: '\\]', display: true},
          {left: '\\(', right: '\\)', display: false}, ...(singleDollar ? [{left: '$', right: '$', display: false}] : [])],
        throwOnError: false, trust: false, strict: 'ignore', maxExpand: 1000,
      });
    } catch { /* keep the original text when math is malformed */ }
  }

  function markdown(value) {
    const out = node('div', 'prose');
    const lines = String(value || '').replace(/\r\n/g, '\n').split('\n');
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      if (/^\s*```/.test(line)) {
        const chunk = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) chunk.push(lines[i++]);
        i++;
        const pre = node('pre');
        pre.append(node('code', '', chunk.join('\n')));
        out.append(pre);
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        const h = node(`h${Math.min(heading[1].length + 2, 6)}`);
        inline(h, heading[2]);
        out.append(h);
        i++;
        continue;
      }
      if (/^\s*(?:---+|\*\*\*+)\s*$/.test(line)) { out.append(node('hr')); i++; continue; }
      if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
        const table = node('table'), head = node('thead'), body = node('tbody');
        const cells = s => s.trim().replace(/^\||\|$/g, '').split('|');
        const row = (s, tag) => { const tr = node('tr'); for (const c of cells(s)) { const td = node(tag); inline(td, c.trim()); tr.append(td); } return tr; };
        head.append(row(line, 'th'));
        i += 2;
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) body.append(row(lines[i++], 'td'));
        table.append(head, body);
        const wrap = node('div', 'table-scroll');
        wrap.append(table);
        out.append(wrap);
        continue;
      }
      if (/^\s*(?:[-*+] |\d+[.)] )/.test(line)) {
        const ordered = /^\s*\d+[.)] /.test(line), list = node(ordered ? 'ol' : 'ul');
        while (i < lines.length && /^\s*(?:[-*+] |\d+[.)] )/.test(lines[i])) {
          const li = node('li');
          const chunk = [lines[i++].replace(/^\s*(?:[-*+] |\d+[.)] )/, '')];
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*(?:[-*+] |\d+[.)] )/.test(lines[i])) chunk.push(lines[i++].trim());
          inline(li, chunk.join('\n'));
          list.append(li);
        }
        out.append(list);
        continue;
      }
      if (/^>\s?/.test(line)) {
        const quote = node('blockquote');
        inline(quote, line.replace(/^>\s?/, ''));
        out.append(quote);
        i++;
        continue;
      }
      const chunk = [line];
      i++;
      let inDisplay = (line.match(/\$\$/g) || []).length % 2 === 1;
      while (i < lines.length && (inDisplay || (lines[i].trim() && !/^(?:#{1,6}\s|\s*```|\s*[-*+]\s|\s*\d+[.)]\s|>\s)/.test(lines[i])))) {
        inDisplay = inDisplay !== ((lines[i].match(/\$\$/g) || []).length % 2 === 1);
        chunk.push(lines[i++]);
      }
      const p = node('p');
      inline(p, chunk.join('\n'));
      out.append(p);
    }
    renderMath(out, String(value || ''));
    return out;
  }

  function plainText(value) {
    const pre = node('pre', 'plain');
    pre.textContent = value || '';
    return pre;
  }

  // Long texts start as plain text so opening a 200k-character reasoning trace stays fast.
  function textBlock(value, {rendered = true} = {}) {
    const box = node('div', 'text-block');
    const long = (value || '').length > LONG_TEXT;
    let asMarkdown = rendered && !long;
    const body = node('div');
    const draw = () => body.replaceChildren(asMarkdown ? markdown(value) : plainText(value));
    const toggle = button('', () => { asMarkdown = !asMarkdown; label(); draw(); }, 'text-button tiny');
    const label = () => { toggle.textContent = asMarkdown ? 'show raw text' : 'render markdown + math'; };
    label();
    const bar = node('div', 'text-bar');
    bar.append(node('span', 'hint', `${fmt((value || '').length)} chars`), toggle);
    box.append(bar, body);
    draw();
    return box;
  }

  function armMode(armId) {
    return MODES.find(mode => armId.startsWith(`attempt_${mode}_`)) || null;
  }

  function experimentById(id) { return state.meta.experiments.find(e => e.id === id); }
  function armById(expId, armId) { return experimentById(expId)?.arms.find(a => a.id === armId); }

  function resolveMarker(key, ctx) {
    const calls = ctx.callsByIndex;
    const problem = state.problem || {};
    let m;
    if (key === 'prompt') return {label: 'Solver prompt', text: problem.solver_prompt};
    if (key === 'problem') return {label: 'Problem statement', text: problem.problem};
    if (key === 'reference') return {label: 'Reference proof', text: problem.reference};
    if (key === 'final') {
      return ctx.traj ? {label: 'Final output', text: finalText(ctx.traj)}
        : {label: 'Final output of the rollout', text: 'Each rollout\'s final output is inserted here.'};
    }
    if (key === 'evidence') {
      const mode = ctx.traj ? armMode(ctx.traj.arm) : null;
      return {label: `Evidence pack · ${MODE_LABELS[mode] || mode}`, render: () => renderPack(problem.packs?.[mode], mode)};
    }
    if ((m = key.match(/^c(\d+)$/))) {
      const call = calls.get(Number(m[1]));
      return {label: `Output of call #${m[1]} (${ROLE_LABELS[call?.role] || call?.role})`, text: call?.content};
    }
    if ((m = key.match(/^r(\d+)$/))) {
      const call = calls.get(Number(m[1]));
      return {label: `Reasoning of call #${m[1]} (${ROLE_LABELS[call?.role] || call?.role})`, text: call?.reasoning, raw: true};
    }
    if ((m = key.match(/^a(\d+)\.(.+)$/))) {
      const call = calls.get(Number(m[1]));
      const args = (call?.tool_calls || []).map(tc => tc.args).find(a => a && typeof a === 'object' && m[2] in a);
      return {label: `${m[2]} from call #${m[1]} (${ROLE_LABELS[call?.role] || call?.role})`, text: args ? String(args[m[2]]) : ''};
    }
    return {label: key, text: ''};
  }

  function markerChip(key, ctx) {
    const info = resolveMarker(key, ctx);
    const title = node('span', 'chip-label', `⟦${info.label}⟧`);
    return lazy(title, () => {
      if (info.render) return info.render();
      return textBlock(info.text || '(empty)', {rendered: !info.raw});
    }, {cls: 'chip'});
  }

  function markerText(text, ctx) {
    const box = node('div', 'marked');
    const re = /⟦([^⟧]{1,80})⟧/g;
    let last = 0, m, run = null;
    const flush = s => {
      if (!s) return;
      run = run || node('span', 'run');
      run.append(document.createTextNode(s));
    };
    while ((m = re.exec(text))) {
      flush(text.slice(last, m.index));
      if (run) { box.append(run); run = null; }
      box.append(markerChip(m[1], ctx));
      last = re.lastIndex;
    }
    flush(text.slice(last));
    if (run) box.append(run);
    return box;
  }

  function jsonView(value, ctx) {
    if (typeof value === 'string') return markerText(value, ctx);
    if (value === null || typeof value !== 'object') return node('code', 'scalar', JSON.stringify(value));
    if (Array.isArray(value)) {
      const ol = node('ol', 'json-list');
      value.forEach(item => { const li = node('li'); li.append(jsonView(item, ctx)); ol.append(li); });
      return ol;
    }
    const dl = node('dl', 'kv');
    for (const [key, item] of Object.entries(value)) {
      dl.append(node('dt', '', key));
      const dd = node('dd');
      dd.append(jsonView(item, ctx));
      dl.append(dd);
    }
    return dl;
  }

  function renderMessage(message, ctx) {
    const box = node('div', `msg msg-${message.role}`);
    box.append(node('div', 'msg-role', `${message.role}${message.name ? ` · ${message.name}` : ''}`));
    if (message.ref !== undefined) {
      const link = button(`↩ assistant response of call #${message.ref}`, () => focusCall(ctx, message.ref), 'text-button');
      box.append(link);
      return box;
    }
    if (message.json !== undefined) box.append(jsonView(message.json, ctx));
    else box.append(markerText(message.content || '', ctx));
    if (message.reasoning) box.append(lazy('reasoning_content', () => markerText(message.reasoning, ctx)));
    if (message.tool_calls) for (const tc of message.tool_calls) box.append(renderToolCall(tc, ctx, true));
    return box;
  }

  function verdictBadge(outcome) {
    return badge(outcome, `verdict-${outcome}`);
  }

  function renderToolCall(tc, ctx, compact = false) {
    const box = node('div', 'tool-call');
    const head = node('div', 'tool-head');
    head.append(node('span', 'tool-name', `🛠 ${tc.name}`));
    box.append(head);
    const args = tc.args;
    if (!args || typeof args !== 'object') {
      if (args) box.append(plainText(String(args)));
      return box;
    }
    if (tc.name === 'submit_verdict') {
      if (args.outcome) head.append(verdictBadge(args.outcome));
      if (args.success_probability !== undefined) head.append(badge(`p = ${args.success_probability}`, 'prob'));
    }
    if (tc.name === 'submit_probability' && args.success_probability !== undefined) head.append(badge(`p = ${args.success_probability}`, 'prob'));
    if (compact && Object.keys(args).length === 0) return box;
    const dl = node('dl', 'kv');
    for (const [key, value] of Object.entries(args)) {
      if (['outcome', 'success_probability'].includes(key) && ['submit_verdict', 'submit_probability'].includes(tc.name)) continue;
      dl.append(node('dt', '', key));
      const dd = node('dd');
      if (typeof value === 'string') dd.append(value.length > 300 ? textBlock(value) : markdown(value));
      else dd.append(jsonView(value, ctx));
      dl.append(dd);
    }
    if (dl.childElementCount) box.append(dl);
    return box;
  }

  function renderVerdictRecord(verdict) {
    const box = node('div', 'verdict-record');
    const head = node('div', 'tool-head');
    head.append(node('span', 'tool-name', `Verdict routed to the solver (cycle ${verdict.cycle})`), verdictBadge(verdict.verdict));
    if (verdict.success_probability !== undefined) head.append(badge(`p = ${verdict.success_probability}`, 'prob'));
    box.append(head);
    const dl = node('dl', 'kv');
    for (const key of ['fault_category', 'critique', 'rationale', 'candidate_excerpt']) {
      if (!verdict[key]) continue;
      dl.append(node('dt', '', key));
      const dd = node('dd');
      dd.append(markdown(verdict[key]));
      dl.append(dd);
    }
    box.append(dl);
    box.append(node('p', 'hint', 'This is the verdict after the harness cleaned it (field limits, and removal of spans copied only from privileged evidence). The next call\'s prompt shows the exact feedback text.'));
    return box;
  }

  function usageLine(call) {
    const u = call.usage || {};
    const parts = [`prompt ${kfmt(u.prompt)}`, `completion ${kfmt(u.completion)}`];
    if (u.reasoning) parts.push(`reasoning ${kfmt(u.reasoning)}`);
    if (call.finish) parts.push(`finish: ${call.finish}`);
    return parts.join(' · ');
  }

  function focusCall(ctx, index) {
    const el = ctx.root.querySelector(`[data-call="${index}"]`);
    if (!el) return;
    el.scrollIntoView({behavior: 'smooth', block: 'start'});
    el.classList.add('flash');
    setTimeout(() => el.classList.remove('flash'), 1200);
  }

  function renderCall(call, ctx) {
    const card = node('article', `call role-${call.role}`);
    card.dataset.call = call.i;
    const head = node('header', 'call-head');
    head.append(node('span', 'call-index', `#${call.i}`), node('span', 'call-role', ROLE_LABELS[call.role] || call.role),
      node('span', 'call-label', call.label || ''));
    head.append(node('span', 'call-usage', usageLine(call)));
    card.append(head);
    if (call.cont) {
      const note = node('p', 'hint cont');
      note.append(document.createTextNode(`Continues the conversation of `),
        button(`call #${call.cont.call}`, () => focusCall(ctx, call.cont.call), 'text-button'),
        document.createTextNode(` (its first ${call.cont.n} messages are not repeated).`));
      card.append(note);
    }
    const promptTitle = `Prompt · ${call.messages.length} message${call.messages.length === 1 ? '' : 's'}${call.tools?.length ? ` · tools: ${call.tools.join(', ')}` : ''}`;
    card.append(lazy(promptTitle, () => {
      const box = node('div', 'messages');
      for (const message of call.messages) box.append(renderMessage(message, ctx));
      return box;
    }, {cls: 'prompt'}));
    if (call.reasoning) {
      const isVerifier = ['verifier', 'value_verifier', 'reviewer'].includes(call.role);
      card.append(lazy(`Reasoning · ${fmt(call.reasoning.length)} chars`, () => textBlock(call.reasoning, {rendered: false}),
        {cls: `reasoning${isVerifier ? ' verifier-reasoning' : ''}`}));
    }
    if (call.content && call.content.trim()) {
      const section = node('section', 'response');
      section.append(node('h4', '', 'Response'));
      section.append(textBlock(call.content));
      card.append(section);
    }
    for (const tc of call.tool_calls || []) card.append(renderToolCall(tc, ctx));
    for (const verdict of ctx.verdictsByCall.get(call.i) || []) card.append(renderVerdictRecord(verdict));
    for (const estimate of ctx.valuesByVerifier.get(call.i) || []) {
      const box = node('div', 'verdict-record');
      const h = node('div', 'tool-head');
      h.append(node('span', 'tool-name', `Returned to the solver (query ${estimate.query_index})`), badge(`p = ${estimate.probability}`, 'prob'));
      box.append(h);
      if (estimate.rationale) box.append(markdown(estimate.rationale));
      card.append(box);
    }
    if (call.error) card.append(node('p', 'error', call.error));
    return card;
  }

  function finalText(traj) {
    if (!traj.final) return '';
    if (traj.final.content !== undefined) return traj.final.content;
    return (traj.calls.find(c => c.i === traj.final.call) || {}).content || '';
  }

  function flowSummary(traj) {
    const steps = [];
    for (const t of traj.transitions) {
      const [, source, action, target] = t;
      if (source === 'scheduled') continue;
      if (source === 'verifier') steps.push(`V:${action}`);
      else if (action === 'candidate') steps.push(ROLE_LABELS[source] || source);
      else if (target === 'final_output') steps.push(action === 'return' ? 'return' : action);
      else steps.push(`${source}→${target}`);
    }
    return steps.join(' → ');
  }

  function scoreBadge(score) {
    if (score === null || score === undefined) return badge('unjudged', 'score-na');
    return badge(`${score}/7`, `score score-s${score}`);
  }

  function renderRollout(traj, expId) {
    const arm = armById(expId, traj.arm) || {label: traj.arm};
    const exp = experimentById(expId);
    const root = node('div', 'rollout');
    root.dataset.key = `${expId}|${traj.arm}|${traj.seed}`;
    const ctx = {
      traj, root, callsByIndex: new Map(traj.calls.map(c => [c.i, c])),
      verdictsByCall: new Map(), valuesByVerifier: new Map(),
    };
    for (const v of traj.verdicts || []) {
      const key = Number(v.call_index);
      if (!ctx.verdictsByCall.has(key)) ctx.verdictsByCall.set(key, []);
      ctx.verdictsByCall.get(key).push(v);
    }
    for (const v of traj.value_estimates || []) {
      const key = Number(v.verifier_call_index);
      if (!ctx.valuesByVerifier.has(key)) ctx.valuesByVerifier.set(key, []);
      ctx.valuesByVerifier.get(key).push(v);
    }
    const head = node('header', 'rollout-head');
    const title = node('div');
    title.append(node('p', 'kicker', `${exp.label} · seed ${traj.seed}`), node('h3', '', arm.label));
    const badges = node('div', 'badges');
    badges.append(scoreBadge(traj.judge?.score), badge(traj.status, `status status-${traj.status}`),
      badge(`${kfmt(traj.tokens?.generated)} generated tokens`), badge(`${traj.calls.length} calls`));
    head.append(title, badges);
    root.append(head);
    if (arm.description) root.append(node('p', 'hint', arm.description));
    const flow = flowSummary(traj);
    if (flow) root.append(node('p', 'flow', flow));
    if (traj.error) root.append(node('p', 'error', traj.error));
    for (const call of traj.calls) root.append(renderCall(call, ctx));
    const final = node('section', 'final');
    final.append(node('h4', '', 'Final output → external judge'));
    final.append(textBlock(finalText(traj)));
    const judge = traj.judge;
    if (judge) {
      const jh = node('div', 'tool-head');
      jh.append(node('span', 'tool-name', `Judge (gpt-oss-20b): ${judge.judge_status}`), scoreBadge(judge.score));
      final.append(jh);
      if (judge.raw) final.append(lazy('Judge response', () => textBlock(judge.raw), {open: judge.raw.length < 4000}));
      const template = judge.template ? state.problem?.judge_templates?.[judge.template] : judge.prompt;
      if (template) final.append(lazy('Judge prompt (grading rubric)', () => markerText(template, ctx)));
      if (judge.error) final.append(node('p', 'error', String(judge.error)));
    }
    root.append(final);
    if (traj.transitions?.length) {
      root.append(lazy('Transitions', () => {
        const ol = node('ol', 'transitions');
        for (const t of traj.transitions) ol.append(node('li', '', `cycle ${t[0]}: ${t[1]} —${t[2]}→ ${t[3]}${t[4] ? ` (${t[4]})` : ''}`));
        return ol;
      }));
    }
    root.append(node('p', 'hint mono', traj.run_id));
    return root;
  }

  function renderPack(pack, mode) {
    const box = node('div', 'pack');
    if (!pack) { box.append(node('p', 'hint', 'No evidence pack for this problem.')); return box; }
    const content = pack.content || {};
    const labels = pack.labels || content.authoritative_attempt_labels || [];
    if (labels.length) {
      const list = node('div', 'labels');
      list.append(node('span', 'hint', 'Prior Direct attempts (9B, seeds 0–7): '));
      for (const label of labels) {
        const b = button(`seed ${label.seed}: ${label.correct ? '✓' : '✗'}`, () => openRollout('q9_base', 'direct', label.seed),
          `label-chip ${label.correct ? 'ok' : 'bad'}`);
        b.title = 'Open this Direct rollout';
        list.append(b);
      }
      box.append(list);
    }
    if (content.limitations) box.append(node('p', 'hint', content.limitations));
    if (typeof content.summary === 'string') box.append(textBlock(content.summary));
    if (Array.isArray(content.attempts)) {
      for (const attempt of content.attempts) {
        const seed = labels.find(l => l.attempt_id === attempt.attempt_id)?.seed;
        box.append(lazy(`Attempt${seed !== undefined ? ` seed ${seed}` : ''} · labeled ${attempt.correct ? 'correct' : 'incorrect'} · ${fmt(attempt.solution?.length)} chars`,
          () => textBlock(attempt.solution)));
      }
    }
    if (typeof content === 'string') box.append(textBlock(content));
    return box;
  }

  function bankOf(row) {
    if (!row.bank) return null;
    const pos = row.bank.filter(Boolean).length;
    return {pos, n: row.bank.length, kind: pos === 0 ? 'negative' : pos === row.bank.length ? 'positive' : 'mixed'};
  }

  function rowStats(row) {
    const stats = {};
    let wins = 0, total = 0;
    for (const [expId, arms] of Object.entries(row.cells)) {
      let w = 0, t = 0;
      for (const cells of Object.values(arms)) for (const c of cells) { t++; if (c[1] === 7) w++; }
      stats[expId] = t ? w / t : null;
      wins += w;
      total += t;
    }
    stats.all = total ? wins / total : 0;
    return stats;
  }

  function renderBenchTabs() {
    const tabs = $('bench-tabs');
    tabs.replaceChildren();
    for (const bench of state.meta.benchmarks) {
      const b = button(bench.label, () => selectBenchmark(bench.id), `tab${bench.id === state.bench ? ' active' : ''}`);
      b.setAttribute('role', 'tab');
      tabs.append(b);
    }
  }

  function shortExp(expId) {
    return {q9_base: '9B', q9_attempt: '9B-att', q27_base: '27B'}[expId] || expId;
  }

  function renderList() {
    const text = $('search').value.trim().toLowerCase();
    const bank = $('bank-filter').value;
    const sort = $('sort').value;
    let rows = (state.index[state.bench]?.problems || []).filter(row => {
      if (text && !row.id.toLowerCase().includes(text) && !row.preview.toLowerCase().includes(text)) return false;
      if (bank && bankOf(row)?.kind !== bank) return false;
      return true;
    });
    rows = rows.map(row => ({row, stats: rowStats(row)}));
    if (sort === 'hard') rows.sort((a, b) => a.stats.all - b.stats.all || a.row.id.localeCompare(b.row.id));
    else if (sort === 'easy') rows.sort((a, b) => b.stats.all - a.stats.all || a.row.id.localeCompare(b.row.id));
    state.rows = rows.map(r => r.row);
    $('list-count').textContent = `${rows.length} problem${rows.length === 1 ? '' : 's'}`;
    const list = $('problem-list');
    list.replaceChildren();
    for (const {row, stats} of rows) {
      const li = node('li');
      const b = button('', () => selectProblem(row.id), `problem-item${row.id === state.problemId ? ' active' : ''}`);
      b.dataset.id = row.id;
      const top = node('div', 'problem-top');
      top.append(node('span', 'problem-id', row.id));
      const bk = bankOf(row);
      if (bk) top.append(badge(`prior ${bk.pos}/${bk.n}`, `bank-${bk.kind}`));
      b.append(top, node('p', 'problem-preview', row.preview));
      const meta = node('div', 'problem-stats');
      for (const exp of state.meta.experiments) {
        if (stats[exp.id] === null || stats[exp.id] === undefined) continue;
        meta.append(node('span', '', `${shortExp(exp.id)} ${pct(stats[exp.id])}`));
      }
      b.append(meta);
      li.append(b);
      list.append(li);
    }
  }

  async function selectBenchmark(benchId, {problemId = null} = {}) {
    state.bench = benchId;
    renderBenchTabs();
    $('problem-list').replaceChildren(node('li', 'hint', 'Loading problems…'));
    if (!state.index[benchId]) state.index[benchId] = await fetchJSON(`index/${benchId}.json.gz`);
    renderList();
    const first = problemId && state.index[benchId].problems.some(r => r.id === problemId) ? problemId : state.rows[0]?.id;
    if (first) await selectProblem(first);
  }

  function gridFor(exp, row) {
    const wrap = node('section', 'grid-section');
    const cellsByArm = row.cells[exp.id] || {};
    const h = node('div', 'grid-head');
    h.append(node('h3', '', exp.label), node('span', 'hint', exp.description));
    wrap.append(h);
    const table = node('table', 'grid');
    const thead = node('thead');
    const tr = node('tr');
    tr.append(node('th', 'arm-col', 'Method'));
    for (const seed of exp.seeds) tr.append(node('th', '', `s${seed}`));
    tr.append(node('th', '', '7/7'), node('th', '', 'mean'));
    thead.append(tr);
    table.append(thead);
    const tbody = node('tbody');
    let group = null;
    for (const arm of exp.arms) {
      const cells = cellsByArm[arm.id];
      if (!cells) continue;
      if (exp.id === 'q9_attempt') {
        const g = arm.label.split(' · ').slice(0, 2).join(' · ');
        if (g !== group) {
          group = g;
          const gr = node('tr', 'group-row');
          const td = node('td', '', g);
          td.colSpan = exp.seeds.length + 3;
          gr.append(td);
          tbody.append(gr);
        }
      }
      const r = node('tr');
      const name = node('td', 'arm-col');
      const label = exp.id === 'q9_attempt' ? arm.label.split(' · ').slice(2).join(' · ') || arm.label : arm.label;
      name.append(node('span', 'arm-name', label));
      name.title = arm.description;
      r.append(name);
      const bySeed = new Map(cells.map(c => [c[0], c]));
      let wins = 0, sum = 0, judged = 0;
      for (const seed of exp.seeds) {
        const td = node('td');
        const c = bySeed.get(seed);
        if (c) {
          const [s, score, status, verdicts, tokens, ncalls] = c;
          if (score === 7) wins++;
          if (score !== null && score !== undefined) { sum += score; judged++; }
          const cell = button(`${score ?? '–'}${STATUS_GLYPH[status] || ''}`, () => openRollout(exp.id, arm.id, s),
            `cell score-s${score ?? 'na'}${isOpen(exp.id, arm.id, s) ? ' selected' : ''}`);
          cell.dataset.key = `${exp.id}|${arm.id}|${s}`;
          cell.title = `seed ${s} · judge ${score ?? '–'}/7 · ${state.meta.status_codes[status] || status}` +
            `${verdicts ? ` · verdicts ${verdicts}` : ''} · ${kfmt(tokens)} tokens · ${ncalls} calls`;
          td.append(cell);
        }
        r.append(td);
      }
      r.append(node('td', 'num', `${wins}/${cells.length}`), node('td', 'num', judged ? (sum / judged).toFixed(1) : '–'));
      tbody.append(r);
    }
    table.append(tbody);
    const scroll = node('div', 'table-scroll');
    scroll.append(table);
    wrap.append(scroll);
    return wrap;
  }

  function isOpen(expId, armId, seed) {
    return [state.open, state.pinned].some(o => o && o.expId === expId && o.armId === armId && o.seed === seed);
  }

  function markCells() {
    for (const el of document.querySelectorAll('.cell')) {
      const [expId, armId, seed] = el.dataset.key.split('|');
      el.classList.toggle('selected', isOpen(expId, armId, Number(seed)));
    }
  }

  async function selectProblem(problemId, {keepRollout = false} = {}) {
    state.problemId = problemId;
    if (!keepRollout) { state.open = null; state.pinned = null; }
    for (const el of document.querySelectorAll('.problem-item')) el.classList.toggle('active', el.dataset.id === problemId);
    const detail = $('detail');
    detail.replaceChildren(node('div', 'empty-state', 'Loading problem…'));
    const row = state.index[state.bench].problems.find(r => r.id === problemId);
    let problem;
    try {
      problem = await fetchJSON(`problems/${state.bench}/${problemId}.json.gz`);
    } catch (error) {
      detail.replaceChildren(node('p', 'error', error.message));
      return;
    }
    if (state.problemId !== problemId) return;
    state.problem = problem;
    detail.replaceChildren();
    const head = node('header', 'problem-head');
    const title = node('div');
    const benchLabel = state.meta.benchmarks.find(b => b.id === state.bench)?.label;
    title.append(node('p', 'kicker', benchLabel), node('h2', '', problemId));
    const nav = node('div', 'problem-nav');
    const pos = state.rows.findIndex(r => r.id === problemId);
    nav.append(button('← prev', () => step(-1), 'button secondary'), node('span', 'hint', pos >= 0 ? `${pos + 1} / ${state.rows.length}` : ''),
      button('next →', () => step(1), 'button secondary'));
    head.append(title, nav);
    detail.append(head);
    const bk = bankOf(row);
    if (bk) detail.append(node('p', `bank-line bank-${bk.kind}`, `Prior 9B Direct attempts (seeds 0–7): ${bk.pos}/${bk.n} judged 7/7 — this is the evidence bank the attempt-conditioned verifiers see.`));
    const statement = node('section', 'statement');
    statement.append(markdown(problem.problem));
    detail.append(statement);
    const extras = node('div', 'extras');
    extras.append(lazy('Reference proof', () => textBlock(problem.reference)));
    const templates = Object.values(problem.judge_templates || {});
    if (templates.length) {
      extras.append(lazy('Judge grading prompt', () => markerText(templates[0], {traj: null, callsByIndex: new Map(), root: detail})));
    }
    for (const mode of MODES) {
      if (problem.packs?.[mode]) extras.append(lazy(`Evidence pack · ${MODE_LABELS[mode]}`, () => renderPack(problem.packs[mode], mode)));
    }
    detail.append(extras);
    for (const exp of state.meta.experiments) if (row.cells[exp.id]) detail.append(gridFor(exp, row));
    const area = node('section', 'rollout-area');
    area.id = 'rollout-area';
    detail.append(area);
    updateHash();
    await drawRollouts();
  }

  function step(delta) {
    const pos = state.rows.findIndex(r => r.id === state.problemId);
    const next = state.rows[pos + delta];
    if (next) selectProblem(next.id);
  }

  async function loadTrajectory(sel) {
    const trajectories = await fetchJSON(`rollouts/${sel.expId}/${state.bench}/${state.problemId}/${sel.armId}.json.gz`);
    const traj = trajectories.find(t => t.seed === sel.seed);
    if (!traj) throw new Error(`seed ${sel.seed} not found`);
    return traj;
  }

  async function slot(sel, pinned) {
    const box = node('div', 'rollout-slot');
    const bar = node('div', 'slot-bar');
    if (pinned) bar.append(badge('pinned', 'pinned'), button('unpin', () => { state.pinned = null; markCells(); updateHash(); drawRollouts(); }, 'text-button'));
    else {
      bar.append(button('📌 pin for comparison', () => { state.pinned = state.open; state.open = null; markCells(); updateHash(); drawRollouts(); }, 'text-button'));
      bar.append(button('close', () => { state.open = null; markCells(); updateHash(); drawRollouts(); }, 'text-button'));
    }
    box.append(bar);
    const body = node('div', '', 'Loading rollout…');
    box.append(body);
    try {
      const traj = await loadTrajectory(sel);
      body.replaceChildren(renderRollout(traj, sel.expId));
    } catch (error) {
      body.replaceChildren(node('p', 'error', error.message));
    }
    return box;
  }

  async function drawRollouts() {
    const area = $('rollout-area');
    if (!area) return;
    const sels = [state.pinned && {...state.pinned, pinned: true}, state.open].filter(Boolean);
    if (!sels.length) {
      area.replaceChildren(node('p', 'empty-rollout', 'Click a cell in a grid above to open that rollout.'));
      return;
    }
    area.classList.toggle('split', sels.length === 2);
    const slots = await Promise.all(sels.map(sel => slot(sel, Boolean(sel.pinned))));
    area.replaceChildren(...slots);
  }

  async function openRollout(expId, armId, seed) {
    state.open = {expId, armId, seed};
    markCells();
    updateHash();
    await drawRollouts();
    $('rollout-area')?.scrollIntoView({behavior: 'smooth', block: 'start'});
  }

  function updateHash() {
    const p = new URLSearchParams();
    if (state.view !== 'rollouts') {
      p.set('v', state.view);
      for (const [key, value] of window.RolloutSummary.params()) p.set(key, value);
      history.replaceState(null, '', `#${p}`);
      return;
    }
    if (state.bench) p.set('b', state.bench);
    if (state.problemId) p.set('p', state.problemId);
    if (state.open) { p.set('e', state.open.expId); p.set('a', state.open.armId); p.set('s', state.open.seed); }
    if (state.pinned) { p.set('pe', state.pinned.expId); p.set('pa', state.pinned.armId); p.set('ps', state.pinned.seed); }
    history.replaceState(null, '', `#${p}`);
  }

  async function restore() {
    const p = new URLSearchParams(location.hash.slice(1));
    const bench = state.meta.benchmarks.some(b => b.id === p.get('b')) ? p.get('b') : state.meta.benchmarks[0].id;
    const sel = (e, a, s) => (p.get(e) && p.get(a) && p.get(s) !== null ? {expId: p.get(e), armId: p.get(a), seed: Number(p.get(s))} : null);
    const open = sel('e', 'a', 's');
    const pinned = sel('pe', 'pa', 'ps');
    await selectBenchmark(bench, {problemId: p.get('p')});
    if (open || pinned) {
      state.open = open;
      state.pinned = pinned;
      markCells();
      updateHash();
      await drawRollouts();
    }
  }

  // Summary and Analysis are drawn by summary.js; Rollouts loads its problem index on first visit.
  async function setView(view, params = new URLSearchParams()) {
    state.view = view;
    for (const tab of document.querySelectorAll('[data-view]')) {
      tab.classList.toggle('active', tab.dataset.view === view);
      tab.setAttribute('aria-selected', String(tab.dataset.view === view));
    }
    $('rollouts-view').hidden = view !== 'rollouts';
    $('stats-view').hidden = view === 'rollouts';
    if (view === 'rollouts') {
      window.RolloutSummary.hide();
      if (state.bench) updateHash();
      else await restore();
      return;
    }
    await window.RolloutSummary.show($('stats-view'), view, params, {fetchJSON, meta: state.meta, onChange: updateHash});
    updateHash();
  }

  async function init() {
    $('help-toggle').addEventListener('click', () => { $('help').hidden = !$('help').hidden; });
    $('search').addEventListener('input', renderList);
    $('bank-filter').addEventListener('change', renderList);
    $('sort').addEventListener('change', renderList);
    document.addEventListener('keydown', event => {
      if (state.view !== 'rollouts' || event.target.matches('input, select, textarea')) return;
      if (event.key === 'j') step(1);
      if (event.key === 'k') step(-1);
    });
    const link = $('dataset-link');
    if (CONFIG.datasetRepo) link.href = `https://huggingface.co/datasets/${CONFIG.datasetRepo}`;
    else link.hidden = true;
    for (const tab of document.querySelectorAll('[data-view]')) tab.addEventListener('click', () => setView(tab.dataset.view));
    try {
      state.meta = await fetchJSON('index/experiments.json');
      const p = new URLSearchParams(location.hash.slice(1));
      await setView(VIEWS.includes(p.get('v')) ? p.get('v') : 'rollouts', p);
    } catch (error) {
      $('detail').replaceChildren(node('p', 'error', `Could not load data from ${DATA}: ${error.message}`));
    }
  }

  init();
})();
