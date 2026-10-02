'use strict';
(() => {
  const CONFIG = window.TREE_CONFIG || {};
  const query = new URLSearchParams(location.search);
  const DATA = (query.get('data') || CONFIG.dataBase || '../data').replace(/\/$/, '');
  const NODE_W = 52, NODE_H = 26, COL_GAP = 40, ROW_H = 38, TOP = 46, LEFT = 10;
  const MODE_LABELS = {generate: 'first generation', recheck: 're-check', revise: 'revise', regenerate: 'regenerate'};
  const MODE_FOR_VERDICT = {correct: 'recheck', minor_fix: 'revise', critical_flaw: 'regenerate'};
  const VERDICT_LABELS = {correct: 'says correct', minor_fix: 'minor fix', critical_flaw: 'critical flaw'};
  const MODE_SERIES = {recheck: 1, revise: 2, regenerate: 3};
  const BUCKET_LABELS = {zero: 'never solved', low: 'solved < half', high: 'solved ≥ half', one: 'always solved'};
  const PAGE = 300;
  const LONG_TEXT = 60000;
  const state = {
    view: 'trees', rows: [], split: 'all', filtered: [], shown: PAGE, treeId: null, tree: null, layout: null,
    selected: null, overview: null, prompts: null, tableView: false,
  };
  const cache = new Map();

  const $ = id => document.getElementById(id);
  const node = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  };
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const svg = (tag, attrs = {}) => {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    return el;
  };
  const fmt = n => (n === null || n === undefined ? '–' : Number(n).toLocaleString('en-US'));
  const kfmt = n => (n === null || n === undefined ? '–' : n >= 1000 ? `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k` : String(n));
  const pct = (x, digits = 0) => (x === null || x === undefined ? '–' : `${(100 * x).toFixed(digits)}%`);
  const button = (text, handler, cls = 'button') => {
    const b = node('button', cls, text);
    b.type = 'button';
    b.addEventListener('click', handler);
    return b;
  };
  const badge = (text, cls = '') => node('span', `badge ${cls}`, text);
  const letter = index => {
    let out = '';
    for (let i = index + 1; i > 0; i = Math.floor((i - 1) / 26)) out = String.fromCharCode(65 + ((i - 1) % 26)) + out;
    return out;
  };
  const glyph = correct => (correct === true ? '✓' : correct === false ? '✗' : '?');
  const judgeText = correct => (correct === true ? '✓ correct' : correct === false ? '✗ incorrect' : 'not judged');
  const lazy = (title, make, {open = false, cls = 'lazy-block'} = {}) => {
    const d = node('details', cls);
    const s = node('summary');
    if (title instanceof Node) s.append(title); else s.textContent = title;
    d.append(s);
    const fill = async () => {
      if (d.dataset.loaded) return;
      d.dataset.loaded = '1';
      const slot = node('div', 'hint', 'Loading…');
      d.append(slot);
      try { slot.replaceWith(await make()); } catch (error) { slot.replaceWith(node('p', 'error', `Could not load: ${error.message}`)); }
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
    while (cache.size >= 40) {
      const oldest = [...cache.keys()].find(key => key.startsWith('trees/') || key.startsWith('reasoning/'));
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

  // A bare LaTeX fragment, such as the contents of \boxed{}.
  function mathInline(source) {
    const span = node('span', 'answer-math');
    if (source === null || source === undefined) { span.textContent = 'no \\boxed{} answer'; span.classList.add('hint'); return span; }
    if (window.katex) {
      try {
        window.katex.render(source, span, {throwOnError: true, trust: false, strict: 'ignore', maxExpand: 1000});
        return span;
      } catch { /* fall back to the raw text */ }
    }
    span.textContent = source;
    span.classList.add('mono');
    return span;
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
        const h = node(`h${Math.min(heading[1].length + 3, 6)}`);
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

  // Prompt templates: ⟦problem⟧-style markers stand for texts shown elsewhere on the page.
  function promptText(text) {
    const pre = node('div', 'prompt-text');
    for (const part of String(text || '').split(/(⟦[a-z]+⟧)/)) {
      if (/^⟦[a-z]+⟧$/.test(part)) pre.append(node('span', 'marker', part));
      else if (part) pre.append(document.createTextNode(part));
    }
    return pre;
  }

  function promptExample(kind) {
    const example = state.prompts && state.prompts[kind];
    const box = node('div');
    if (!example) { box.append(node('p', 'hint', 'No example prompt for this step.')); return box; }
    box.append(node('p', 'hint', `Exact template of a "${example.label.replace(/^r\d+\.b\d+\./, '')}" call. ⟦problem⟧, ⟦candidate⟧, ⟦critique⟧ and ⟦category⟧ stand for this tree's problem, the candidate under review and the verifier's feedback.`));
    for (const message of example.messages) {
      const item = node('div', 'prompt-message');
      item.append(node('div', 'prompt-role', message.role), promptText(message.content));
      box.append(item);
    }
    return box;
  }

  // ---------------------------------------------------------------- tooltip
  const tooltip = () => $('tooltip');
  // Anchored to an element's box, or to the pointer for long marks such as edges.
  function showTooltip(anchor, rows, event = null) {
    const tip = tooltip();
    tip.replaceChildren(...rows);
    tip.hidden = false;
    const box = event ? {left: event.clientX, right: event.clientX + 6, top: event.clientY + 6} : anchor.getBoundingClientRect();
    const width = tip.offsetWidth, height = tip.offsetHeight;
    let left = box.right + 10, top = box.top - 4;
    if (left + width > window.innerWidth - 8) left = Math.max(8, box.left - width - 10);
    if (top + height > window.innerHeight - 8) top = Math.max(8, window.innerHeight - height - 8);
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }
  function hideTooltip() { tooltip().hidden = true; }
  const ttTitle = text => node('div', 'tt-title', text);
  const ttRow = (label, value) => { const row = node('div', 'tt-row'); row.append(node('span', '', label), node('span', '', value)); return row; };
  const ttText = text => node('div', 'tt-text', text);
  const clip = (text, n) => { const s = String(text || '').replace(/\s+/g, ' ').trim(); return s.length > n ? `${s.slice(0, n - 1)}…` : s; };

  // ---------------------------------------------------------------- problem list
  function matches(row) {
    if (state.split !== 'all' && row.split !== state.split) return false;
    const bucket = $('bucket-filter').value, kind = $('kind-filter').value;
    if (bucket && row.bucket !== bucket) return false;
    if (kind === 'fixed' || kind === 'broken') { if (row.change !== kind) return false; }
    else if (kind && row.kind !== kind) return false;
    const q = $('search').value.trim().toLowerCase();
    return !q || row.search.includes(q);
  }

  function sortRows(rows) {
    const order = $('sort').value;
    const balance = row => Math.min(row.correct, row.nodes - row.correct);
    const by = {
      id: (a, b) => a.order - b.order,
      mixed: (a, b) => balance(b) - balance(a) || a.order - b.order,
      hard: (a, b) => a.pass - b.pass || a.order - b.order,
      easy: (a, b) => b.pass - a.pass || a.order - b.order,
      answers: (a, b) => b.answers - a.answers || a.order - b.order,
    }[order] || ((a, b) => a.order - b.order);
    return rows.slice().sort(by);
  }

  function renderSplitTabs() {
    const counts = {all: state.rows.length, train: 0, eval: 0};
    for (const row of state.rows) counts[row.split]++;
    const tabs = $('split-tabs');
    tabs.replaceChildren();
    for (const [key, label] of [['all', 'All'], ['train', 'Train'], ['eval', 'Eval']]) {
      const b = button(`${label} · ${fmt(counts[key])}`, () => { state.split = key; state.shown = PAGE; renderSplitTabs(); renderList(); updateHash(); }, `tab${state.split === key ? ' active' : ''}`);
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', String(state.split === key));
      tabs.append(b);
    }
  }

  function spark(spine) {
    const wrap = node('span', 'spark');
    wrap.setAttribute('role', 'img');
    wrap.setAttribute('aria-label', `spine c1 to c10: ${spine.map(v => (v ? '✓' : '✗')).join(' ')}`);
    wrap.title = `Spine c1 → c10: ${spine.map(v => (v ? '✓' : '✗')).join(' ')}`;
    for (const value of spine) wrap.append(node('i', value ? 'ok' : 'bad'));
    return wrap;
  }

  function listItem(row) {
    const li = node('li');
    const item = node('button', `problem-item${row.id === state.treeId ? ' active' : ''}`);
    item.type = 'button';
    item.dataset.id = row.id;
    const top = node('div', 'problem-top');
    const left = node('span');
    left.append(node('span', 'problem-id', row.id), badge(row.split));
    top.append(left, node('span', 'problem-stats', `${row.correct}/${row.nodes} ✓`));
    const stats = node('div', 'problem-stats');
    stats.append(spark(row.spine), node('span', '', `Qwen3.6 ${pct(row.pass)}`), node('span', '', `${row.answers} ans`));
    item.append(top, stats, node('div', 'problem-preview', clip(row.problem, 150)));
    item.addEventListener('click', () => selectTree(row.id));
    li.append(item);
    return li;
  }

  function renderList() {
    state.filtered = sortRows(state.rows.filter(matches));
    const list = $('problem-list');
    list.replaceChildren(...state.filtered.slice(0, state.shown).map(listItem));
    if (state.filtered.length > state.shown) {
      const li = node('li');
      li.append(button(`Show ${Math.min(PAGE, state.filtered.length - state.shown)} more`, () => { state.shown += PAGE; renderList(); }, 'button secondary more'));
      list.append(li);
    }
    $('list-count').textContent = `${fmt(state.filtered.length)} of ${fmt(state.rows.length)} trees`;
  }

  function step(delta) {
    if (!state.filtered.length) return;
    const index = state.filtered.findIndex(row => row.id === state.treeId);
    const next = state.filtered[Math.min(state.filtered.length - 1, Math.max(0, index + delta))];
    if (next && next.id !== state.treeId) selectTree(next.id);
  }

  // ---------------------------------------------------------------- tree layout and drawing
  function layoutTree(tree) {
    const byCall = new Map(tree.nodes.map(n => [n.call, n]));
    const verdictByCall = new Map(tree.verdicts.map(v => [v.call, v]));
    const childrenOf = new Map();
    for (const n of tree.nodes) {
      if (n.parent === null || n.parent === undefined) continue;
      if (!childrenOf.has(n.parent)) childrenOf.set(n.parent, []);
      childrenOf.get(n.parent).push(n);
    }
    const spine = tree.nodes.filter(n => n.spine).sort((a, b) => a.cycle - b.cycle);
    const columns = [[spine[0] || tree.nodes[0]]];
    for (const parent of spine) {
      const kids = (childrenOf.get(parent.call) || []).slice().sort((a, b) => Number(b.spine) - Number(a.spine) || a.branch - b.branch);
      if (kids.length) columns.push(kids);
    }
    const pos = new Map();
    columns.forEach((column, col) => column.forEach((n, row) => {
      pos.set(n.call, {col, row, x: LEFT + col * (NODE_W + COL_GAP), y: TOP + row * ROW_H});
    }));
    return {byCall, verdictByCall, childrenOf, spine, columns, pos};
  }

  const nodeName = n => (n.spine ? `c${n.cycle}` : `c${n.point} · b${n.branch}`);
  const nodeRole = n => (n.parent === null || n.parent === undefined
    ? 'first generation'
    : n.spine ? `spine · from c${n.point}, branch ${n.branch}` : `leaf · revision of c${n.point}, branch ${n.branch}`);
  const answerLetter = n => (n.answer === null || n.answer === undefined ? '–' : letter(n.answer));
  const edgeMode = (n, verdict) => (MODE_SERIES[n.mode] ? n.mode : MODE_FOR_VERDICT[verdict && verdict.verdict] || 'recheck');

  function nodeTooltip(n) {
    const L = state.layout, tree = state.tree;
    const verdict = L.verdictByCall.get(n.verdict_call);
    const answer = tree.answers[n.answer];
    const rows = [ttTitle(`${nodeName(n)} — ${judgeText(n.correct)}`), ttRow('Position', nodeRole(n))];
    if (verdict) rows.push(ttRow('Made by', `${VERDICT_LABELS[verdict.verdict] || verdict.verdict} → ${MODE_LABELS[n.mode] || n.mode}`));
    rows.push(ttRow('Answer', `${answerLetter(n)}${answer && answer.text ? ` = ${clip(answer.text, 60)}` : ''}`));
    rows.push(ttRow('Tokens', `${fmt(n.tokens)} (${fmt(n.reasoning_tokens)} reasoning)`));
    if (n.recovered) rows.push(ttText('The first attempt returned no answer; a forced-recovery turn produced this one.'));
    return rows;
  }

  function edgeTooltip(n) {
    const verdict = state.layout.verdictByCall.get(n.verdict_call);
    const rows = [ttTitle(`v${n.point} · branch ${n.branch}`)];
    if (verdict) {
      rows.push(ttRow('Verifier on c' + n.point, VERDICT_LABELS[verdict.verdict] || verdict.verdict));
      if (verdict.category && verdict.category !== 'None') rows.push(ttRow('Category', clip(verdict.category, 48)));
      rows.push(ttRow('Revision', MODE_LABELS[n.mode] || n.mode));
      if (verdict.critique) rows.push(ttText(clip(verdict.critique, 260)));
    }
    return rows;
  }

  function pointTooltip(point) {
    const L = state.layout;
    const parent = L.spine[point - 1];
    const kids = parent ? (L.childrenOf.get(parent.call) || []) : [];
    const said = {correct: 0, minor_fix: 0, critical_flaw: 0};
    for (const kid of kids) { const v = L.verdictByCall.get(kid.verdict_call); if (v) said[v.verdict] = (said[v.verdict] || 0) + 1; }
    return [
      ttTitle(`Verification point ${point}`),
      ttText(`${kids.length} verifiers judged c${point}; each verdict routed one revision.`),
      ttRow('Says correct', said.correct), ttRow('Minor fix', said.minor_fix), ttRow('Critical flaw', said.critical_flaw),
      ttRow('Revisions judged ✓', `${kids.filter(k => k.correct).length}/${kids.length}`),
    ];
  }

  function drawTree() {
    const tree = state.tree, L = state.layout;
    const cols = L.columns.length, rows = Math.max(...L.columns.map(c => c.length));
    const width = LEFT * 2 + cols * NODE_W + (cols - 1) * COL_GAP;
    const height = TOP + rows * ROW_H + 6;
    const root = svg('svg', {class: 'tree-svg', width, height, viewBox: `0 0 ${width} ${height}`, role: 'group', 'aria-label': `Tree for ${tree.id}: ${tree.nodes.length} candidates`});
    const last = L.spine[L.spine.length - 1];
    if (last) {
      const end = L.pos.get(last.call);
      root.append(svg('rect', {class: 'spine-band', x: LEFT - 6, y: TOP - 6, width: end.x + NODE_W + 12 - LEFT, height: NODE_H + 12, rx: 9}));
    }
    // Column and point labels.
    L.columns.forEach((column, col) => {
      const x = LEFT + col * (NODE_W + COL_GAP) + NODE_W / 2;
      const spineNode = column.find(n => n.spine);
      const label = spineNode ? `c${spineNode.cycle}${spineNode === last ? ' · final' : ''}` : 'leaves';
      const text = svg('text', {class: 'col-label', x, y: 14, 'text-anchor': 'middle'});
      text.textContent = label;
      root.append(text);
      if (col > 0) {
        const px = x - NODE_W / 2 - COL_GAP / 2;
        const g = svg('g', {class: 'point-hit', tabindex: '-1'});
        const hit = svg('rect', {x: px - 14, y: 22, width: 28, height: 16, fill: 'transparent'});
        const t = svg('text', {class: 'point-label', x: px, y: 34, 'text-anchor': 'middle'});
        t.textContent = `v${col}`;
        g.append(hit, t);
        g.addEventListener('pointerenter', () => showTooltip(g, pointTooltip(col)));
        g.addEventListener('pointerleave', hideTooltip);
        root.append(g);
      }
    });
    // Edges under nodes; spine edges last so they sit on top of sibling edges.
    const edges = svg('g');
    const ordered = tree.nodes.filter(n => n.parent !== null && n.parent !== undefined && L.pos.has(n.parent) && L.pos.has(n.call))
      .sort((a, b) => Number(a.spine) - Number(b.spine));
    for (const n of ordered) {
      const from = L.pos.get(n.parent), to = L.pos.get(n.call);
      const x1 = from.x + NODE_W, y1 = from.y + NODE_H / 2, x2 = to.x, y2 = to.y + NODE_H / 2, mid = (x1 + x2) / 2;
      const d = `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`;
      const verdict = L.verdictByCall.get(n.verdict_call);
      const mode = edgeMode(n, verdict);
      edges.append(svg('path', {d, class: `edge edge-${mode}${n.spine ? ' spine-edge' : ''}`}));
      const hit = svg('path', {d, class: 'edge-hit'});
      hit.addEventListener('pointerenter', event => showTooltip(hit, edgeTooltip(n), event));
      hit.addEventListener('pointermove', event => showTooltip(hit, edgeTooltip(n), event));
      hit.addEventListener('pointerleave', hideTooltip);
      hit.addEventListener('click', () => selectNode(n.call, {focus: true, section: 'verdict'}));
      edges.append(hit);
    }
    root.append(edges);
    // Nodes.
    for (const n of tree.nodes) {
      const p = L.pos.get(n.call);
      if (!p) continue;
      const cls = n.correct === true ? 'ok' : n.correct === false ? 'bad' : 'unknown';
      const g = svg('g', {class: `tnode ${cls}${state.selected === n.call ? ' selected' : ''}`, transform: `translate(${p.x},${p.y})`, tabindex: '0', role: 'button', 'data-call': n.call,
        'aria-label': `${nodeName(n)}, ${nodeRole(n)}, answer ${answerLetter(n)}, judge ${judgeText(n.correct)}`});
      g.append(svg('rect', {class: 'ring', x: -4, y: -4, width: NODE_W + 8, height: NODE_H + 8, rx: 9}));
      g.append(svg('rect', {class: 'body', width: NODE_W, height: NODE_H, rx: 6}));
      const t = svg('text', {x: NODE_W / 2, y: NODE_H / 2 + 4.5, 'text-anchor': 'middle'});
      t.textContent = `${answerLetter(n)} ${glyph(n.correct)}`;
      g.append(t);
      g.addEventListener('pointerenter', () => showTooltip(g, nodeTooltip(n)));
      g.addEventListener('pointerleave', hideTooltip);
      g.addEventListener('focus', () => showTooltip(g, nodeTooltip(n)));
      g.addEventListener('blur', hideTooltip);
      g.addEventListener('click', () => selectNode(n.call, {focus: true}));
      g.addEventListener('keydown', event => onNodeKey(event, n));
      root.append(g);
    }
    return root;
  }

  function onNodeKey(event, n) {
    const L = state.layout;
    const p = L.pos.get(n.call);
    let target = null;
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectNode(n.call, {focus: true}); return; }
    if (event.key === 'ArrowRight') target = (L.columns[p.col + 1] || []).find(k => k.parent === n.call);
    if (event.key === 'ArrowLeft') target = L.byCall.get(n.parent);
    if (event.key === 'ArrowDown') target = L.columns[p.col][p.row + 1];
    if (event.key === 'ArrowUp') target = L.columns[p.col][p.row - 1];
    if (!target) return;
    event.preventDefault();
    focusNode(target.call);
  }

  function focusNode(call) {
    const el = document.querySelector(`.tnode[data-call="${call}"]`);
    if (el) el.focus();
  }

  function legend() {
    const box = node('div', 'legend');
    const nodeKey = (cls, text, label) => {
      const item = node('span', 'legend-item');
      item.append(node('span', `legend-node ${cls}`, text), node('span', '', label));
      return item;
    };
    const edgeKey = (mode, label) => {
      const item = node('span', 'legend-item');
      const s = svg('svg', {class: 'legend-edge', width: 30, height: 10, 'aria-hidden': 'true'});
      s.append(svg('line', {x1: 1, y1: 5, x2: 29, y2: 5, class: `edge edge-${mode}`}));
      item.append(s, node('span', '', label));
      return item;
    };
    box.append(
      nodeKey('ok', 'A ✓', 'judge: correct'), nodeKey('bad', 'B ✗', 'judge: incorrect'),
      edgeKey('recheck', 'says correct → re-check'), edgeKey('revise', 'minor fix → revise'), edgeKey('regenerate', 'critical flaw → regenerate'),
      node('span', '', 'Letters name final answers · thick row = spine'),
    );
    return box;
  }

  function answersTable() {
    const tree = state.tree;
    const selected = state.layout.byCall.get(state.selected);
    const table = node('table', 'answers');
    const head = node('tr');
    for (const label of ['Answer', 'Final answer (extracted \\boxed{})', 'Candidates', 'Judge']) head.append(node('th', '', label));
    const thead = node('thead');
    thead.append(head);
    const body = node('tbody');
    tree.answers.forEach((answer, index) => {
      const tr = node('tr', selected && selected.answer === index ? 'chosen' : '');
      const first = tree.nodes.filter(n => n.answer === index).sort((a, b) => a.call - b.call)[0];
      const cellLetter = node('td', 'letter');
      cellLetter.append(button(letter(index), () => first && selectNode(first.call, {focus: true}), 'text-button'));
      const cellAnswer = node('td');
      cellAnswer.append(mathInline(answer.text));
      const verdict = answer.correct === true ? badge('✓ correct', 'ok') : answer.correct === false ? badge('✗ incorrect', 'bad') : badge('labels differ');
      const cellJudge = node('td');
      cellJudge.append(verdict);
      tr.append(cellLetter, cellAnswer, node('td', 'num', answer.count), cellJudge);
      body.append(tr);
    });
    table.append(thead, body);
    const wrap = node('div', 'table-scroll');
    wrap.append(table);
    return wrap;
  }

  async function reasoningOf(call) {
    const reasoning = await fetchJSON(`reasoning/${state.treeId}.json.gz`);
    return reasoning[String(call)] || '';
  }

  function reasoningBlock(title, call) {
    return lazy(title, async () => {
      const text = await reasoningOf(call);
      if (!text) return node('p', 'hint', 'No reasoning was recorded for this call.');
      const block = textBlock(text, {rendered: false});
      block.classList.add('reasoning');
      return block;
    });
  }

  // The same tree as rows, for reading without colour or line style.
  function treeTable() {
    const tree = state.tree, L = state.layout;
    const rows = [...L.pos.entries()].sort((a, b) => a[1].col - b[1].col || a[1].row - b[1].row).map(([call]) => L.byCall.get(call));
    const table = node('table', 'answers tree-table');
    const head = node('tr');
    for (const label of ['Node', 'Position', 'Verifier on parent', 'Revision', 'Answer', 'Judge', 'Tokens']) head.append(node('th', '', label));
    const thead = node('thead');
    thead.append(head);
    const body = node('tbody');
    for (const n of rows) {
      const verdict = L.verdictByCall.get(n.verdict_call);
      const tr = node('tr', n.call === state.selected ? 'chosen' : '');
      const name = node('td', 'letter');
      name.append(button(nodeName(n), () => selectNode(n.call, {focus: true}), 'text-button'));
      const answer = tree.answers[n.answer];
      tr.append(name, node('td', '', nodeRole(n)), node('td', '', verdict ? VERDICT_LABELS[verdict.verdict] || verdict.verdict : '–'),
        node('td', '', MODE_LABELS[n.mode] || n.mode), node('td', 'mono', `${answerLetter(n)}${answer && answer.text ? ` = ${clip(answer.text, 40)}` : ''}`),
        node('td', '', judgeText(n.correct)), node('td', 'num', fmt(n.tokens)));
      body.append(tr);
    }
    table.append(thead, body);
    const wrap = node('div', 'table-scroll');
    wrap.append(table);
    return wrap;
  }

  function modeBadge(mode) {
    const b = badge('', `mode-${mode}`);
    if (MODE_SERIES[mode]) {
      const swatch = node('span', 'swatch');
      swatch.style.background = `var(--series-${MODE_SERIES[mode]})`;
      b.append(swatch);
    }
    b.append(document.createTextNode(MODE_LABELS[mode] || mode));
    return b;
  }

  function renderNodeDetail() {
    const slot = $('node-detail');
    if (!slot) return;
    const L = state.layout, tree = state.tree;
    const n = L.byCall.get(state.selected);
    slot.replaceChildren();
    if (!n) { slot.append(node('p', 'hint', 'Click a node to read that candidate.')); return; }
    const head = node('div', 'node-head');
    head.append(node('h3', '', nodeName(n)), modeBadge(n.mode), badge(judgeText(n.correct), n.correct ? 'ok' : n.correct === false ? 'bad' : ''),
      node('span', 'hint', `${nodeRole(n)} · ${fmt(n.tokens)} tokens (${fmt(n.reasoning_tokens)} reasoning) · finish: ${n.finish || '–'}`));
    if (n.parent !== null && n.parent !== undefined) head.append(button('↑ parent', () => selectNode(n.parent, {focus: true}), 'button secondary'));
    slot.append(head);

    const answer = node('div', 'gold-line');
    answer.append(node('span', 'label', `Answer ${answerLetter(n)}:`), mathInline(tree.answers[n.answer] ? tree.answers[n.answer].text : null),
      node('span', 'label', 'Gold:'), mathInline(tree.gold));
    slot.append(answer);
    if (n.recovered) slot.append(node('p', 'hint', 'The first attempt at this step returned no answer, so a forced-recovery turn asked for a complete response; this is that response.'));

    const verdict = L.verdictByCall.get(n.verdict_call);
    if (verdict) {
      const section = node('div', 'node-section');
      section.id = 'verdict-section';
      section.append(node('h4', '', `Verifier on c${verdict.point}, branch ${verdict.branch}`));
      const line = node('div', 'node-head');
      line.append(badge(VERDICT_LABELS[verdict.verdict] || verdict.verdict), node('span', 'hint', `category: ${verdict.category || '–'} · ${fmt(verdict.tokens)} tokens${verdict.recovered ? ' · parsed on the recovery attempt' : ''} · routed to ${MODE_LABELS[n.mode] || n.mode}`));
      section.append(line);
      const critique = node('div', 'critique');
      critique.append(markdown(verdict.critique || '(no critique)'));
      section.append(critique, reasoningBlock('Verifier reasoning', verdict.call), lazy('Verifier prompt template', async () => promptExample('verify')));
      slot.append(section);
    }

    const candidate = node('div', 'node-section');
    candidate.append(node('h4', '', 'Candidate'), textBlock(n.text));
    const promptKind = n.mode === 'generate' ? 'gen' : n.mode;
    candidate.append(reasoningBlock('Reasoning behind this candidate', n.call), lazy(`Prompt template for this step (${MODE_LABELS[n.mode] || n.mode})`, async () => promptExample(promptKind)));
    slot.append(candidate);

    const kids = (L.childrenOf.get(n.call) || []).slice().sort((a, b) => a.branch - b.branch);
    if (kids.length) {
      const section = node('div', 'node-section');
      section.append(node('h4', '', `Verified at point ${n.cycle} by ${kids.length} branches`));
      const list = node('div', 'children');
      for (const kid of kids) {
        const v = L.verdictByCall.get(kid.verdict_call);
        const b = button('', () => selectNode(kid.call, {focus: true}), 'child-button');
        b.append(node('span', 'mono', `b${kid.branch}`), node('span', 'subtle', `${v ? VERDICT_LABELS[v.verdict] : '?'} → ${MODE_LABELS[kid.mode] || kid.mode} →`),
          node('span', `legend-node ${kid.correct ? 'ok' : 'bad'}`, `${answerLetter(kid)} ${glyph(kid.correct)}`));
        if (kid.spine) b.append(node('span', 'subtle', 'spine'));
        list.append(b);
      }
      section.append(list);
      slot.append(section);
    }
  }

  function selectNode(call, {focus = false, section = null, scroll = true} = {}) {
    state.selected = call;
    for (const el of document.querySelectorAll('.tnode')) el.classList.toggle('selected', Number(el.dataset.call) === call);
    const scrollBox = document.querySelector('.tree-scroll');
    if (state.tableView && scrollBox) scrollBox.replaceChildren(treeTable());
    const answers = $('answers-slot');
    if (answers) answers.replaceChildren(answersTable());
    renderNodeDetail();
    updateHash();
    if (focus) focusNode(call);
    if (scroll) {
      const target = section === 'verdict' ? $('verdict-section') : $('node-detail');
      if (target && target.getBoundingClientRect().top > window.innerHeight - 80) target.scrollIntoView({block: 'start', behavior: 'smooth'});
    }
  }

  function renderTree() {
    const tree = state.tree, L = state.layout;
    const detail = $('detail');
    detail.replaceChildren();
    const row = state.rows.find(r => r.id === tree.id);
    const head = node('div', 'tree-head');
    const title = node('div');
    const kicker = node('p', 'kicker');
    kicker.append(document.createTextNode(`${tree.split} · arXiv `));
    const arxiv = node('a', '', tree.arxiv_id);
    arxiv.href = `https://arxiv.org/abs/${encodeURIComponent(tree.arxiv_id)}`;
    arxiv.target = '_blank';
    arxiv.rel = 'noopener noreferrer';
    kicker.append(arxiv);
    title.append(kicker, node('h2', '', tree.id));
    const nav = node('div', 'nav-buttons');
    nav.append(button('‹ prev', () => step(-1), 'button secondary'), button('next ›', () => step(1), 'button secondary'));
    head.append(title, nav);
    detail.append(head);

    const problem = node('section', 'panel');
    problem.append(markdown(tree.problem));
    const gold = node('div', 'gold-line');
    gold.append(node('span', 'label', 'Gold answer:'), mathInline(tree.gold));
    problem.append(gold);
    const facts = node('dl', 'facts');
    const fact = (label, value) => { const d = node('div'); d.append(node('dt', '', label), node('dd', '', value)); facts.append(d); };
    const q = tree.qwen36 || {};
    fact('Qwen3.6-35B', q.attempts ? `${q.correct}/${q.attempts} correct` : pct(q.pass_rate));
    fact('Candidates judged ✓', `${row ? row.correct : tree.nodes.filter(n => n.correct).length}/${tree.nodes.length}`);
    const first = L.spine[0], final = L.spine[L.spine.length - 1];
    if (first && final) fact('Spine', `c1 ${glyph(first.correct)} → c${final.cycle} ${glyph(final.correct)}`);
    fact('Distinct answers', tree.answers.length);
    fact('Generated tokens', fmt(tree.generated_tokens));
    problem.append(facts);
    detail.append(problem);

    const panel = node('section', 'panel');
    const panelHead = node('div', 'panel-head');
    const toggle = button(state.tableView ? 'Show tree' : 'Table view', () => {
      state.tableView = !state.tableView;
      toggle.textContent = state.tableView ? 'Show tree' : 'Table view';
      drawBody();
    }, 'text-button');
    panelHead.append(node('h3', '', 'Tree'), toggle);
    panel.append(panelHead, legend());
    const scroll = node('div', 'tree-scroll');
    const drawBody = () => { hideTooltip(); scroll.replaceChildren(state.tableView ? treeTable() : drawTree()); };
    drawBody();
    panel.append(scroll);
    const answers = node('div');
    answers.id = 'answers-slot';
    answers.append(answersTable());
    panel.append(answers);
    detail.append(panel);

    const nodePanel = node('section', 'panel');
    nodePanel.id = 'node-detail';
    detail.append(nodePanel);
    renderNodeDetail();
  }

  async function selectTree(id, {nodeCall = null} = {}) {
    state.treeId = id;
    for (const el of document.querySelectorAll('.problem-item')) el.classList.toggle('active', el.dataset.id === id);
    const detail = $('detail');
    detail.replaceChildren(node('div', 'empty-state', `Loading ${id}…`));
    hideTooltip();
    try {
      const tree = await fetchJSON(`trees/${id}.json.gz`);
      if (state.treeId !== id) return;
      state.tree = tree;
      state.layout = layoutTree(tree);
      const final = state.layout.spine[state.layout.spine.length - 1];
      state.selected = nodeCall !== null && state.layout.byCall.has(nodeCall) ? nodeCall : (final || tree.nodes[0]).call;
      renderTree();
      updateHash();
    } catch (error) {
      detail.replaceChildren(node('p', 'error', `Could not load ${id}: ${error.message}`));
    }
  }

  // ---------------------------------------------------------------- overview
  function statsTable(headers, rows) {
    const table = node('table', 'stats');
    const thead = node('thead'), tr = node('tr');
    for (const h of headers) tr.append(node('th', '', h));
    thead.append(tr);
    const tbody = node('tbody');
    for (const cells of rows) {
      const row = node('tr');
      for (const cell of cells) {
        const td = node('td');
        if (cell instanceof Node) td.append(cell); else td.textContent = cell;
        row.append(td);
      }
      tbody.append(row);
    }
    table.append(thead, tbody);
    const wrap = node('div', 'table-scroll');
    wrap.append(table);
    return wrap;
  }

  const share = (part, whole) => {
    const span = node('span');
    span.append(document.createTextNode(pct(whole ? part / whole : null, 1)));
    span.append(node('span', 'sub', ` ${fmt(part)}/${fmt(whole)}`));
    return span;
  };

  async function renderOverview() {
    const root = $('overview-view');
    if (root.dataset.loaded) return;
    root.dataset.loaded = '1';
    root.replaceChildren(node('p', 'hint', 'Loading…'));
    try {
      const [overview, prompts] = await Promise.all([fetchJSON('index/overview.json'), fetchJSON('index/prompts.json')]);
      state.overview = overview;
      state.prompts = prompts;
      root.replaceChildren();
      root.append(node('h2', '', 'Overview'));
      root.append(node('p', 'hint', `${overview.model} on ArXivMath (${overview.source_dataset}), ${overview.rounds} verification points × ${overview.branches} branches per tree; judge: ${overview.judge}.`));
      const trees = (overview.trees.train || 0) + (overview.trees.eval || 0);
      const spine = overview.spine_accuracy;
      const kinds = Object.values(overview.kinds_by_bucket).reduce((acc, k) => { for (const [key, v] of Object.entries(k)) acc[key] = (acc[key] || 0) + v; return acc; }, {});
      const tiles = node('div', 'tiles');
      const tile = (value, label) => { const t = node('div', 'tile'); t.append(node('div', 'value', value), node('div', 'label', label)); tiles.append(t); };
      tile(fmt(trees), `trees · ${fmt(overview.trees.train)} train, ${fmt(overview.trees.eval)} eval`);
      tile(fmt(overview.nodes), `candidates, all judged · ${fmt(overview.verifications)} verdicts`);
      tile(`${pct(spine[0], 1)} → ${pct(spine[spine.length - 1], 1)}`, 'spine accuracy, c1 → c10');
      tile(pct(overview.any_correct, 1), 'trees with at least one ✓ candidate');
      tile(pct((kinds.mixed || 0) / trees, 1), 'trees with both ✓ and ✗ candidates');
      tile(`${(overview.generated_tokens / 1e9).toFixed(2)}B`, 'generated tokens');
      root.append(tiles);

      const changes = node('section', 'panel');
      changes.append(node('h3', '', 'What each revision mode does to the judge label'));
      changes.append(statsTable(['Revision mode (verdict)', 'Revisions', 'Fixes a ✗ parent', 'Breaks a ✓ parent'],
        ['recheck', 'revise', 'regenerate'].map(mode => {
          const c = overview.changes[mode];
          const verdict = Object.keys(MODE_FOR_VERDICT).find(v => MODE_FOR_VERDICT[v] === mode);
          return [`${MODE_LABELS[mode]} (${VERDICT_LABELS[verdict]})`, fmt(c['I->C'] + c['I->I'] + c['C->I'] + c['C->C']), share(c['I->C'], c['I->C'] + c['I->I']), share(c['C->I'], c['C->I'] + c['C->C'])];
        })));
      changes.append(node('p', 'note', 'A revision "fixes" its parent when the parent candidate was judged ✗ and the revision ✓, and "breaks" it in the opposite case. Re-checks follow a "says correct" verdict and almost never change the answer.'));
      root.append(changes);

      const verdicts = node('section', 'panel');
      verdicts.append(node('h3', '', 'How the verifier\'s verdict tracks the judge'));
      verdicts.append(statsTable(['Candidate judged', 'Verdicts', 'Says correct', 'Minor fix', 'Critical flaw'],
        ['correct', 'incorrect'].map(label => {
          const v = overview.verdicts_by_label[label];
          const total = v.correct + v.minor_fix + v.critical_flaw;
          return [label === 'correct' ? '✓ correct' : '✗ incorrect', fmt(total), share(v.correct, total), share(v.minor_fix, total), share(v.critical_flaw, total)];
        })));
      verdicts.append(node('p', 'note', 'Each of the 40 verdicts per tree is joined to the judge label of the candidate it assessed.'));
      root.append(verdicts);

      const buckets = node('section', 'panel');
      buckets.append(node('h3', '', 'Tree labels by difficulty'));
      buckets.append(statsTable(['Qwen3.6-35B on the problem', 'Trees', 'Mixed ✓/✗', 'All ✓', 'All ✗'],
        Object.entries(overview.kinds_by_bucket).map(([bucket, k]) => {
          const total = (k.mixed || 0) + (k.all_correct || 0) + (k.all_incorrect || 0);
          return [BUCKET_LABELS[bucket] || bucket, fmt(total), share(k.mixed || 0, total), share(k.all_correct || 0, total), share(k.all_incorrect || 0, total)];
        })));
      buckets.append(node('p', 'note', 'Difficulty is how often Qwen3.6-35B answered correctly across MathArena\'s attempts at the problem.'));
      root.append(buckets);

      const rounds = node('section', 'panel');
      rounds.append(node('h3', '', 'Spine accuracy by round'));
      rounds.append(statsTable(spine.map((_, i) => `c${i + 1}`), [spine.map(v => pct(v, 1))]));
      const change = overview.spine_change || {};
      rounds.append(node('p', 'note', `From c1 to c10, ${fmt(change.fixed || 0)} trees went from ✗ to ✓ and ${fmt(change.broken || 0)} from ✓ to ✗; the rest kept c1's label.`));
      root.append(rounds);

      const promptsPanel = node('section', 'panel');
      promptsPanel.append(node('h3', '', 'Prompts'));
      for (const [kind, label] of [['gen', 'First generation'], ['verify', 'Verifier'], ['recheck', 'Re-check (after "says correct")'], ['revise', 'Revise (after "minor fix")'], ['regenerate', 'Regenerate (after "critical flaw")']]) {
        promptsPanel.append(lazy(label, async () => promptExample(kind)));
      }
      root.append(promptsPanel);
    } catch (error) {
      root.replaceChildren(node('p', 'error', `Could not load the overview: ${error.message}`));
      delete root.dataset.loaded;
    }
  }

  // ---------------------------------------------------------------- routing
  function updateHash() {
    const params = new URLSearchParams();
    if (state.view !== 'trees') params.set('v', state.view);
    if (state.treeId) params.set('t', state.treeId);
    if (state.selected !== null && state.view === 'trees') params.set('n', state.selected);
    if (state.split !== 'all') params.set('s', state.split);
    for (const [key, id] of [['b', 'bucket-filter'], ['k', 'kind-filter'], ['o', 'sort']]) {
      const value = $(id).value;
      if (value && !(key === 'o' && value === 'id')) params.set(key, value);
    }
    const q = $('search').value.trim();
    if (q) params.set('q', q);
    history.replaceState(null, '', `#${params.toString()}`);
  }

  async function setView(view) {
    state.view = view;
    for (const tab of document.querySelectorAll('.view-tab')) {
      const active = tab.dataset.view === view;
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', String(active));
    }
    $('trees-view').hidden = view !== 'trees';
    $('overview-view').hidden = view !== 'overview';
    hideTooltip();
    updateHash();
    if (view === 'overview') await renderOverview();
  }

  async function init() {
    const params = new URLSearchParams(location.hash.slice(1));
    if (CONFIG.datasetRepo) $('dataset-link').href = `https://huggingface.co/datasets/${CONFIG.datasetRepo}`;
    else $('dataset-link').hidden = true;
    $('help-toggle').addEventListener('click', () => {
      const help = $('help');
      help.hidden = !help.hidden;
      $('help-toggle').setAttribute('aria-expanded', String(!help.hidden));
    });
    for (const tab of document.querySelectorAll('.view-tab')) tab.addEventListener('click', () => setView(tab.dataset.view));
    state.split = ['train', 'eval'].includes(params.get('s')) ? params.get('s') : 'all';
    for (const [key, id] of [['b', 'bucket-filter'], ['k', 'kind-filter'], ['o', 'sort']]) {
      const value = params.get(key);
      if (value && [...$(id).options].some(o => o.value === value)) $(id).value = value;
      $(id).addEventListener('change', () => { state.shown = PAGE; renderList(); updateHash(); });
    }
    $('search').value = params.get('q') || '';
    let timer = null;
    $('search').addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => { state.shown = PAGE; renderList(); updateHash(); }, 120); });
    document.addEventListener('keydown', event => {
      if (state.view !== 'trees' || event.target.closest('input, select, textarea')) return;
      if (event.key === 'j') step(1);
      if (event.key === 'k') step(-1);
    });
    window.addEventListener('scroll', hideTooltip, {passive: true});
    try {
      const [index, prompts] = await Promise.all([fetchJSON('index/trees.json.gz'), fetchJSON('index/prompts.json')]);
      state.prompts = prompts;
      state.rows = index.trees.map((row, order) => ({...row, order, search: `${row.id} ${row.arxiv} ${row.problem}`.toLowerCase()}));
    } catch (error) {
      $('detail').replaceChildren(node('p', 'error', `Could not load the tree index: ${error.message}`));
      return;
    }
    renderSplitTabs();
    renderList();
    const wanted = params.get('t');
    const target = state.rows.find(row => row.id === wanted) || state.filtered[0] || state.rows[0];
    const nodeCall = params.has('n') ? Number(params.get('n')) : null;
    if (target) await selectTree(target.id, {nodeCall});
    if (params.get('v') === 'overview') await setView('overview');
  }

  init();
})();
