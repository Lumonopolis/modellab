/* ModelLab labelling page. The server owns answers and revisions; this page holds only an
   unsaved draft. Pane displays and question kinds are registered in ModelLab.displays and
   ModelLab.kinds; a task's plugins add their own the same way. */
(() => {
  'use strict';
  const ModelLab = window.ModelLab = {displays: {}, kinds: {}};
  const $ = id => document.getElementById(id);
  const make = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (key === 'text') node.textContent = value;
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else if (value !== false && value != null) node.setAttribute(key, value === true ? '' : value);
    }
    node.append(...children.filter(child => child != null));
    return node;
  };

  let task, items = [], index = 0, draft = {}, dirty = false, busy = false, retry = null;
  let keys = new Map(), panesByField = {}, labelOf = {}, flicked = false;
  const current = () => items[index];

  // Pane displays: render(container, value, ctx) where ctx = {pane, item, compare, marks, region}:
  // `marks` are a span question's marks to paint in a text, `region` whether boxes are drawn on it.

  const onScreen = fields => current().order.filter(field => fields.includes(field));  // left to right, as shown
  function zoomable(img, ctx) {
    img.addEventListener('click', () => openViewer(ctx.compare.includes(ctx.pane.field) ? onScreen(ctx.compare) : [ctx.pane.field]));
    return img;
  }
  ModelLab.displays.image = {render: (el, value, ctx) => el.append(zoomable(make('img', {src: value.url, alt: ctx.pane.label}), ctx))};
  ModelLab.displays.video = {render(el, value, ctx) {
    // Boxes are drawn on a paused frame, so a region video has its own controls, with frame steps.
    const video = make('video', {src: value.url, controls: !ctx.region, preload: 'metadata', muted: true, playsinline: true});
    if (ctx.compare.includes(ctx.pane.field)) {
      video.dataset.compare = '1';
      const others = () => [...document.querySelectorAll('video[data-compare]')].filter(other => other !== video);
      video.addEventListener('play', () => others().forEach(other => other.paused && other.play()));
      video.addEventListener('pause', () => others().forEach(other => !other.paused && other.pause()));
      video.addEventListener('seeked', () => others().forEach(other => {
        if (Math.abs(other.currentTime - video.currentTime) > 0.005) other.currentTime = video.currentTime;  // under a frame
      }));
    }
    el.append(video);
    if (ctx.region) el.append(videoBar(video));
  }};
  ModelLab.displays.audio = {render: (el, value) => el.append(make('audio', {src: value.url, controls: true, preload: 'none'}))};
  ModelLab.displays.text = {render: (el, value, ctx) => el.append(make('div', {class: 'text'}, ...pieces(String(value), 0, ctx.marks)))};
  ModelLab.displays.markdown = {render: (el, value, ctx) => el.append(markdown(String(value), ctx.marks))};
  ModelLab.displays.chat = {render(el, value) {
    for (const turn of Array.isArray(value) ? value : []) {
      el.append(make('p', {class: 'turn'}, make('b', {text: turn.role || '?'}), make('span', {class: 'text', text: turn.content ?? ''})));
    }
  }};
  ModelLab.displays.fields = {render(el, value) {
    const table = make('table');
    for (const [key, field] of Object.entries(value || {})) {
      table.append(make('tr', {}, make('th', {text: key}), make('td', {text: typeof field === 'object' ? JSON.stringify(field) : String(field)})));
    }
    el.append(table);
  }};
  ModelLab.displays.diff = {render(el, value) {
    const block = make('div', {class: 'diff'});
    for (const [op, line] of value.diff || []) {
      block.append(make('div', {class: op === '+' ? 'add' : op === '-' ? 'remove' : '', text: (op === '=' ? '  ' : op + ' ') + line}));
    }
    el.append(block);
  }};

  // A safe subset of Markdown: every character is text; only headings, emphasis, code and lists are styled.
  // Each piece of text keeps its place in the source, so a span marks the source.
  function markdown(source, marks = []) {
    const root = make('div', {class: 'markdown'});
    let list = null, at = 0;
    for (const line of source.split('\n')) {
      const heading = line.match(/^(#{1,3})\s+(.*)$/d), item = line.match(/^\s*[-*]\s+(.*)$/d);
      if (item) {
        if (!list) root.append(list = make('ul'));
        list.append(inline(make('li'), item[1], at + item.indices[1][0], marks));
      } else {
        list = null;
        if (heading) root.append(inline(make(`h${heading[1].length}`), heading[2], at + heading.indices[2][0], marks));
        else if (line.trim()) root.append(inline(make('p'), line, at, marks));
      }
      at += line.length + 1;
    }
    return root;
  }
  function inline(node, text, at, marks) {
    for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/)) {
      if (/^\*\*.+\*\*$/.test(part)) node.append(make('strong', {}, ...pieces(part.slice(2, -2), at + 2, marks)));
      else if (/^`.+`$/.test(part)) node.append(make('code', {}, ...pieces(part.slice(1, -1), at + 1, marks)));
      else if (/^\*.+\*$/.test(part)) node.append(make('em', {}, ...pieces(part.slice(1, -1), at + 1, marks)));
      else node.append(...pieces(part, at, marks));
      at += part.length;
    }
    return node;
  }
  // Text as pieces that know where they start in the field (data-at, in UTF-16 units), so a
  // selection maps back to offsets. `marks` ({start, end, index, className, label}) are painted in.
  function pieces(text, at, marks = []) {
    const end = at + text.length, cuts = new Set([at, end]);
    for (const mark of marks) if (mark.end > at && mark.start < end) cuts.add(Math.max(at, mark.start)).add(Math.min(end, mark.end));
    const sorted = [...cuts].sort((a, b) => a - b), nodes = [];
    for (let i = 0; i + 1 < sorted.length; i++) {
      const [from, to] = [sorted[i], sorted[i + 1]];
      const mark = marks.find(each => each.start <= from && to <= each.end);
      nodes.push(make(mark ? 'mark' : 'span', {'data-at': from, 'data-mark': mark?.index, class: mark?.className,
        text: text.slice(from - at, to - at)}));
      if (mark?.label && to === mark.end) nodes.push(make('span', {class: `mark-tag ${mark.className}`, 'aria-hidden': 'true', text: mark.label}));
    }
    return nodes;
  }

  // Question kinds: render(container, ctx) where ctx = {question, pane, value, set(value), key(option)}.

  // A key shows on its button but is not part of the button's name; aria-keyshortcuts names it.
  const badge = key => key ? make('kbd', {text: key, 'aria-hidden': 'true'}) : null;
  function buttons(options, ctx, pressed) {
    return make('div', {class: 'options', role: 'group', 'aria-label': ctx.question.prompt},
      ...options.map(option => make('button', {type: 'button', 'aria-pressed': String(pressed(option.value)),
        'aria-keyshortcuts': ctx.key(option.value), onclick: () => ctx.set(option.value)}, option.label, badge(ctx.key(option.value)))));
  }
  ModelLab.kinds.choice = {
    keyed: question => question.options.map(option => ({value: option.value, key: option.key})),
    render: (el, ctx) => el.append(buttons(ctx.question.options, ctx, value => ctx.value === value)),
  };
  ModelLab.kinds.multi = {render(el, ctx) {
    const chosen = ctx.value || [];
    el.append(buttons(ctx.question.options, {...ctx, key: option => ctx.key(option),
      set: value => ctx.set(chosen.includes(value) ? chosen.filter(v => v !== value) : [...chosen, value])},
      value => chosen.includes(value)));
  }};
  const scaleValues = question => Array.from({length: question.range[1] - question.range[0] + 1}, (_, i) => question.range[0] + i);
  ModelLab.kinds.scale = {
    keyed: question => scaleValues(question).length <= 9 ? scaleValues(question).map(value => ({value})) : [],
    render(el, ctx) {
      const values = scaleValues(ctx.question), labels = ctx.question.labels || {};
      const box = make('div', {class: 'scale'});
      el.append(box);
      el = box;
      if (values.length > 11) {
        const input = make('input', {type: 'range', min: values[0], max: values.at(-1), value: ctx.value ?? values[0],
          'aria-label': ctx.question.prompt, onchange: event => ctx.set(Number(event.target.value))});
        el.append(input, make('span', {text: ` ${ctx.value ?? '—'}`}));
      } else {
        el.append(buttons(values.map(value => ({value, label: String(value)})), ctx, value => ctx.value === value));
      }
      if (Object.keys(labels).length) {
        el.append(make('div', {class: 'scale-labels'}, ...values.filter(v => labels[v]).map(v => make('span', {text: `${v}: ${labels[v]}`}))));
      }
    },
  };
  ModelLab.kinds.number = {render(el, ctx) {
    const [low, high] = ctx.question.range || [];
    const input = make('input', {type: 'number', inputmode: 'decimal', step: 'any', min: low, max: high,
      value: ctx.value ?? '', 'aria-label': ctx.question.prompt,
      oninput: event => ctx.set(event.target.value === '' ? null : Number(event.target.value), {quiet: true})});
    el.append(input, ctx.question.unit ? make('span', {text: ` ${ctx.question.unit}`}) : null);
  }};
  ModelLab.kinds.text = {render(el, ctx) {
    el.append(make('textarea', {rows: 2, 'aria-label': ctx.question.prompt, text: ctx.value ?? '',
      oninput: event => ctx.set(event.target.value === '' ? null : event.target.value, {quiet: true})}));
  }};
  ModelLab.kinds.rank = {
    complete: (question, value) => value.flat().length === current().order.filter(field => question.panes.includes(field)).length,
    keyed: (question, order) => {
      if (question.panes.length !== 2) return [];
      const [left, right] = order.filter(field => question.panes.includes(field));
      return [{value: JSON.stringify([[left], [right]])}, ...(question.ties ? [{value: JSON.stringify([[left, right]])}] : []),
              {value: JSON.stringify([[right], [left]])}];
    },
    render(el, ctx) {
      const shown = current().order.filter(field => ctx.question.panes.includes(field));
      const label = field => labelOf[field];
      if (shown.length === 2) {
        const [left, right] = shown;
        const options = [{value: JSON.stringify([[left], [right]]), label: `${label(left)} is better`},
          ...(ctx.question.ties ? [{value: JSON.stringify([[left, right]]), label: 'Equal'}] : []),
          {value: JSON.stringify([[right], [left]]), label: `${label(right)} is better`}];
        el.append(buttons(options, {...ctx, set: value => ctx.set(JSON.parse(value))}, value => JSON.stringify(ctx.value) === value));
        return;
      }
      const order = (ctx.value || []).map(group => group[0]);
      el.append(make('div', {class: 'options'}, ...shown.map(field => make('button', {type: 'button',
        'aria-pressed': String(order.includes(field)),
        onclick: () => ctx.set(order.includes(field) ? order.filter(f => f !== field).map(f => [f]) : [...order, field].map(f => [f]))},
        order.includes(field) ? `${order.indexOf(field) + 1}. ${label(field)}` : label(field)))),
        make('div', {class: 'scale-labels', text: 'Click from best to worst; click again to take one out.'}));
    },
  };

  // Marks. A span marks parts of a text: select it, and it is marked (with several tags, a tag's
  // key or button marks it). A region is a box dragged on an image or a paused video frame. A
  // mark clicked is chosen: a tag's key retags it, Backspace removes it, Esc lets it go. The
  // server counts span offsets in code points, as Python does; the page works in UTF-16 units.
  const toPoints = (text, units) => Array.from(text.slice(0, units)).length;
  let picked = null;   // {field, index}: the chosen mark
  let pending = null;  // {field, box}: a box drawn where several tags are offered, waiting for one
  let armed = null;    // the selection as it was when a tag button was pressed (a tap may clear it)
  let stages = {};     // field -> {layer, media, observer}: where boxes are drawn
  let painted = {};    // field -> the span marks its text was drawn with
  const markOn = field => instances().find(({question, pane}) => pane === field && ModelLab.kinds[question.kind]?.marks && asked(question, pane));
  const tagClass = (question, tag) => `t${Math.max(0, question.tags.findIndex(each => each.value === tag)) % 6}`;
  const tagLabel = (question, tag) => question.tags.length > 1 ? question.tags.find(each => each.value === tag)?.label ?? tag : null;
  const clock = seconds => `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(2).padStart(5, '0')}`;
  const tagged = (value, tag) => tag == null ? value : {...value, tag};

  function markKind(kind) {
    return {
      marks: true,
      empty: [],  // no marks is an answer; `required` asks for at least one
      complete: (question, value) => !question.required || value.length > 0,
      render(el, ctx) {
        const {question, pane: field} = ctx, marks = ctx.value || [], tags = question.tags;
        const chosen = picked?.field === field ? marks[picked.index] : null;
        const waiting = pending?.field === field;
        const video = stages[field]?.media.tagName === 'VIDEO';
        el.append(make('div', {class: 'scale-labels', text: chosen ? 'Chosen: a tag\'s key retags it, Backspace removes it, Esc lets it go.'
          : waiting ? 'Pick a tag for the new box, or press Esc to drop it.'
          : kind === 'span' ? (tags.length > 1 ? 'Select text, then pick its tag.' : 'Select text to mark it.')
          : video ? 'Pause, then drag on the frame to draw a box.' : 'Drag on the picture to draw a box.'}));
        // Buttons for tags, and for marking a selection by touch; a key shows on each.
        const offered = kind === 'span' ? (tags.length ? tags : [{value: null, label: 'Mark the selection'}]) : tags.length > 1 ? tags : [];
        if (offered.length) {
          el.append(make('div', {class: 'options tags', role: 'group', 'aria-label': `${question.prompt}: tags`},
            ...offered.map(tag => make('button', {type: 'button', class: `tagged ${tag.value == null ? 'plain' : tagClass(question, tag.value)}`,
              'data-tag': field, 'aria-pressed': String(chosen ? chosen.tag === tag.value : false), 'aria-keyshortcuts': ctx.key(tag.value),
              onpointerdown: event => { event.preventDefault(); armed = selectedSpan(); },
              onclick: () => { const span = armed ?? selectedSpan(); armed = null; applyTag(question, field, tag.value, span); }},
            tag.label, badge(ctx.key(tag.value))))));
        }
        if (marks.length) {
          el.append(make('div', {class: 'marks'}, ...marks.map((mark, index) => {
            const name = kind === 'span' ? `“${mark.text.length > 48 ? `${mark.text.slice(0, 47)}…` : mark.text}”`
              : `Box ${index + 1}${mark.time != null ? ` at ${clock(mark.time)}` : ''}`;
            const tag = tagLabel(question, mark.tag);
            return make('span', {class: `chip ${tagClass(question, mark.tag)}`},
              make('button', {type: 'button', 'aria-pressed': String(picked?.field === field && picked.index === index),
                onclick: () => pick(field, index)}, name, tag ? make('span', {class: 'chip-tag', text: ` · ${tag}`}) : null),
              make('button', {type: 'button', class: 'remove', 'aria-label': `Remove ${name}`, onclick: () => removeMark(field, index)}, '×'));
          })));
        }
      },
    };
  }
  ModelLab.kinds.span = markKind('span');
  ModelLab.kinds.region = markKind('region');

  // The selection as a span of the field it lies in, trimmed of spaces; null when it is not one.
  function selectedSpan() {
    const selection = document.getSelection();
    if (!current() || !selection?.rangeCount || selection.isCollapsed) return null;
    const range = selection.getRangeAt(0), common = range.commonAncestorContainer;
    const content = (common.nodeType === Node.ELEMENT_NODE ? common : common.parentElement)?.closest('.pane .content');
    const field = content?.closest('.pane').dataset.field, found = field && markOn(field);
    if (!found || found.question.kind !== 'span') return null;
    const text = String(current().fields[field]);
    let start = unitsAt(content, range.startContainer, range.startOffset, 'start');
    let end = unitsAt(content, range.endContainer, range.endOffset, 'end');
    while (start < end && /\s/.test(text[start])) start++;
    while (end > start && /\s/.test(text[end - 1])) end--;
    return start < end ? {question: found.question, field, text, start, end} : null;
  }
  // Between pieces, a start moves to where the next piece starts and an end to where the one
  // before ends, so markup between them (a list's "-", a heading's "#") is not taken.
  function unitsAt(content, node, offset, side) {
    const piece = node.nodeType === Node.TEXT_NODE ? node.parentElement.closest('[data-at]') : null;
    if (piece && content.contains(piece)) return Number(piece.dataset.at) + offset;
    const point = document.createRange();
    point.setStart(node, offset);
    const all = [...content.querySelectorAll('[data-at]')];
    const from = each => Number(each.dataset.at), to = each => from(each) + each.textContent.length;
    if (side === 'start') {
      const next = all.find(each => point.comparePoint(each, 0) >= 0);
      return next ? from(next) : all.length ? to(all.at(-1)) : 0;
    }
    const before = all.findLast(each => point.comparePoint(each, each.childNodes.length) <= 0);
    return before ? to(before) : all.length ? from(all[0]) : 0;
  }
  function addSpan({question, field, text, start, end}, tag) {
    const mark = tagged({start: toPoints(text, start), end: toPoints(text, end), text: text.slice(start, end)}, tag);
    const marks = answerOf(question, field) || [];
    document.getSelection().removeAllRanges();
    const inside = marks.findIndex(each => each.start <= mark.start && mark.end <= each.end);
    if (inside >= 0) {  // a selection within a mark (as a double-click makes) chooses it rather than replacing it
      picked = {field, index: inside};
      if (tag != null && marks[inside].tag !== tag) setAnswer(question, field, marks.map((each, i) => i === inside ? {...each, tag} : each));
      else renderQuestions();
      return;
    }
    picked = null;
    const kept = marks.filter(each => each.end <= mark.start || each.start >= mark.end);  // it replaces marks it overlaps
    setAnswer(question, field, [...kept, mark].sort((a, b) => a.start - b.start));
  }
  function applyTag(question, field, tag, span = selectedSpan()) {
    if (question.kind === 'span' && span?.field === field) addSpan(span, tag);
    else if (pending?.field === field) addBox(question, field, pending.box, tag);
    else if (picked?.field === field && tag != null) {
      const marks = [...answerOf(question, field)];
      marks[picked.index] = {...marks[picked.index], tag};
      setAnswer(question, field, marks);
    }
  }
  function pick(field, index) {
    picked = {field, index};
    pending = null;
    const mark = answerOf(markOn(field).question, field)[index], video = stages[field]?.media;
    if (mark.time != null && video?.tagName === 'VIDEO') { video.pause(); video.currentTime = mark.time; }
    renderQuestions();
  }
  function removeMark(field, index) {
    const {question} = markOn(field);
    picked = null;
    setAnswer(question, field, answerOf(question, field).filter((_, i) => i !== index));
  }
  function letGo() {
    if (!picked && !pending && !selectedSpan()) return false;
    picked = pending = null;
    document.getSelection().removeAllRanges();
    renderQuestions();
    return true;
  }
  // Tag buttons act on the selection or the box drawn, or retag the chosen mark; otherwise they wait.
  function refreshTags() {
    const span = selectedSpan();
    for (const button of document.querySelectorAll('button[data-tag]')) {
      const field = button.dataset.tag, found = markOn(field);
      const ready = found?.question.kind === 'span' ? span?.field === field : pending?.field === field;
      button.disabled = busy || !(ready || (picked?.field === field && !button.classList.contains('plain')));
    }
  }

  // Boxes: a layer over the picture as shown (object-fit leaves bars), in fractions of it.
  function stage(field) {
    const content = document.querySelector(`.pane[data-field="${CSS.escape(field)}"] .content`);
    const media = content?.querySelector('img, video');
    if (!media) return;
    const layer = make('div', {class: 'regions', 'data-field': field});
    content.classList.add('staged');
    content.append(layer);
    const place = () => {
      const [width, height] = media.tagName === 'IMG' ? [media.naturalWidth, media.naturalHeight] : [media.videoWidth, media.videoHeight];
      if (!width || !height) return;
      const scale = Math.min(media.clientWidth / width, media.clientHeight / height);
      Object.assign(layer.style, {left: `${media.offsetLeft + (media.clientWidth - width * scale) / 2}px`,
        top: `${media.offsetTop + (media.clientHeight - height * scale) / 2}px`, width: `${width * scale}px`, height: `${height * scale}px`});
    };
    const observer = new ResizeObserver(place);
    observer.observe(media);
    media.addEventListener(media.tagName === 'IMG' ? 'load' : 'loadedmetadata', place);
    for (const event of ['pause', 'play', 'seeked', 'timeupdate']) media.addEventListener(event, () => paintBoxes(field));
    stages[field] = {layer, media, observer};
    place();

    let drag = null;
    const at = event => {
      const box = layer.getBoundingClientRect();
      return [Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)), Math.min(1, Math.max(0, (event.clientY - box.top) / box.height))];
    };
    const span = (from, to) => ({x: Math.min(from[0], to[0]), y: Math.min(from[1], to[1]), w: Math.abs(to[0] - from[0]), h: Math.abs(to[1] - from[1])});
    layer.addEventListener('pointerdown', event => {
      if (event.button !== 0 || busy || !markOn(field)) return;
      event.preventDefault();
      if (media.tagName === 'VIDEO') lastVideo = media;
      layer.setPointerCapture(event.pointerId);
      if (media.tagName === 'VIDEO') media.pause();
      drag = {from: at(event), box: null, on: event.target.closest('.box[data-index]')};
    });
    layer.addEventListener('pointermove', event => {
      if (!drag) return;
      const box = span(drag.from, at(event)), size = layer.getBoundingClientRect();
      if (!drag.box && box.w * size.width < 4 && box.h * size.height < 4) return;  // still a click
      drag.box = box;
      layer.querySelector('.draft')?.remove();
      layer.append(boxNode(box, 'draft', null));
    });
    layer.addEventListener('pointerup', event => {
      const done = drag, size = layer.getBoundingClientRect();
      drag = null;
      if (!done) return;
      if (done.box && done.box.w * size.width >= 4 && done.box.h * size.height >= 4) drew(field, span(done.from, at(event)));
      else if (done.on) pick(field, Number(done.on.dataset.index));
      else if (media.tagName === 'IMG') openViewer(task.layout.compare.includes(field) ? onScreen(task.layout.compare) : [field]);
      else paintBoxes(field);
    });
    layer.addEventListener('pointercancel', () => { drag = null; paintBoxes(field); });
  }
  function drew(field, box) {
    const {question} = markOn(field), media = stages[field].media;
    const round = value => Math.round(value * 10000) / 10000;
    const x = round(box.x), y = round(box.y);
    const drawn = {x, y, w: Math.min(round(box.w), round(1 - x)), h: Math.min(round(box.h), round(1 - y))};
    if (media.tagName === 'VIDEO') drawn.time = Math.round(media.currentTime * 1000) / 1000;
    if (question.tags.length > 1) { pending = {field, box: drawn}; picked = null; renderQuestions(); return; }
    addBox(question, field, drawn, question.tags[0]?.value);
  }
  function addBox(question, field, box, tag) {
    pending = picked = null;
    setAnswer(question, field, [...(answerOf(question, field) || []), tagged(box, tag)]);
  }
  function boxNode(box, className, label, index) {
    const node = make('div', {class: `box ${className}`, 'data-index': index}, label ? make('span', {text: label}) : null);
    Object.assign(node.style, {left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%`});
    return node;
  }
  // A video shows the boxes of the frame it is paused on.
  function paintBoxes(field) {
    const place = stages[field];
    if (!place) return;
    const found = markOn(field), video = place.media.tagName === 'VIDEO' ? place.media : null;
    const here = box => !video || (video.paused && Math.abs(box.time - video.currentTime) < 0.02);
    place.layer.classList.toggle('off', !found);
    const marks = found ? answerOf(found.question, field) || [] : [];
    const name = (mark, index) => [index + 1, tagLabel(found.question, mark.tag)].filter(part => part != null).join(' · ');  // as its chip says
    place.layer.replaceChildren(...marks.flatMap((mark, index) => here(mark) ? [boxNode(mark,
      `${tagClass(found.question, mark.tag)}${picked?.field === field && picked.index === index ? ' picked' : ''}`,
      name(mark, index), index)] : []),
      ...(found && pending?.field === field && here(pending.box) ? [boxNode(pending.box, 'pending', 'tag?')] : []));
  }
  function paint() {
    for (const {question, pane} of instances()) {
      if (question.kind === 'span' && painted[pane] !== JSON.stringify(spanMarks(pane))) renderContent(pane);
      else if (question.kind === 'region') paintBoxes(pane);
    }
  }

  // A region video's controls: play, a frame back and forward (1/30 s), and the time.
  function videoBar(video) {
    const play = make('button', {type: 'button', onclick: () => (video.paused ? video.play() : video.pause())}, 'Play', badge('Space'));
    const time = make('input', {type: 'range', min: 0, max: 1000, value: 0, 'aria-label': 'Time',
      oninput: event => { video.pause(); video.currentTime = event.target.value / 1000 * (video.duration || 0); }});
    const shown = make('span', {class: 'time', text: clock(0)});
    const update = () => {
      play.firstChild.textContent = video.paused ? 'Play' : 'Pause';
      if (document.activeElement !== time) time.value = video.duration ? Math.round(video.currentTime / video.duration * 1000) : 0;
      shown.textContent = clock(video.currentTime);
    };
    for (const event of ['play', 'pause', 'timeupdate', 'seeked', 'loadedmetadata']) video.addEventListener(event, update);
    return make('div', {class: 'video-bar', onpointerdown: () => { lastVideo = video; }}, play,
      make('button', {type: 'button', 'aria-label': 'Back one frame', 'aria-keyshortcuts': ',', onclick: () => stepFrame(video, -1)}, '‹', badge(',')),
      make('button', {type: 'button', 'aria-label': 'Forward one frame', 'aria-keyshortcuts': '.', onclick: () => stepFrame(video, 1)}, '›', badge('.')),
      time, shown);
  }
  function stepFrame(video, frames) {
    video.pause();
    video.currentTime = Math.min(video.duration || 0, Math.max(0, video.currentTime + frames / 30));
  }
  let lastVideo = null;  // the region video last drawn on or controlled, which the keys control
  const regionVideo = () => {
    const videos = Object.values(stages).map(place => place.media).filter(media => media.tagName === 'VIDEO');
    return videos.includes(lastVideo) ? lastVideo : videos[0];
  };

  // Which questions are asked now, and whether the answer is complete.

  const perPane = question => question.panes.length > 0 && question.kind !== 'rank';
  function answerOf(question, pane) {
    const value = draft[question.id];
    return perPane(question) ? (value || {})[pane] : value;
  }
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  function holds(rule, answers, pane) {
    return Object.entries(rule).every(([id, values]) => {
      let answer = answers[id];
      if (answer && typeof answer === 'object' && !Array.isArray(answer)) answer = pane ? answer[pane] : undefined;
      return values.some(value => same(value, answer));
    });
  }
  const askedIn = (question, answers, pane) => holds(question.when, answers, pane)
    && !(Object.keys(question.unless).length && holds(question.unless, answers, pane));
  function instances() {
    const shown = current().order;
    return task.questions.flatMap(question => perPane(question)
      ? question.panes.filter(pane => shown.includes(pane)).map(pane => ({question, pane}))
      : [{question, pane: null}]);
  }
  const complete = (question, value) => !ModelLab.kinds[question.kind]?.complete || ModelLab.kinds[question.kind].complete(question, value);
  // Questions in order, as the server checks them: a question is asked when its conditions hold
  // on the complete answers kept so far. `kept` is what a save sends; `gaps` are required and open.
  function walk() {
    const kept = {}, gaps = [];
    for (const {question, pane} of instances()) {
      if (!askedIn(question, kept, pane)) continue;
      const value = answerOf(question, pane) ?? ModelLab.kinds[question.kind]?.empty;
      if (value == null || !complete(question, value)) { if (question.required) gaps.push({question, pane}); continue; }
      if (perPane(question)) (kept[question.id] ||= {})[pane] = value; else kept[question.id] = value;
    }
    return {kept, gaps};
  }
  const asked = (question, pane) => askedIn(question, walk().kept, pane);
  const missing = () => walk().gaps;
  const pruned = () => walk().kept;

  function setAnswer(question, pane, value, {quiet = false} = {}) {
    if (busy) return;
    if (picked && !(ModelLab.kinds[question.kind]?.marks && pane === picked.field)) picked = null;  // chosen for this answer only
    if (perPane(question)) {
      draft[question.id] = {...(draft[question.id] || {})};
      if (value == null) delete draft[question.id][pane]; else draft[question.id][pane] = value;
    } else if (value == null) delete draft[question.id]; else draft[question.id] = value;
    dirty = true;
    if (quiet) { refreshState(); return; }  // typing: keep focus, no redraw
    renderQuestions();
    if (autoSaves()) save();
  }
  // The task's auto_save condition holds for every pane each named question is asked about.
  function autoSaves() {
    const rule = task.queue.auto_save, {kept, gaps} = walk();
    if (!Object.keys(rule).length || $('details').open || gaps.length || pending) return false;
    return Object.entries(rule).every(([id, values]) => instances().filter(({question}) => question.id === id)
      .every(({question, pane}) => {
        const value = perPane(question) ? (kept[id] || {})[pane] : kept[id];
        return values.some(allowed => same(allowed, value));
      }));
  }

  // Drawing an item.

  function renderItem() {
    const item = current();
    panesByField = Object.fromEntries(task.panes.map(pane => [pane.field, pane]));
    // Shuffled panes are named by where they stand (A, B, ...), so a label cannot say which is which.
    labelOf = Object.fromEntries(task.panes.map(pane => [pane.field, pane.label]));
    item.order.filter(field => task.queue.shuffle.includes(field)).forEach((field, place) => { labelOf[field] = String.fromCharCode(65 + place); });
    const holder = $('panes');
    holder.className = task.layout.arrange;
    holder.replaceChildren();
    $('held').replaceChildren();
    flicked = false;
    for (const place of Object.values(stages)) place.observer.disconnect();
    stages = {};
    painted = {};
    for (const field of item.order) {
      const pane = {...panesByField[field], label: labelOf[field]};
      const shuffled = task.queue.shuffle.includes(field);
      const section = make('section', {class: 'pane', 'data-field': field, hidden: pane.hidden},
        make('header', {}, make('h2', {text: pane.label}), pane.target && !shuffled ? make('span', {class: 'tag', text: 'target'}) : null),
        make('div', {class: 'content'}), make('div', {class: 'questions'}));
      holder.append(section);
      renderContent(field);
      if (task.questions.some(question => question.kind === 'region' && question.panes[0] === field)) stage(field);
      if (pane.hidden) {
        $('held').append(make('button', {type: 'button', 'aria-pressed': 'false', 'data-field': field, title: `Hold ${pane.key.toUpperCase()} to show, or click to keep it shown`,
          onclick: event => { const open = section.hidden; section.hidden = !open; event.currentTarget.setAttribute('aria-pressed', String(open)); }},
          pane.label, badge(`hold ${pane.key}`)));
      }
    }
    assignKeys();
    renderQuestions();
  }

  function renderContent(field) {
    const content = document.querySelector(`.pane[data-field="${CSS.escape(field)}"] .content`);
    const pane = {...panesByField[field], label: labelOf[field]}, value = current().fields[field], display = ModelLab.displays[pane.show];
    const top = content.querySelector('.text')?.scrollTop ?? 0;  // repainting marks keeps the place
    const marks = spanMarks(field);
    painted[field] = JSON.stringify(marks);
    content.replaceChildren();
    if (value == null) content.append(make('p', {class: 'tag', text: 'Nothing here for this item.'}));
    else if (!display) content.append(make('p', {class: 'tag', text: `No display named ${pane.show}.`}));
    else display.render(content, value, {pane, item: current(), compare: task.layout.compare, marks,
      region: task.questions.some(question => question.kind === 'region' && question.panes[0] === field)});
    const text = content.querySelector('.text');
    if (text) text.scrollTop = top;
  }
  function spanMarks(field) {
    const found = markOn(field);
    if (found?.question.kind !== 'span') return [];
    const units = [0];  // code point -> UTF-16 unit, once per text
    for (const character of String(current().fields[field])) units.push(units.at(-1) + character.length);
    return (answerOf(found.question, field) || []).map((mark, index) => ({start: units[mark.start], end: units[mark.end], index,
      className: `${tagClass(found.question, mark.tag)}${picked?.field === field && picked.index === index ? ' picked' : ''}`,
      label: tagLabel(found.question, mark.tag)}));
  }

  // Keys, in this order: keys the task gives; a scale's own digits (5 means 5); a single choice's
  // first letter when no other option on screen shares it; then the free digits, in order.
  const RESERVED = new Set(['d', 's', 'f']);
  function assignKeys() {
    keys = new Map();
    const order = current().order;
    const taken = new Set([...RESERVED, ...task.panes.filter(pane => pane.key).map(pane => pane.key.toLowerCase()),
      ...task.questions.flatMap(question => [...question.options || [], ...question.tags || []].map(option => option.key?.toLowerCase()).filter(Boolean))]);
    const keyed = instances().filter(({question}) => !question.details && ModelLab.kinds[question.kind]);
    const bind = (key, question, pane, value) => { taken.add(key); keys.set(key, {question, pane, value}); };
    const later = [];
    for (const {question, pane} of keyed) {
      for (const option of [...question.options || [], ...question.tags || []]) if (option.key) keys.set(option.key.toLowerCase(), {question, pane, value: option.value});
    }
    for (const {question, pane} of keyed.filter(({question}) => question.kind === 'scale')) {
      const values = ModelLab.kinds.scale.keyed(question, order).map(option => option.value);
      if (values.length && values.every(value => value >= 1 && value <= 9 && !taken.has(String(value)))) {
        values.forEach(value => bind(String(value), question, pane, value));
      } else later.push(...values.map(value => ({question, pane, value})));
    }
    // A single choice's options and a mark's tags (when there are several to pick from).
    const lettered = keyed.flatMap(({question, pane}) => (question.kind === 'choice' ? question.options
      : ModelLab.kinds[question.kind].marks && question.tags.length > 1 ? question.tags : [])
      .filter(option => !option.key).map(option => ({question, pane, option, letter: option.label.trim()[0]?.toLowerCase()})));
    for (const {question, pane, option, letter} of lettered) {
      if (/^[a-z]$/.test(letter) && lettered.filter(other => other.letter === letter).length === 1 && !taken.has(letter)) {
        bind(letter, question, pane, option.value);
      } else later.push({question, pane, value: option.value});
    }
    for (const {question, pane} of keyed.filter(({question}) => question.kind === 'rank')) {
      later.push(...ModelLab.kinds.rank.keyed(question, order).map(option => ({question, pane, value: option.value})));
    }
    let digit = 1;
    for (const {question, pane, value} of later) {
      while (digit <= 9 && taken.has(String(digit))) digit++;
      if (digit > 9) break;
      bind(String(digit), question, pane, value);
    }
  }
  const keyFor = (question, pane) => value => {
    for (const [key, target] of keys) {
      if (target.question === question && target.pane === pane && (target.value === value || target.value === JSON.stringify(value) || String(target.value) === String(value))) return key;
    }
    return null;
  };

  function renderQuestions() {
    const holders = {item: $('item-questions'), details: $('detail-questions')};
    for (const node of [holders.item, holders.details, ...document.querySelectorAll('.pane .questions')]) node.replaceChildren();
    for (const {question, pane} of instances()) {
      const kind = ModelLab.kinds[question.kind];
      const block = make('div', {class: 'question', hidden: !asked(question, pane)},
        make('div', {class: 'prompt'}, make('b', {text: question.prompt}), question.required ? null : ' · optional',
          question.help ? make('span', {text: ` — ${question.help}`}) : null));
      if (!kind) block.append(make('p', {class: 'tag', text: `No question kind named ${question.kind}.`}));
      else kind.render(block, {question, pane, value: answerOf(question, pane), key: keyFor(question, pane),
        set: (value, how) => setAnswer(question, pane, value, how)});
      const home = question.details ? holders.details
        : pane ? document.querySelector(`.pane[data-field="${CSS.escape(pane)}"] .questions`) : holders.item;
      home.append(block);
    }
    for (const node of document.querySelectorAll('.pane .questions')) node.hidden = !node.children.length;
    $('details').hidden = !holders.details.children.length;
    paint();
    refreshTags();
    refreshState();
  }

  function refreshState() {
    const item = current();
    const answered = items.filter(entry => entry.answer && !entry.answer.skip).length;
    $('progress').replaceChildren(`${answered} / ${items.length}`, make('span', {class: 'wide', text: ' answered'}));
    $('position').textContent = item ? `Item ${index + 1} of ${items.length}` : '';
    const gaps = item ? missing() : [];
    $('save').disabled = busy || !item || gaps.length > 0 || !!pending;
    $('skip').disabled = busy || !item;
    $('previous').disabled = busy || index === 0;
    $('hint').textContent = !item ? '' : busy ? 'Saving…' : pending ? 'Tag the new box first, or press Esc to drop it'
      : gaps.length ? `To answer: ${[...new Set(gaps.map(gap => gap.question.prompt))].join(', ')}`
      : item.answer && !dirty ? 'Saved · change anything, then Save & next' : 'Ready to save';
  }

  // Loading, saving, moving.

  async function request(url, options = {}) {
    let response;
    try {
      response = await fetch(url, {cache: 'no-store', ...options});
    } catch {
      throw Object.assign(new Error('Could not reach the labelling server. Your answer is still here.'), {status: 0});
    }
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw Object.assign(new Error(body.error || `The server answered ${response.status}.`), {status: response.status});
    return body;
  }

  function problem(message, action, label, other, otherLabel) {
    $('problem').hidden = false;
    $('problem').querySelector('span').textContent = message;
    $('problem-action').textContent = label;
    retry = action;
    $('problem-other').hidden = !other;
    $('problem-other').textContent = otherLabel || '';
    $('problem-other').onclick = other || null;
  }

  function show() {
    $('problem').hidden = true;
    const item = current();
    $('empty').hidden = items.length > 0;
    $('done').hidden = !items.length || !!item;
    $('work').hidden = !item;
    $('dock').hidden = !item;
    dirty = false;
    picked = pending = null;
    if (!item) { refreshState(); return; }
    draft = item.answer && !item.answer.skip ? JSON.parse(JSON.stringify(item.answer)) : defaults();
    $('details').open = false;
    renderItem();
    window.scrollTo(0, 0);
  }
  function defaults() {
    const values = {};
    for (const question of task.questions) {
      if (question.default == null) continue;
      values[question.id] = perPane(question) ? Object.fromEntries(question.panes.map(pane => [pane, question.default])) : question.default;
    }
    return values;
  }

  async function save(skip = false, then = 'next') {
    const item = current();
    if (busy || !item || (!skip && (missing().length || pending))) return;
    busy = true;
    refreshState();
    const body = skip ? {revision: item.revision, skip: true, order: item.order}
      : {revision: item.revision, answers: pruned(), order: item.order};
    try {
      const reply = await request(`/api/answer/${encodeURIComponent(item.id)}`, {method: 'POST',
        headers: {'Content-Type': 'application/json', 'X-ModelLab': '1'}, body: JSON.stringify(body)});
      item.revision = reply.revision;
      item.answer = skip ? {skip: true} : body.answers;
      busy = false;
      index += then === 'back' ? -1 : 1;
      show();
    } catch (error) {
      busy = false;
      refreshState();
      if (error.status === 409) problem(error.message, () => load(item.id), 'Reload this item');
      else problem(error.message, () => save(skip, then), 'Retry');
    }
  }

  function previous() {
    if (busy || index === 0) return;
    if (dirty) {
      problem('This answer is not saved.', () => save(false, 'back'), 'Save it', () => { dirty = false; previous(); }, 'Leave it');
      return;
    }
    index -= 1;
    show();
  }

  async function load(itemId = null) {
    try {
      items = (await request('/api/items')).items;
    } catch (error) {
      problem(`Could not load the items: ${error.message}`, () => load(itemId), 'Retry');
      return;
    }
    index = itemId ? items.findIndex(item => item.id === itemId) : items.findIndex(item => !item.answer);
    if (index < 0) index = items.length;
    show();
  }

  // Viewing large, comparing, instructions.

  function openViewer(fields) {
    const body = $('viewer-body');
    body.classList.remove('zoomed');
    $('zoom').setAttribute('aria-pressed', 'false');
    body.replaceChildren(...fields.filter(field => current().fields[field]?.url && panesByField[field].show === 'image').map(field =>
      make('figure', {}, make('figcaption', {text: labelOf[field]}),
        make('div', {class: 'scroll', onscroll: syncScroll}, make('img', {src: current().fields[field].url, alt: labelOf[field]})))));
    $('viewer-title').textContent = fields.map(field => labelOf[field]).join(' · ');
    $('viewer').showModal();
    $('close-viewer').focus();
  }
  let syncing = false;
  function syncScroll(event) {
    if (syncing) return;
    syncing = true;
    for (const other of document.querySelectorAll('#viewer-body .scroll')) {
      if (other !== event.target) { other.scrollTop = event.target.scrollTop; other.scrollLeft = event.target.scrollLeft; }
    }
    syncing = false;
  }

  function flick() {
    const [first, second] = onScreen(task.layout.compare);
    if (!first || !second) return;
    const pane = document.querySelector(`.pane[data-field="${CSS.escape(first)}"]`);
    const img = pane?.querySelector('.content img'), other = current().fields[second];
    if (!img || !other?.url) return;
    flicked = !flicked;
    img.src = flicked ? other.url : current().fields[first].url;
    pane.classList.toggle('flicked', flicked);
    if (stages[first]) stages[first].layer.hidden = flicked;  // its boxes belong to its own picture
    pane.querySelector('header h2').textContent = flicked ? `${labelOf[second]} (F to go back)` : labelOf[first];
  }

  function openGuide() {
    const body = $('guide-body');
    body.replaceChildren();
    if (task.instructions) body.append(markdown(task.instructions));
    const rows = [['Enter', 'Save and go to the next item'], ['Backspace', 'Go back'], ['S', 'Skip'], ['D', 'More questions'],
      ['?', 'This page'], ...task.panes.filter(pane => pane.hidden).map(pane => [pane.key.toUpperCase(), `Hold to show ${pane.label}`]),
      ...(task.layout.compare.length > 1 ? [['F', `Flick between ${onScreen(task.layout.compare).slice(0, 2).map(f => labelOf[f] || f).join(' and ')}`]] : []),
      ...(task.questions.some(question => ModelLab.kinds[question.kind]?.marks) ? [['Backspace', 'Remove the chosen mark (click a mark to choose it)'],
        ['Esc', 'Let go of the chosen mark, the new box or the selection']] : []),
      ...(regionVideo() ? [['Space', 'Play or pause'], [',', 'One frame back'], ['.', 'One frame forward']] : []),
      ...[...keys].sort(([a], [b]) => a.localeCompare(b, undefined, {numeric: true})).map(([key, target]) => [key, `${target.question.prompt}${target.pane ? ` (${labelOf[target.pane]})` : ''}: ${
        typeof target.value === 'string' && target.value.startsWith('[') ? JSON.parse(target.value).map(group => group.map(f => labelOf[f]).join(' = ')).join(' > ') : target.value}`])];
    body.append(make('h3', {text: 'Keys'}), make('table', {}, ...rows.map(([key, what]) => make('tr', {}, make('td', {}, make('kbd', {text: key})), make('td', {text: what})))));
    $('guide').showModal();
    $('close-guide').focus();
  }

  // Keys and buttons.

  const keyOf = event => event.key.length === 1 ? event.key.toLowerCase() : event.key;  // Caps Lock changes nothing
  document.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey || !task) return;
    if ($('viewer').open || $('guide').open) return;  // Escape closes them natively
    if (event.target.closest('textarea, input:not([type="range"]), select, [contenteditable="true"]')) return;  // a slider takes only arrows
    const key = keyOf(event);
    const hidden = task.panes.find(pane => pane.hidden && pane.key.toLowerCase() === key);
    if (hidden) {
      const section = document.querySelector(`.pane[data-field="${CSS.escape(hidden.field)}"]`);
      if (section) { section.hidden = false; event.preventDefault(); }
      return;
    }
    const video = regionVideo();
    if (video && [',', '.'].includes(key) && current()) { event.preventDefault(); stepFrame(video, key === ',' ? -1 : 1); return; }
    if (event.repeat) return;
    const target = keys.get(key);
    let action = null;
    if (target) {
      action = () => {
        const kind = target.question.kind;
        if (ModelLab.kinds[kind]?.marks) { applyTag(target.question, target.pane, target.value); return; }
        const value = kind === 'rank' ? JSON.parse(target.value) : target.value;
        setAnswer(target.question, target.pane, kind === 'multi' ? toggle(answerOf(target.question, target.pane) || [], value) : value);
      };
    } else if (key === 'Backspace' && pending) {
      action = letGo;  // the box waiting for its tag
    } else if (key === 'Backspace' && picked && markOn(picked.field)) {
      action = () => removeMark(picked.field, picked.index);
    } else if (key === 'Escape') {
      if (letGo()) event.preventDefault();
      return;
    } else if (key === ' ' && video) {
      // Space presses a button reached by Tab; a button only clicked keeps no claim on it.
      if (event.target.closest('button, summary, a') && event.target.matches(':focus-visible')) return;
      action = () => (video.paused ? video.play() : video.pause());
    } else {
      action = {'Enter': () => save(), 'Backspace': previous, 's': () => save(true), 'd': () => { $('details').open = !$('details').open; },
                'f': flick, '?': openGuide}[key];
    }
    if (action && current()) { event.preventDefault(); action(); }
    else if (key === '?') { event.preventDefault(); openGuide(); }
  });
  document.addEventListener('keyup', event => {
    const key = keyOf(event);
    const hidden = task?.panes.find(pane => pane.hidden && pane.key.toLowerCase() === key);
    const section = hidden && document.querySelector(`.pane[data-field="${CSS.escape(hidden.field)}"]`);
    const button = hidden && $('held').querySelector(`[data-field="${CSS.escape(hidden.field)}"]`);
    if (section && button?.getAttribute('aria-pressed') !== 'true') section.hidden = true;
  });
  const toggle = (list, value) => list.includes(value) ? list.filter(v => v !== value) : [...list, value];

  document.addEventListener('selectionchange', () => { if (task && current()) refreshTags(); });
  $('panes').addEventListener('pointerup', event => {
    if (event.pointerType === 'touch') return;  // a finger's selection is marked by its button
    const span = selectedSpan();  // final by now, so a key pressed next finds the mark made
    if (span && span.question.tags.length <= 1) addSpan(span, span.question.tags[0]?.value);
  });
  $('panes').addEventListener('click', event => {
    const mark = event.target.closest('mark[data-mark]');  // not one just redrawn by the release before this click
    if (mark?.isConnected && document.getSelection().isCollapsed) pick(mark.closest('.pane').dataset.field, Number(mark.dataset.mark));
  });
  $('save').addEventListener('click', () => save());
  $('skip').addEventListener('click', () => save(true));
  $('previous').addEventListener('click', previous);
  $('last').addEventListener('click', () => { index = items.length - 1; show(); });
  $('help').addEventListener('click', openGuide);
  $('problem-action').addEventListener('click', () => { const action = retry; $('problem').hidden = true; action?.(); });
  $('close-viewer').addEventListener('click', () => $('viewer').close());
  for (const dialog of [$('viewer'), $('guide')]) {  // a closed dialog keeps no focus, so the page's keys work again
    dialog.addEventListener('close', () => { if (dialog.contains(document.activeElement)) document.activeElement.blur(); });
  }
  $('close-guide').addEventListener('click', () => $('guide').close());
  $('zoom').addEventListener('click', () => {
    const zoomed = $('viewer-body').classList.toggle('zoomed');
    $('zoom').setAttribute('aria-pressed', String(zoomed));
    $('zoom').textContent = zoomed ? 'Fit' : 'Zoom 2×';
    for (const frame of document.querySelectorAll('#viewer-body .scroll')) {  // start at the middle, not a corner
      frame.scrollTo((frame.scrollWidth - frame.clientWidth) / 2, (frame.scrollHeight - frame.clientHeight) / 2);
    }
  });
  $('details').addEventListener('toggle', refreshState);
  window.addEventListener('beforeunload', event => { if (dirty || busy) { event.preventDefault(); event.returnValue = ''; } });

  async function start() {
    try {
      task = await request('/api/task');
    } catch (error) {
      problem(`Could not load the task: ${error.message}`, start, 'Retry');
      return;
    }
    document.title = `${task.title} · Labelling`;
    $('title').textContent = `${task.title}`;
    for (const path of task.plugins) {
      try {
        await new Promise((resolve, reject) => document.head.append(make('script', {src: `/plugins/${encodeURI(path)}`, onload: resolve,
          onerror: () => reject(new Error(path))})));
      } catch (error) {
        problem(`Could not load the plugin ${error.message}, so this task cannot be shown. Fix it and reload.`, () => location.reload(), 'Reload');
        return;
      }
    }
    let seen = null;
    try { seen = localStorage.getItem(`modellab-guide:${task.title}`); } catch { /* private window: show it again */ }
    await load();
    if (task.instructions && !seen) {
      openGuide();
      try { localStorage.setItem(`modellab-guide:${task.title}`, '1'); } catch { /* not kept; harmless */ }
    }
  }
  start();
})();
