/* ============================================================================
   app.js — Завантаження фото, вибір моделі, виклик API та показ результатів.
   ============================================================================ */

(function () {
  const CITY_COLORS = { kyiv: '#a78bfa', warsaw: '#ff6b6b', prague: '#ffd166', budapest: '#4ecdc4' };

  const fileInput   = document.getElementById('fileInput');
  const dropzone    = document.getElementById('dropzone');
  const dropPrompt  = document.getElementById('dropPrompt');
  const preview     = document.getElementById('preview');
  const modelList   = document.getElementById('modelList');
  const predictBtn  = document.getElementById('predictBtn');
  const btnText     = predictBtn.querySelector('.btn-text');
  const spinner     = predictBtn.querySelector('.spinner');
  const compareBtn  = document.getElementById('compareBtn');
  const compareBox  = document.getElementById('compareBox');

  const placeholder = document.getElementById('resultPlaceholder');
  const resultBox   = document.getElementById('resultBox');
  const oodBanner   = document.getElementById('oodBanner');
  const verdict     = document.getElementById('verdict');
  const bars        = document.getElementById('bars');
  const mapEl       = document.getElementById('map');
  const modelMeta   = document.getElementById('modelMeta');

  const CITY_COORDS = {
    kyiv:     { lat: 50.4501, lon: 30.5234, ua: 'Київ' },
    warsaw:   { lat: 52.2297, lon: 21.0122, ua: 'Варшава' },
    prague:   { lat: 50.0755, lon: 14.4378, ua: 'Прага' },
    budapest: { lat: 47.4979, lon: 19.0402, ua: 'Будапешт' },
  };
  let leafletMap = null;
  let mapLayer = null;   // група маркерів/кіл поточного результату

  let imageDataUrl = null;
  let selectedModel = null;
  let modelMetaMap = {};
  let lastPredict = null;   // {image, model} останнього передбачення (для attention)
  let lastResultData = null;  // останній результат (для перемальовування при зміні мови)
  let lastCompareData = null;
  const T = k => (window.i18n ? window.i18n.t(k) : k);

  const attnWrap   = document.getElementById('attnWrap');
  const attnBtn    = document.getElementById('attnBtn');
  const attnResult = document.getElementById('attnResult');
  const attnImg    = document.getElementById('attnImg');

  // ── Завантаження списку моделей ────────────────────────────────────────
  fetch('/api/models')
    .then(r => r.json())
    .then(data => renderModels(data.models || []))
    .catch(() => {
      modelList.innerHTML = '<p class="muted">Не вдалося завантажити список моделей.</p>';
    });

  // ── Приклади для швидкого старту ───────────────────────────────────────
  const examplesBox   = document.getElementById('examples');
  const exampleGroups = document.getElementById('exampleGroups');
  fetch('/static/examples/manifest.json')
    .then(r => r.json())
    .then(renderExamples)
    .catch(() => { /* галерея необов'язкова */ });

  function renderExamples(man) {
    if (!man || !man.cities) return;
    const groups = [...man.cities];
    if (man.ood) groups.push({ id: 'ood', ua: man.ood.ua, photos: man.ood.photos, ood: true });
    exampleGroups.innerHTML = '';
    groups.forEach(g => {
      const color = CITY_COLORS[g.id] || 'var(--warn)';
      const el = document.createElement('div');
      el.className = 'ex-group' + (g.ood ? ' ood' : '');
      const thumbs = g.photos.map(p =>
        `<img class="ex-thumb" src="./${p}" alt="${g.ua}" loading="lazy" data-src="./${p}">`).join('');
      el.innerHTML =
        `<div class="ex-group-title"><span class="ex-dot" style="background:${color}"></span>${g.ua}</div>
         <div class="ex-thumbs">${thumbs}</div>`;
      exampleGroups.appendChild(el);
    });
    exampleGroups.querySelectorAll('.ex-thumb').forEach(img =>
      img.addEventListener('click', () => loadExample(img.dataset.src)));
    examplesBox.hidden = false;
  }

  function loadExample(url) {
    fetch(url).then(r => r.blob()).then(blob => {
      const reader = new FileReader();
      reader.onload = () => {
        imageDataUrl = reader.result;
        preview.src = imageDataUrl;
        preview.hidden = false;
        dropPrompt.hidden = true;
        updateButton();
      };
      reader.readAsDataURL(blob);
    }).catch(() => showError('Не вдалося завантажити приклад.'));
  }

  function renderModels(models) {
    modelList.innerHTML = '';
    models.forEach(m => {
      modelMetaMap[m.id] = m;
      const card = document.createElement('div');
      card.className = 'model-card' + (m.available ? '' : ' disabled');
      card.dataset.id = m.id;
      card.innerHTML = `
        <span class="mc-radio"></span>
        <div class="mc-body">
          <div class="mc-title">${m.label}</div>
          <div class="mc-sub">${m.subtitle}</div>
        </div>
        <span class="mc-acc" title="Точність на тестовому наборі">${m.accuracy}</span>`;
      if (m.available) {
        card.addEventListener('click', () => selectModel(m.id));
        if (!selectedModel) selectModel(m.id);
      } else {
        card.title = 'Чекпоінт недоступний';
      }
      modelList.appendChild(card);
    });
    updateButton();
  }

  function selectModel(id) {
    selectedModel = id;
    modelList.querySelectorAll('.model-card').forEach(c =>
      c.classList.toggle('selected', c.dataset.id === id));
    updateButton();
    if (window.showAnatomy) {
      window.showAnatomy(id);
    }
  }

  // ── Завантаження зображення ────────────────────────────────────────────
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

  ['dragenter', 'dragover'].forEach(ev =>
    dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach(ev =>
    dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('dragover'); }));
  dropzone.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = () => {
      imageDataUrl = reader.result;
      preview.src = imageDataUrl;
      preview.hidden = false;
      dropPrompt.hidden = true;
      updateButton();
    };
    reader.readAsDataURL(file);
  }

  function updateButton() {
    predictBtn.disabled = !(imageDataUrl && selectedModel);
    compareBtn.disabled = !imageDataUrl;
  }

  // ── Порівняти всі моделі (#6) ──────────────────────────────────────────
  compareBtn.addEventListener('click', () => {
    if (!imageDataUrl) return;
    const cspin = compareBtn.querySelector('.spinner');
    const ctext = compareBtn.querySelector('.btn-text');
    cspin.hidden = false; ctext.textContent = T('dyn.cmp.run'); compareBtn.disabled = true;
    compareBox.hidden = false;
    compareBox.innerHTML = '<p class="muted">⏳ Запускаємо StreetCLIP, GeoCLIP і Baseline…</p>';
    fetch('/api/predict_all', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageDataUrl }),
    })
      .then(r => r.json())
      .then(renderCompare)
      .catch(err => { compareBox.innerHTML = `<p class="muted">Помилка: ${err.message}</p>`; })
      .finally(() => {
        cspin.hidden = true; ctext.textContent = T('btn.compare');
        compareBtn.disabled = !imageDataUrl;
      });
  });

  function renderCompare(data) {
    lastCompareData = data;
    const rows = (data.results || []).map(res => {
      if (res.error) return `<tr><td>${res.model}</td><td colspan="3" class="muted">—</td></tr>`;
      const top = res.predictions[0];
      const color = CITY_COLORS[top.city.toLowerCase()] || '#4f8cff';
      const ood = res.ood && res.ood.is_ood;
      const meta = modelMetaMap[res.model] || {};
      return `<tr>
        <td><b>${meta.label || res.model}</b><small>${meta.accuracy || ''}</small></td>
        <td><span class="ex-dot" style="background:${color}"></span>${top.city_ua}</td>
        <td class="num">${(top.prob * 100).toFixed(1)}%</td>
        <td>${ood ? `<span class="tag-ood">${T('dyn.tag.ood')}</span>` : `<span class="tag-ok">${T('dyn.tag.ok')}</span>`}</td>
      </tr>`;
    }).join('');
    compareBox.innerHTML = `
      <h3 class="compare-title">${T('dyn.cmp.title')}</h3>
      <table class="compare-table">
        <thead><tr><th>${T('dyn.cmp.model')}</th><th>${T('dyn.cmp.city')}</th><th>${T('dyn.cmp.conf')}</th><th>${T('dyn.cmp.ood')}</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  // ── Передбачення ───────────────────────────────────────────────────────
  predictBtn.addEventListener('click', () => {
    if (!imageDataUrl || !selectedModel) return;
    lastPredict = { image: imageDataUrl, model: selectedModel };
    setLoading(true);
    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selectedModel, image: imageDataUrl }),
    })
      .then(async r => {
        const text = await r.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          throw new Error('Некоректна відповідь сервера (HTTP ' + r.status + '). ' +
                          'Переконайтеся, що сервер запущено.');
        }
        if (!r.ok) throw new Error(data.error || ('Помилка сервера (HTTP ' + r.status + ').'));
        return data;
      })
      .then(data => {
        try {
          renderResult(data);
        } catch (e) {
          console.error('renderResult failed:', e, data);
          showError('Не вдалося відобразити результат: ' + e.message);
        }
      })
      .catch(err => {
        console.error('Помилка передбачення:', err);
        showError(err.message || 'Невідома помилка');
      })
      .finally(() => setLoading(false));
  });

  function setLoading(on) {
    predictBtn.disabled = on || !(imageDataUrl && selectedModel);
    spinner.hidden = !on;
    btnText.textContent = on ? 'Обробка…' : 'Визначити місто';
    // Одразу показуємо панель результату з індикатором, щоб клік не виглядав «мертвим».
    if (on) {
      placeholder.hidden = true;
      resultBox.hidden = false;
      oodBanner.hidden = true;
      verdict.innerHTML = '<span class="loading-note">' + T('dyn.loading') + '</span>';
      bars.innerHTML = '';
      modelMeta.textContent = '';
    }
  }

  function showError(msg) {
    placeholder.hidden = true;
    resultBox.hidden = false;
    oodBanner.hidden = false;
    oodBanner.className = 'ood-banner danger';
    oodBanner.innerHTML = `<span>⚠</span><div><b>Помилка</b>${msg}</div>`;
    verdict.innerHTML = ''; bars.innerHTML = ''; modelMeta.textContent = '';
  }

  // ── Відображення результату ────────────────────────────────────────────
  function renderResult(data) {
    lastResultData = data;
    placeholder.hidden = true;
    resultBox.hidden = false;
    const preds = data.predictions;
    const top = preds[0];
    const isOod = data.ood && data.ood.is_ood;

    // Банер OOD
    if (isOod) {
      oodBanner.hidden = false;
      oodBanner.className = 'ood-banner';
      const fb = data.ood.fallback;
      const fbLine = fb
        ? `<div class="ood-fallback">🧭 ${T('dyn.ood.second')}
             <b>${fb.city_ua}</b> (${(fb.prob * 100).toFixed(0)}%).</div>`
        : '';
      oodBanner.innerHTML =
        `<span>🌍</span><div><b>${T('dyn.ood.title')}</b>
         ${T('dyn.ood.body')}
         <div class="ood-metrics">${T('dyn.ood.metric')} <b>${fmtSim(data.ood.score)}</b>
           ${T('dyn.ood.thr')} <b>${fmtSim(data.ood.threshold)}</b> · Mahalanobis</div>
         ${fbLine}
         <div class="muted" style="margin-top:6px">${T('dyn.ood.note')}</div></div>`;
    } else {
      oodBanner.hidden = true;
    }

    // Вердикт
    if (isOod) {
      verdict.innerHTML = `${T('dyn.lowconf')}
        <span class="city">${top.city_ua}</span>`;
    } else {
      verdict.innerHTML = `${T('dyn.likely')}
        <span class="city">${top.city_ua}</span>
        <span class="conf">${(top.prob * 100).toFixed(1)}% ${T('dyn.conf')}</span>`;
    }

    // Стовпчики ймовірностей
    bars.innerHTML = '';
    preds.forEach(p => {
      const color = CITY_COLORS[p.city.toLowerCase()] || '#4f8cff';
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML = `
        <div class="bar-city">${p.city_ua}<small>${p.country}</small></div>
        <div class="bar-track"><div class="bar-fill" style="background:${color}"></div></div>
        <div class="bar-pct">${(p.prob * 100).toFixed(1)}%</div>`;
      bars.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector('.bar-fill').style.width = (p.prob * 100).toFixed(1) + '%';
      });
    });

    updateMap(preds, isOod);

    const meta = modelMetaMap[data.model] || {};
    let line = `Модель: ${meta.label || data.model}`;
    if (data.ood && data.ood.enabled) {
      line += ` · OOD-гейт активний — Mahalanobis (поріг ${fmtSim(data.ood.threshold)})`;
    } else {
      line += ` · OOD-гейт вимкнено (лише softmax)`;
    }
    modelMeta.textContent = line;

    // GeoScore обраної моделі (тестова метрика) як шкала 0–5000.
    const gsBox = document.getElementById('geoscoreBox');
    const bm = benchData && benchData.models && benchData.models.find(x => x.id === data.model);
    if (gsBox && bm && bm.geoscore != null) {
      gsBox.hidden = false;
      document.getElementById('gsVal').textContent = `${Math.round(bm.geoscore)} / 5000`;
      document.getElementById('gsFill').style.width = (bm.geoscore / 5000 * 100).toFixed(1) + '%';
    } else if (gsBox) { gsBox.hidden = true; }

    // #4 Карта уваги доступна лише для ViT (CLIP) моделей.
    const isClip = data.model === 'streetclip' || data.model === 'geoclip';
    attnWrap.hidden = !isClip;
    attnResult.hidden = true;
    attnImg.removeAttribute('src');
  }

  // ── Карта уваги ViT (#4) ───────────────────────────────────────────────
  attnBtn.addEventListener('click', () => {
    if (!lastPredict) return;
    const sp = attnBtn.querySelector('.spinner'), tx = attnBtn.querySelector('.btn-text');
    sp.hidden = false; tx.textContent = T('dyn.attn.run'); attnBtn.disabled = true;
    fetch('/api/explain', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(lastPredict),
    })
      .then(r => r.json())
      .then(d => {
        attnResult.hidden = false;
        const cap = attnResult.querySelector('p');
        if (d.available && d.heatmap) {
          attnImg.src = d.heatmap; attnImg.style.display = '';
          if (cap) cap.textContent = 'Яскравіші зони — куди ViT приділяв більше уваги (attention-rollout по всіх шарах).';
        } else {
          attnImg.style.display = 'none';
          if (cap) cap.textContent = d.reason || 'Карта уваги недоступна.';
        }
      })
      .catch(e => { attnResult.hidden = false; const cap = attnResult.querySelector('p');
        if (cap) cap.textContent = 'Помилка: ' + e.message; })
      .finally(() => { sp.hidden = true; tx.textContent = T('btn.attn'); attnBtn.disabled = false; });
  });

  // Перемальовування динамічних результатів при зміні мови.
  if (window.i18n) window.i18n.onChange(() => {
    if (lastResultData) renderResult(lastResultData);
    if (lastCompareData) renderCompare(lastCompareData);
  });

  function fmtSim(v) { return v == null ? '—' : v.toFixed(Math.abs(v) >= 10 ? 1 : 3); }

  // ── Інтерактивна карта (Leaflet + OSM) ─────────────────────────────────
  function ensureMap() {
    if (leafletMap || typeof L === 'undefined') return leafletMap;
    leafletMap = L.map(mapEl, { zoomControl: true }).setView([50.2, 22], 4);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18, attribution: '© OpenStreetMap',
    }).addTo(leafletMap);
    return leafletMap;
  }

  function updateMap(preds, isOod) {
    const map = ensureMap();
    if (!map) return;                       // Leaflet ще не завантажився
    setTimeout(() => map.invalidateSize(), 0);
    if (mapLayer) { map.removeLayer(mapLayer); mapLayer = null; }
    const group = L.layerGroup().addTo(map);
    mapLayer = group;

    const top = preds[0];
    const color = CITY_COLORS[top.city.toLowerCase()] || '#4f8cff';

    // Слабкі маркери всіх міст для контексту.
    preds.forEach(p => {
      const c = CITY_COORDS[p.city.toLowerCase()];
      if (!c) return;
      L.circleMarker([c.lat, c.lon], {
        radius: 4, color: '#8a96ad', weight: 1, fillOpacity: .4,
      }).addTo(group).bindTooltip(`${p.city_ua}: ${(p.prob * 100).toFixed(1)}%`);
    });

    const tc = CITY_COORDS[top.city.toLowerCase()];
    if (tc) {
      // Радіус невпевненості: менша впевненість → більше коло.
      const radiusM = (30 + (1 - top.prob) * 320) * 1000;
      L.circle([tc.lat, tc.lon], {
        radius: radiusM, color, weight: isOod ? 1 : 2,
        opacity: isOod ? .4 : .8, fillColor: color,
        fillOpacity: isOod ? .05 : .15, dashArray: isOod ? '6 6' : null,
      }).addTo(group);
      const marker = L.circleMarker([tc.lat, tc.lon], {
        radius: isOod ? 7 : 10, color: '#0e1626', weight: 2,
        fillColor: color, fillOpacity: 1,
      }).addTo(group);
      marker.bindPopup(isOod
        ? `<b>${top.city_ua}?</b><br>низька довіра — можливо, не це місто`
        : `<b>${top.city_ua}</b><br>${(top.prob * 100).toFixed(1)}% впевненості`
      ).openPopup();
      map.setView([tc.lat, tc.lon], isOod ? 4 : 6, { animate: true });
    }
  }

  // ── Бенчмарк (#10) ──────────────────────────────────────────────────────
  const benchTable = document.getElementById('benchTable');
  const cmTabs = document.getElementById('cmTabs');
  const cmHost = document.getElementById('cmHost');
  const tsneCanvas = document.getElementById('tsneCanvas');
  const tsneLegend = document.getElementById('tsneLegend');
  const relHost = document.getElementById('relHost');
  let benchData = null;
  const CITY_UA = { kyiv: 'Київ', warsaw: 'Варшава', prague: 'Прага', budapest: 'Будапешт' };
  const pct = x => (x * 100).toFixed(1) + '%';
  const cityUa = c => CITY_UA[c] || c;

  fetch('/api/benchmark').then(r => r.json()).then(renderBench)
    .catch(() => { benchTable.innerHTML = '<p class="muted">Бенчмарк недоступний.</p>'; });

  function renderBench(data) {
    if (!data || !data.models) { benchTable.innerHTML = '<p class="muted">Немає даних.</p>'; return; }
    benchData = data;
    const cityHdr = data.models[0].per_class.map(pc => `<th class="small">${cityUa(pc.city)} F1</th>`).join('');
    const rows = data.models.map(m => `
      <tr>
        <td><b>${m.label}</b></td>
        <td class="num hl">${pct(m.top1)}</td>
        <td class="num">${pct(m.macro_f1)}</td>
        <td class="num">${pct(m.balanced_acc)}</td>
        ${m.per_class.map(pc => `<td class="num small">${pct(pc.f1)}</td>`).join('')}
      </tr>`).join('');
    benchTable.innerHTML = `
      <table class="bench-table">
        <thead><tr><th>Модель</th><th>Top-1</th><th>macro-F1</th><th>bal.acc</th>${cityHdr}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="muted small">Тест: ${data.test_size} фото · 4 міста · праворуч — F1 по кожному місту.</p>`;
    cmTabs.innerHTML = data.models.map((m, i) =>
      `<button class="cm-tab${i === 0 ? ' active' : ''}" data-i="${i}">${m.label}</button>`).join('');
    cmTabs.querySelectorAll('.cm-tab').forEach(b => b.addEventListener('click', () => {
      cmTabs.querySelectorAll('.cm-tab').forEach(x => x.classList.remove('active'));
      b.classList.add('active'); selectBenchModel(+b.dataset.i);
    }));
    selectBenchModel(0);
  }

  function selectBenchModel(idx) {
    drawCM(idx); drawTSNE(idx); drawReliability(idx);
  }

  // t-SNE scatter на canvas
  function drawTSNE(idx) {
    if (!tsneCanvas) return;
    const m = benchData.models[idx];
    const ctx = tsneCanvas.getContext('2d');
    const W = tsneCanvas.width, H = tsneCanvas.height, pad = 24;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0e1626'; ctx.fillRect(0, 0, W, H);
    const sx = v => pad + (v + 1) / 2 * (W - 2 * pad);
    const sy = v => pad + (1 - (v + 1) / 2) * (H - 2 * pad);
    (m.tsne || []).forEach(p => {
      ctx.beginPath();
      ctx.fillStyle = CITY_COLORS[p.city] || '#4f8cff';
      ctx.globalAlpha = 0.72;
      ctx.arc(sx(p.x), sy(p.y), 3, 0, 6.2832);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
    tsneLegend.innerHTML = benchData.class_names.map(c =>
      `<span class="lg"><span class="ex-dot" style="background:${CITY_COLORS[c] || '#4f8cff'}"></span>${cityUa(c)}</span>`).join('');
  }

  // Reliability diagram (SVG)
  function drawReliability(idx) {
    if (!relHost) return;
    const rel = benchData.models[idx].reliability;
    if (!rel) { relHost.innerHTML = '<p class="muted small">немає даних</p>'; return; }
    const W = 460, H = 300, pad = 38, n = rel.bins.length;
    const bw = (W - 2 * pad) / n;
    const X = v => pad + v * (W - 2 * pad);
    const Y = v => H - pad - v * (H - 2 * pad);
    let s = `<svg viewBox="0 0 ${W} ${H}" class="rel-svg">`;
    s += `<rect x="0" y="0" width="${W}" height="${H}" fill="#0e1626" rx="10"/>`;
    for (let g = 0; g <= 1.0001; g += 0.25) {
      s += `<line x1="${pad}" y1="${Y(g)}" x2="${W - pad}" y2="${Y(g)}" stroke="#1b2438"/>`;
      s += `<text x="${pad - 6}" y="${Y(g) + 4}" fill="#8a96ad" font-size="10" text-anchor="end">${(g * 100).toFixed(0)}</text>`;
    }
    s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="#8a96ad" stroke-dasharray="5 5"/>`;
    rel.bins.forEach((b, i) => {
      if (b.acc == null) return;
      const x = pad + i * bw + bw * 0.15, w = bw * 0.7;
      const gap = Math.abs(b.acc - b.conf);
      const col = gap > 0.12 ? '#ff5d6c' : '#21d4a8';
      s += `<rect x="${x}" y="${Y(b.acc)}" width="${w}" height="${Y(0) - Y(b.acc)}" fill="${col}" opacity="0.8">
              <title>впевненість ${(b.conf * 100).toFixed(0)}% → точність ${(b.acc * 100).toFixed(0)}% (n=${b.count})</title>
            </rect>`;
    });
    s += `<text x="${W - pad}" y="${pad}" fill="#e8edf6" font-size="13" text-anchor="end" font-weight="700">ECE = ${(rel.ece * 100).toFixed(1)}%</text>`;
    s += `<text x="${W / 2}" y="${H - 8}" fill="#8a96ad" font-size="11" text-anchor="middle">впевненість моделі →</text>`;
    s += `</svg>`;
    relHost.innerHTML = s;
  }

  function drawCM(idx) {
    const m = benchData.models[idx], cn = benchData.class_names, cm = m.confusion;
    let h = `<table class="cm-table"><thead><tr><th class="cm-corner">істина ↓ / передбачено →</th>`;
    cn.forEach(c => h += `<th>${cityUa(c)}</th>`);
    h += '</tr></thead><tbody>';
    cm.forEach((row, i) => {
      const tot = row.reduce((a, b) => a + b, 0) || 1;
      h += `<tr><th>${cityUa(cn[i])}</th>`;
      row.forEach((v, j) => {
        const frac = v / tot, diag = i === j;
        const bg = diag ? `rgba(33,212,168,${0.12 + 0.7 * frac})` : `rgba(255,93,108,${0.06 + 0.7 * frac})`;
        h += `<td class="cm-cell" style="background:${bg}" title="${cityUa(cn[i])}→${cityUa(cn[j])}: ${pct(frac)}">${v}</td>`;
      });
      h += '</tr>';
    });
    h += '</tbody></table>';
    h += '<table class="cm-table tpr"><thead><tr><th>Місто</th><th>TPR</th><th>FPR</th><th>F1</th></tr></thead><tbody>';
    m.per_class.forEach(pc =>
      h += `<tr><th>${cityUa(pc.city)}</th><td class="num">${pct(pc.tpr)}</td><td class="num">${pct(pc.fpr)}</td><td class="num">${pct(pc.f1)}</td></tr>`);
    h += '</tbody></table>';
    cmHost.innerHTML = h;
  }
})();
