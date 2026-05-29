/* ============================================================================
   app.js — Завантаження фото, вибір моделі, виклик API та показ результатів.
   ============================================================================ */

(function () {
  const CITY_COLORS = { warsaw: '#ff6b6b', prague: '#ffd166', budapest: '#4ecdc4' };

  const fileInput   = document.getElementById('fileInput');
  const dropzone    = document.getElementById('dropzone');
  const dropPrompt  = document.getElementById('dropPrompt');
  const preview     = document.getElementById('preview');
  const modelList   = document.getElementById('modelList');
  const predictBtn  = document.getElementById('predictBtn');
  const btnText     = predictBtn.querySelector('.btn-text');
  const spinner     = predictBtn.querySelector('.spinner');

  const placeholder = document.getElementById('resultPlaceholder');
  const resultBox   = document.getElementById('resultBox');
  const oodBanner   = document.getElementById('oodBanner');
  const verdict     = document.getElementById('verdict');
  const bars        = document.getElementById('bars');
  const miniMap     = document.getElementById('miniMap');
  const modelMeta   = document.getElementById('modelMeta');

  let imageDataUrl = null;
  let selectedModel = null;
  let modelMetaMap = {};

  // ── Завантаження списку моделей ────────────────────────────────────────
  fetch('/api/models')
    .then(r => r.json())
    .then(data => renderModels(data.models || []))
    .catch(() => {
      modelList.innerHTML = '<p class="muted">Не вдалося завантажити список моделей.</p>';
    });

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
  }

  // ── Передбачення ───────────────────────────────────────────────────────
  predictBtn.addEventListener('click', () => {
    if (!imageDataUrl || !selectedModel) return;
    setLoading(true);
    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selectedModel, image: imageDataUrl }),
    })
      .then(async r => {
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || 'Помилка сервера');
        return data;
      })
      .then(renderResult)
      .catch(err => showError(err.message))
      .finally(() => setLoading(false));
  });

  function setLoading(on) {
    predictBtn.disabled = on || !(imageDataUrl && selectedModel);
    spinner.hidden = !on;
    btnText.textContent = on ? 'Обробка…' : 'Визначити місто';
  }

  function showError(msg) {
    placeholder.hidden = true;
    resultBox.hidden = false;
    oodBanner.hidden = false;
    oodBanner.className = 'ood-banner danger';
    oodBanner.innerHTML = `<span>⚠</span><div><b>Помилка</b>${msg}</div>`;
    verdict.innerHTML = ''; bars.innerHTML = ''; miniMap.innerHTML = ''; modelMeta.textContent = '';
  }

  // ── Відображення результату ────────────────────────────────────────────
  function renderResult(data) {
    placeholder.hidden = true;
    resultBox.hidden = false;
    const preds = data.predictions;
    const top = preds[0];
    const isOod = data.ood && data.ood.is_ood;

    // Банер OOD
    if (isOod) {
      oodBanner.hidden = false;
      oodBanner.className = 'ood-banner';
      oodBanner.innerHTML =
        `<span>🌍</span><div><b>Фото, ймовірно, НЕ з цих трьох міст</b>
         Знімок не схожий на Варшаву, Прагу чи Будапешт
         (подібність ${fmtSim(data.ood.max_similarity)} нижча за поріг ${fmtSim(data.ood.threshold)}).
         Нижче — найближчі здогадки моделі, але довіряти їм не варто.</div>`;
    } else {
      oodBanner.hidden = true;
    }

    // Вердикт
    if (isOod) {
      verdict.innerHTML = `Найімовірніший збіг (низька довіра):
        <span class="city">${top.city_ua}</span>`;
    } else {
      verdict.innerHTML = `Це, найімовірніше:
        <span class="city">${top.city_ua}</span>
        <span class="conf">${(top.prob * 100).toFixed(1)}% впевненості</span>`;
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

    drawMiniMap(preds, isOod);

    const meta = modelMetaMap[data.model] || {};
    let line = `Модель: ${meta.label || data.model}`;
    if (data.ood && data.ood.enabled) {
      line += ` · OOD-гейт активний (поріг подібності ${fmtSim(data.ood.threshold)})`;
    } else {
      line += ` · OOD-гейт вимкнено (лише softmax)`;
    }
    modelMeta.textContent = line;
  }

  function fmtSim(v) { return v == null ? '—' : v.toFixed(3); }

  // ── Міні-карта Центральної Європи ──────────────────────────────────────
  function project(lat, lon) {
    const x = 40 + ((lon - 12) / 11) * 240;
    const y = 30 + ((54 - lat) / 8) * 170;
    return [x, y];
  }

  function drawMiniMap(preds, isOod) {
    const topCity = preds[0].city.toLowerCase();
    let svg = `<rect x="0" y="0" width="320" height="240" rx="10" fill="#0e1626"/>`;
    // ледь помітна сітка
    for (let gx = 40; gx <= 280; gx += 60)
      svg += `<line x1="${gx}" y1="20" x2="${gx}" y2="220" stroke="#1b2438" stroke-width="1"/>`;
    for (let gy = 30; gy <= 210; gy += 45)
      svg += `<line x1="30" y1="${gy}" x2="290" y2="${gy}" stroke="#1b2438" stroke-width="1"/>`;

    const cities = {
      warsaw:   { lat: 52.2297, lon: 21.0122, ua: 'Варшава' },
      prague:   { lat: 50.0755, lon: 14.4378, ua: 'Прага' },
      budapest: { lat: 47.4979, lon: 19.0402, ua: 'Будапешт' },
    };
    for (const key in cities) {
      const c = cities[key];
      const [x, y] = project(c.lat, c.lon);
      const isTop = key === topCity && !isOod;
      const color = CITY_COLORS[key];
      if (isTop) {
        svg += `<circle cx="${x}" cy="${y}" r="18" fill="${color}" opacity="0.18">
                  <animate attributeName="r" values="14;22;14" dur="2s" repeatCount="indefinite"/>
                </circle>`;
      }
      svg += `<circle cx="${x}" cy="${y}" r="${isTop ? 8 : 5}" fill="${color}"
                stroke="#0e1626" stroke-width="2"/>`;
      svg += `<text x="${x}" y="${y - 12}" fill="${isTop ? '#fff' : '#8a96ad'}"
                font-size="${isTop ? 13 : 11}" font-weight="${isTop ? 700 : 500}"
                text-anchor="middle">${c.ua}</text>`;
    }
    if (isOod) {
      svg += `<text x="160" y="228" fill="#ffb454" font-size="11" text-anchor="middle"
                font-weight="600">фото поза межами цих міст</text>`;
    }
    miniMap.innerHTML = svg;
  }
})();
