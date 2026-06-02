/* ============================================================================
   slides.js — Вбудована двомовна «презентація» для захисту.
   Кожен слайд має поля {ua,en}; мова береться з window.i18n.lang.
   ============================================================================ */
(function () {
  const SLIDES = [
    {
      icon: '🌍',
      kicker: { ua: 'Дипломна робота', en: 'Diploma project' },
      title: { ua: 'Геолокація вуличних фотографій', en: 'Street-photo geolocation' },
      html: {
        ua: `<p class="s-lead">Нейромережа визначає <b>місто</b> за одним знімком вулиці.</p>
          <div class="s-chips">
            <span class="s-chip" style="--c:#a78bfa">Київ</span>
            <span class="s-chip" style="--c:#ff6b6b">Варшава</span>
            <span class="s-chip" style="--c:#ffd166">Прага</span>
            <span class="s-chip" style="--c:#4ecdc4">Будапешт</span>
          </div>
          <p class="s-foot">Три архітектури: <b>StreetCLIP</b> · <b>EfficientNet-B2</b> · <b>GeoCLIP</b></p>`,
        en: `<p class="s-lead">A neural network identifies the <b>city</b> from a single street photo.</p>
          <div class="s-chips">
            <span class="s-chip" style="--c:#a78bfa">Kyiv</span>
            <span class="s-chip" style="--c:#ff6b6b">Warsaw</span>
            <span class="s-chip" style="--c:#ffd166">Prague</span>
            <span class="s-chip" style="--c:#4ecdc4">Budapest</span>
          </div>
          <p class="s-foot">Three architectures: <b>StreetCLIP</b> · <b>EfficientNet-B2</b> · <b>GeoCLIP</b></p>`,
      },
    },
    {
      icon: '🎯',
      kicker: { ua: 'Задача', en: 'The task' },
      title: { ua: 'Визначити місто за фото', en: 'Identify the city from a photo' },
      html: {
        ua: `<ul class="s-list">
            <li><b>Вхід:</b> одна фотографія вулиці.</li>
            <li><b>Вихід:</b> одне з 4 міст + рівень упевненості.</li>
            <li><b>Виклик:</b> центральноєвропейські столиці візуально схожі — спільна
                архітектурна спадщина, тому задача нетривіальна.</li>
          </ul>
          <p class="s-foot">Аналог гри GeoGuessr, але автоматичний і обмежений 4 містами.</p>`,
        en: `<ul class="s-list">
            <li><b>Input:</b> a single street photo.</li>
            <li><b>Output:</b> one of 4 cities + a confidence level.</li>
            <li><b>Challenge:</b> Central-European capitals look alike — shared architectural
                heritage makes the task non-trivial.</li>
          </ul>
          <p class="s-foot">Like GeoGuessr, but automatic and limited to 4 cities.</p>`,
      },
    },
    {
      icon: '🗂️',
      kicker: { ua: 'Дані', en: 'Data' },
      title: { ua: '25 039 фото Google Street View', en: '25,039 Google Street View photos' },
      html: {
        ua: `<div class="s-stats">
            <div class="s-stat"><b>4</b><span>міста</span></div>
            <div class="s-stat"><b>4</b><span>напрямки / точку</span></div>
            <div class="s-stat"><b>17 552</b><span>train</span></div>
            <div class="s-stat"><b>3 771</b><span>val</span></div>
            <div class="s-stat"><b>3 716</b><span>test</span></div>
          </div>
          <p class="s-foot">🔒 <b>Leak-free</b> розбиття за H3-комірками: геоблок цілком в одному
             спліті → однакові панорами не «перетікають» між train і test. Без цього метрики
             завищені й невалідні для диплома.</p>`,
        en: `<div class="s-stats">
            <div class="s-stat"><b>4</b><span>cities</span></div>
            <div class="s-stat"><b>4</b><span>headings / point</span></div>
            <div class="s-stat"><b>17,552</b><span>train</span></div>
            <div class="s-stat"><b>3,771</b><span>val</span></div>
            <div class="s-stat"><b>3,716</b><span>test</span></div>
          </div>
          <p class="s-foot">🔒 <b>Leak-free</b> split by H3 cells: a whole geo-block stays in one
             split → identical panoramas don't leak between train and test. Without this, metrics
             are inflated and invalid for a thesis.</p>`,
      },
    },
    {
      icon: '🧠',
      kicker: { ua: 'Підхід', en: 'Approach' },
      title: { ua: 'Три парадигми — одне порівняння', en: 'Three paradigms — one comparison' },
      html: {
        ua: `<div class="s-cards">
            <div class="s-card"><h4>Baseline CNN</h4>
              <p>EfficientNet-B2, навчений «з нуля». Класична згорткова мережа — точка відліку.</p></div>
            <div class="s-card hl"><h4>StreetCLIP</h4>
              <p>CLIP ViT-L/14, дообучений саме на вуличних фото. Лінійний пробник над замороженим
                 енкодером. <b>Лідер.</b></p></div>
            <div class="s-card"><h4>GeoCLIP</h4>
              <p>CLIP ViT + GPS-енкодер (Random Fourier). Контрастивне навчання (InfoNCE) зображення↔координати.</p></div>
          </div>`,
        en: `<div class="s-cards">
            <div class="s-card"><h4>Baseline CNN</h4>
              <p>EfficientNet-B2 trained from scratch. A classic convolutional net — the reference point.</p></div>
            <div class="s-card hl"><h4>StreetCLIP</h4>
              <p>CLIP ViT-L/14 fine-tuned on street photos. A linear probe over a frozen encoder. <b>The leader.</b></p></div>
            <div class="s-card"><h4>GeoCLIP</h4>
              <p>CLIP ViT + a GPS encoder (Random Fourier). Contrastive learning (InfoNCE) image↔coordinates.</p></div>
          </div>`,
      },
    },
    {
      icon: '⚙️',
      kicker: { ua: 'Як це працює', en: 'How it works' },
      title: { ua: 'Від пікселів до міста', en: 'From pixels to a city' },
      html: {
        ua: `<div class="s-flow">
            <span class="s-node">📷 Фото</span><i>→</i>
            <span class="s-node">Препроцесинг<small>224–336px</small></span><i>→</i>
            <span class="s-node">Backbone<small>CNN / ViT</small></span><i>→</i>
            <span class="s-node">Embedding<small>512–768D</small></span><i>→</i>
            <span class="s-node">Класифікатор</span><i>→</i>
            <span class="s-node hl">Місто + %</span>
          </div>
          <p class="s-foot">Backbone стискає зображення у вектор-ознаку (embedding); голова-класифікатор
             перетворює його на ймовірності 4 міст через softmax.</p>`,
        en: `<div class="s-flow">
            <span class="s-node">📷 Photo</span><i>→</i>
            <span class="s-node">Preprocess<small>224–336px</small></span><i>→</i>
            <span class="s-node">Backbone<small>CNN / ViT</small></span><i>→</i>
            <span class="s-node">Embedding<small>512–768D</small></span><i>→</i>
            <span class="s-node">Classifier</span><i>→</i>
            <span class="s-node hl">City + %</span>
          </div>
          <p class="s-foot">The backbone compresses the image into a feature vector (embedding);
             the classifier head turns it into probabilities over 4 cities via softmax.</p>`,
      },
    },
    {
      icon: '🏋️',
      kicker: { ua: 'Навчання', en: 'Training' },
      title: { ua: 'Двоетапне навчання', en: 'Two-stage training' },
      html: {
        ua: `<div class="s-cards">
            <div class="s-card"><h4>Стадія 1 — заморожений backbone</h4>
              <p>Вчимо лише голову-класифікатор. Швидко й стабільно: backbone уже «бачив» світ.</p></div>
            <div class="s-card"><h4>Стадія 2 — розморожування</h4>
              <p>Розморожуємо останні шари й тонко донавчаємо під наші 4 міста — приріст точності.</p></div>
          </div>
          <p class="s-foot">AdamW + CosineAnnealing · mixed precision (bf16) · early stopping ·
             локально на <b>RTX 3090</b>.</p>`,
        en: `<div class="s-cards">
            <div class="s-card"><h4>Stage 1 — frozen backbone</h4>
              <p>Train only the classifier head. Fast and stable: the backbone already "knows" the world.</p></div>
            <div class="s-card"><h4>Stage 2 — unfreezing</h4>
              <p>Unfreeze the last layers and fine-tune to our 4 cities — an accuracy gain.</p></div>
          </div>
          <p class="s-foot">AdamW + CosineAnnealing · mixed precision (bf16) · early stopping ·
             locally on an <b>RTX 3090</b>.</p>`,
      },
    },
    {
      icon: '🏆',
      kicker: { ua: 'Результати', en: 'Results' },
      title: { ua: 'StreetCLIP — лідер', en: 'StreetCLIP — the leader' },
      html: { ua: resultsFallback('ua'), en: resultsFallback('en') },
    },
    {
      icon: '🛡️',
      kicker: { ua: 'Надійність', en: 'Reliability' },
      title: { ua: 'А якщо фото не з цих міст?', en: "What if the photo isn't from these cities?" },
      html: {
        ua: `<p class="s-lead">OOD-гейт ловить «чужі» знімки до того, як модель видасть хибну відповідь.</p>
          <ul class="s-list">
            <li><b>Метод:</b> Mahalanobis-відстань в embedding-просторі до найближчого міста.</li>
            <li><b>Якість:</b> AUROC <b>0.906</b> (проти 0.845 у простого косинус-гейта).</li>
            <li><b>Поведінка:</b> нижче порога → банер «ймовірно, не з цих 4 міст» + друга думка GeoCLIP.</li>
          </ul>`,
        en: `<p class="s-lead">The OOD gate catches "foreign" shots before the model gives a wrong answer.</p>
          <ul class="s-list">
            <li><b>Method:</b> Mahalanobis distance in embedding space to the nearest city.</li>
            <li><b>Quality:</b> AUROC <b>0.906</b> (vs 0.845 for a simple cosine gate).</li>
            <li><b>Behavior:</b> below threshold → "likely not from these 4 cities" banner + a GeoCLIP second opinion.</li>
          </ul>`,
      },
    },
    {
      icon: '🔍',
      kicker: { ua: 'Пояснюваність (XAI)', en: 'Explainability (XAI)' },
      title: { ua: 'Куди дивиться модель', en: 'Where the model looks' },
      html: {
        ua: `<p class="s-lead">Attention-rollout по всіх шарах ViT → теплова карта уваги поверх фото.</p>
          <ul class="s-list">
            <li>Видно, що рішення спирається на <b>фасади, дахи, вивіски</b>, а не на випадкові артефакти.</li>
            <li>Це і пояснюваність ШІ, і перевірка, що модель «вчиться правильному».</li>
          </ul>
          <p class="s-foot">Доступно для ViT-моделей (StreetCLIP, GeoCLIP) прямо у демо — кнопка
             «Показати, на що дивилась модель».</p>`,
        en: `<p class="s-lead">Attention-rollout across all ViT layers → a heatmap over the photo.</p>
          <ul class="s-list">
            <li>Shows the decision relies on <b>facades, roofs, signage</b> — not random artifacts.</li>
            <li>Both AI explainability and a check that the model "learns the right thing".</li>
          </ul>
          <p class="s-foot">Available for ViT models (StreetCLIP, GeoCLIP) right in the demo —
             the "Show where the model looked" button.</p>`,
      },
    },
    {
      icon: '✅',
      kicker: { ua: 'Висновки', en: 'Conclusions' },
      title: { ua: 'Підсумки', en: 'Takeaways' },
      html: {
        ua: `<ul class="s-list big">
            <li>Дообучений на вуличних фото <b>CLIP (StreetCLIP)</b> — найкращий для геолокації.</li>
            <li>Класифікація <b>рівня міста</b> вирішена надійно (89% Top-1, Top-3 ≥ 96%).</li>
            <li><b>Leak-free</b> методологія → чесні, захищувані результати.</li>
            <li>Робоче <b>веб-демо</b>: карта, OOD-детекція, пояснюваність, порівняння моделей.</li>
          </ul>`,
        en: `<ul class="s-list big">
            <li>A street-fine-tuned <b>CLIP (StreetCLIP)</b> is best for geolocation.</li>
            <li><b>City-level</b> classification is solved reliably (89% Top-1, Top-3 ≥ 96%).</li>
            <li><b>Leak-free</b> methodology → honest, defensible results.</li>
            <li>A working <b>web demo</b>: map, OOD detection, explainability, model comparison.</li>
          </ul>`,
      },
    },
  ];

  function resultsTable(rows, lang) {
    const h = lang === 'en'
      ? ['Model', 'Top-1', 'macro-F1', 'GeoScore']
      : ['Модель', 'Top-1', 'macro-F1', 'GeoScore'];
    const foot = lang === 'en'
      ? 'Top-3 ≥ 96% for all models — the correct city is almost always among the top three. Kyiv is recognized best.'
      : 'Top-3 ≥ 96% у всіх моделей — правильне місто майже завжди серед трьох найімовірніших. Київ розпізнається найкраще.';
    return `<table class="s-table"><thead><tr>${h.map(x => `<th>${x}</th>`).join('')}</tr></thead>
      <tbody>${rows}</tbody></table><p class="s-foot">${foot}</p>`;
  }
  function resultsFallback(lang) {
    const r = [['StreetCLIP', '89.4%', '87.9%', '4793', true],
               ['GeoCLIP', '83.8%', '81.3%', '4698', false],
               ['Baseline CNN', '71.5%', '67.8%', '4482', false]]
      .map(x => `<tr class="${x[4] ? 'hl' : ''}"><td>${x[0]}</td><td>${x[1]}</td><td>${x[2]}</td><td>${x[3]}</td></tr>`).join('');
    return resultsTable(r, lang);
  }
  function resultsLive(models, lang) {
    const order = ['streetclip', 'geoclip', 'baseline'];
    const byId = Object.fromEntries(models.map(m => [m.id, m]));
    const rows = order.map((id, i) => {
      const m = byId[id]; if (!m) return '';
      return `<tr class="${i === 0 ? 'hl' : ''}"><td>${m.label}</td>` +
        `<td>${(m.top1 * 100).toFixed(1)}%</td><td>${(m.macro_f1 * 100).toFixed(1)}%</td>` +
        `<td>${m.geoscore != null ? Math.round(m.geoscore) : '—'}</td></tr>`;
    }).join('');
    return resultsTable(rows, lang);
  }

  const stage   = document.getElementById('deckStage');
  const dotsEl  = document.getElementById('deckDots');
  const counter = document.getElementById('deckCounter');
  const deck    = document.getElementById('deck');
  if (!stage) return;

  let idx = 0;
  const L = () => (window.i18n ? window.i18n.lang : 'ua');
  const pick = o => (o && typeof o === 'object' && ('ua' in o)) ? (o[L()] || o.ua) : o;

  function render() {
    const s = SLIDES[idx];
    stage.classList.remove('fade');
    void stage.offsetWidth;
    stage.classList.add('fade');
    stage.innerHTML = `
      <div class="s-head">
        <span class="s-icon">${s.icon}</span>
        <div><div class="s-kicker">${pick(s.kicker)}</div><h3 class="s-title">${pick(s.title)}</h3></div>
      </div>
      <div class="s-body">${pick(s.html)}</div>`;
    counter.textContent = `${idx + 1} / ${SLIDES.length}`;
    dotsEl.querySelectorAll('.deck-dot').forEach((d, i) => d.classList.toggle('active', i === idx));
  }

  function go(n) { idx = (n + SLIDES.length) % SLIDES.length; render(); }

  dotsEl.innerHTML = SLIDES.map((_, i) =>
    `<button class="deck-dot" data-i="${i}" aria-label="Slide ${i + 1}"></button>`).join('');
  dotsEl.querySelectorAll('.deck-dot').forEach(d =>
    d.addEventListener('click', () => go(+d.dataset.i)));

  document.getElementById('deckPrev').addEventListener('click', () => go(idx - 1));
  document.getElementById('deckNext').addEventListener('click', () => go(idx + 1));
  document.getElementById('deckFs').addEventListener('click', () => {
    if (!document.fullscreenElement) deck.requestFullscreen?.();
    else document.exitFullscreen?.();
  });

  document.addEventListener('keydown', e => {
    const r = deck.getBoundingClientRect();
    const visible = document.fullscreenElement === deck ||
      (r.top < window.innerHeight * 0.6 && r.bottom > window.innerHeight * 0.4);
    if (!visible) return;
    if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); go(idx + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); go(idx - 1); }
  });

  let tx = null;
  stage.addEventListener('touchstart', e => { tx = e.touches[0].clientX; }, { passive: true });
  stage.addEventListener('touchend', e => {
    if (tx == null) return;
    const dx = e.changedTouches[0].clientX - tx;
    if (Math.abs(dx) > 50) go(idx + (dx < 0 ? 1 : -1));
    tx = null;
  });

  // Перемальовування при зміні мови.
  if (window.i18n) window.i18n.onChange(render);

  // Live-числа з бенчмарку у слайді «Результати».
  fetch('/api/benchmark').then(r => r.json()).then(d => {
    if (!d || !d.models) return;
    const ri = SLIDES.findIndex(s => s.kicker && s.kicker.ua === 'Результати');
    if (ri >= 0) {
      SLIDES[ri].html = { ua: resultsLive(d.models, 'ua'), en: resultsLive(d.models, 'en') };
      render();
    }
  }).catch(() => { /* лишаємо фолбек */ });

  render();
})();
