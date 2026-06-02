/* ============================================================================
   i18n.js — Перемикач мови UA/EN. Статичні рядки розмічені [data-i18n] (текст)
   або [data-i18n-html] (innerHTML). Динамічні рядки беруться через i18n.t(key).
   Вибір зберігається у localStorage. Підключати ПЕРЕД app.js/slides.js.
   (Тексти 3D-анатомії лишаються українською — поза скоупом.)
   ============================================================================ */
(function () {
  const DICT = {
    'nav.recognize':  { ua: 'Розпізнавання', en: 'Recognition' },
    'nav.benchmark':  { ua: 'Бенчмарк', en: 'Benchmark' },
    'nav.slides':     { ua: 'Як це працює', en: 'How it works' },
    'nav.anatomy':    { ua: 'Анатомія моделей', en: 'Model anatomy' },

    'header.title':   { ua: 'Геолокація вуличних фото', en: 'Street-photo geolocation' },
    'header.tagline': { ua: 'Нейромережа визначає місто за фотографією вулиці',
                        en: 'A neural network identifies the city from a street photo' },

    'predict.h':   { ua: 'Завантажте фото вулиці', en: 'Upload a street photo' },
    'predict.sub': { ua: 'Модель спробує вгадати, до якого з чотирьох міст належить знімок: <b>Київ</b>, <b>Варшава</b>, <b>Прага</b> або <b>Будапешт</b>. Якщо фото не схоже на жодне з них — система повідомить про це.',
                     en: 'The model guesses which of the four cities the photo belongs to: <b>Kyiv</b>, <b>Warsaw</b>, <b>Prague</b> or <b>Budapest</b>. If the photo matches none — the system will say so.' },

    'examples.label': { ua: '⚡ Спробуйте на прикладі:', en: '⚡ Try an example:' },
    'examples.hint':  { ua: 'натисніть фото — воно одразу завантажиться', en: 'click a photo — it loads instantly' },

    'drop.title': { ua: 'Перетягніть фото сюди', en: 'Drag a photo here' },
    'drop.sub':   { ua: 'або натисніть, щоб обрати файл', en: 'or click to choose a file' },
    'picker.label': { ua: 'Оберіть модель:', en: 'Choose a model:' },

    'btn.predict': { ua: 'Визначити місто', en: 'Identify city' },
    'btn.compare': { ua: '⚖ Порівняти всі 3 моделі', en: '⚖ Compare all 3 models' },
    'btn.attn':    { ua: '🔍 Показати, на що дивилась модель', en: '🔍 Show where the model looked' },

    'result.placeholder': { ua: "Тут з'являться результати розпізнавання", en: 'Recognition results will appear here' },
    'howto.summary': { ua: '❔ Як читати результат', en: '❔ How to read the result' },
    'howto.geoscore': { ua: 'GeoScore моделі (на тесті)', en: 'Model GeoScore (on test set)' },

    'benchmark.h':   { ua: 'Бенчмарк моделей', en: 'Model benchmark' },
    'benchmark.sub': { ua: 'Метрики на <b>тестовому наборі</b> (без витоку): Top-1, macro-F1, збалансована точність, а також матриця плутанини й TPR/FPR по містах.',
                       en: 'Metrics on the <b>held-out test set</b> (leak-free): Top-1, macro-F1, balanced accuracy, plus a confusion matrix and per-city TPR/FPR.' },
    'cm.h':    { ua: 'Деталі по моделі', en: 'Per-model details' },
    'viz.tsne':{ ua: 'Простір ознак (t-SNE)', en: 'Feature space (t-SNE)' },
    'viz.rel': { ua: 'Калібрування впевненості', en: 'Confidence calibration' },

    'slides.h':   { ua: 'Як працює нейромережа', en: 'How the neural network works' },
    'slides.sub': { ua: 'Інтерактивні слайди — пояснення проєкту для захисту (замість презентації). Гортайте стрілками <b>← →</b>, клавіатурою або крапками знизу, кнопка <b>⛶</b> — на весь екран.',
                    en: 'Interactive slides explaining the project for the defense (instead of a slide deck). Navigate with <b>← →</b>, keyboard or the dots below; the <b>⛶</b> button goes fullscreen.' },

    'anatomy.h':   { ua: '3D-анатомія моделей', en: '3D model anatomy' },
    'anatomy.sub': { ua: 'Подивіться, як влаштована кожна нейромережа зсередини. <b>Обертайте</b> сцену мишею, <b>наближайте</b> колесом і <b>натискайте</b> на блоки, щоб прочитати, що вони роблять.',
                     en: 'See how each network is built inside. <b>Rotate</b> with the mouse, <b>zoom</b> with the wheel and <b>click</b> blocks to read what they do.' },
    'footer': { ua: 'Дипломна робота · класифікація вуличних фотографій за містом · StreetCLIP / EfficientNet-B2 / GeoCLIP',
                en: 'Diploma project · street-photo classification by city · StreetCLIP / EfficientNet-B2 / GeoCLIP' },

    // Динамічні (через i18n.t)
    'dyn.loading':   { ua: '⏳ Аналізуємо фото… Перший запуск кожної моделі може зайняти до хвилини (завантаження ваг і калібрування). Зачекайте, будь ласка.',
                       en: '⏳ Analyzing the photo… The first run of each model can take up to a minute (loading weights and calibration). Please wait.' },
    'dyn.likely':    { ua: 'Це, найімовірніше:', en: 'Most likely:' },
    'dyn.conf':      { ua: 'впевненості', en: 'confidence' },
    'dyn.lowconf':   { ua: 'Найімовірніший збіг (низька довіра):', en: 'Best match (low confidence):' },
    'dyn.ood.title': { ua: 'Фото, ймовірно, НЕ з цих чотирьох міст', en: 'The photo is likely NOT from these four cities' },
    'dyn.ood.body':  { ua: 'Знімок не схожий на Київ, Варшаву, Прагу чи Будапешт.', en: 'It does not resemble Kyiv, Warsaw, Prague or Budapest.' },
    'dyn.ood.metric':{ ua: 'показник належності', en: 'membership score' },
    'dyn.ood.thr':   { ua: 'нижчий за поріг', en: 'below threshold' },
    'dyn.ood.note':  { ua: 'Нижче — найближчі здогадки моделі, але довіряти їм не варто.', en: 'Below are the model\'s closest guesses, but they are unreliable.' },
    'dyn.ood.second':{ ua: 'Друга думка (GeoCLIP): найближче —', en: 'Second opinion (GeoCLIP): closest is' },
    'dyn.cmp.title': { ua: 'Порівняння моделей на цьому фото', en: 'Model comparison on this photo' },
    'dyn.cmp.model': { ua: 'Модель', en: 'Model' },
    'dyn.cmp.city':  { ua: 'Топ-місто', en: 'Top city' },
    'dyn.cmp.conf':  { ua: 'Впевненість', en: 'Confidence' },
    'dyn.cmp.ood':   { ua: 'OOD-гейт', en: 'OOD gate' },
    'dyn.cmp.run':   { ua: 'Рахуємо всі моделі…', en: 'Running all models…' },
    'dyn.tag.ood':   { ua: 'OOD', en: 'OOD' },
    'dyn.tag.ok':    { ua: 'у межах', en: 'in-domain' },
    'dyn.attn.run':  { ua: 'Рахуємо карту уваги…', en: 'Computing attention map…' },
  };

  let lang = localStorage.getItem('lang') || 'ua';
  const listeners = [];

  function t(key) {
    const e = DICT[key];
    return e ? (e[lang] || e.ua) : key;
  }

  function apply() {
    document.documentElement.lang = lang;
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const k = el.getAttribute('data-i18n');
      if (DICT[k]) el.textContent = t(k);
    });
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
      const k = el.getAttribute('data-i18n-html');
      if (DICT[k]) el.innerHTML = t(k);
    });
    const btn = document.getElementById('langToggle');
    if (btn) btn.textContent = lang === 'ua' ? 'EN' : 'UA';
    listeners.forEach(fn => { try { fn(lang); } catch (e) {} });
  }

  function setLang(l) {
    lang = l; localStorage.setItem('lang', l); apply();
  }

  window.i18n = {
    t, apply, setLang,
    get lang() { return lang; },
    toggle() { setLang(lang === 'ua' ? 'en' : 'ua'); },
    onChange(fn) { listeners.push(fn); },
  };

  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('langToggle');
    if (btn) btn.addEventListener('click', () => window.i18n.toggle());
    apply();
  });
})();
