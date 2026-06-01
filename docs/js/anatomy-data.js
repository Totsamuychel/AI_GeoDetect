/* ============================================================================
   anatomy-data.js — Детальний опис внутрішньої будови нейромереж для 3D-сцени.
   Включає мікроархітектуру кожного блоку (details), що розкривається при dblclick.
   Поле `shape` — форма тензора на ВИХОДІ блоку (для анімації forward-pass).
   ============================================================================ */

const ANATOMY_CATEGORIES = {
  input:       { color: 0x21d4a8, label: 'Вхідні дані' },
  conv_frozen: { color: 0x3b66c4, label: 'Згортки (Заморожено)' },
  conv_train:  { color: 0xffb454, label: 'Згортки (Донавчено)' },
  vit_frozen:  { color: 0x5a3bc4, label: 'Трансформер (Заморожено)' },
  vit_train:   { color: 0xff8c42, label: 'Трансформер (Донавчено)' },
  pool:        { color: 0x5fa8d3, label: 'Пулінг / Токени' },
  dense:       { color: 0xe05e5e, label: 'Повнозв\'язні (Linear)' },
  norm_act:    { color: 0x8a96ad, label: 'Нормалізація / Активація' },
  embed:       { color: 0xb07cff, label: 'Простір ознак (Embedding)' },
  loss:        { color: 0xffd166, label: 'Оптимізація (Loss)' },
  output:      { color: 0x00f5d4, label: 'Вихід (Softmax)' },
};

const ANATOMY = {
  baseline: {
    title: 'Baseline CNN — EfficientNet-B2',
    summary: 'Класична згорткова мережа з механізмом Squeeze-and-Excitation. Відмінно знаходить локальні ознаки (текстури, вікна, фасади). Подвійний клік розкриває мікроархітектуру блоку.',
    blocks: [
      { id: 'in', cat: 'input', w: 0.8, title: 'Вхід 260×260×3', shape: '260×260×3', desc: 'Масштабування та нормалізація за константами ImageNet.', chips: ['RGB', 'Norm'],
        details: [{title: 'Bilinear Resize (260x260)', color: 0x21d4a8}, {title: 'To Tensor [0, 1]', color: 0x5fa8d3}, {title: 'Normalize (ImageNet Mean/Std)', color: 0x8a96ad}] },
      { id: 'stem', cat: 'conv_frozen', w: 1.0, title: 'Stem Conv 3×3', shape: '130×130×32', desc: 'Перша згортка з кроком 2, яка зменшує роздільну здатність удвічі та виділяє базові грані.', chips: ['130×130×32', 'Swish'],
        details: [{title: 'Conv2D 3x3 (Stride=2)', color: 0x3b66c4}, {title: 'Batch Normalization', color: 0x8a96ad}, {title: 'Swish Activation', color: 0x00f5d4}] },
      { id: 'stage1_3', cat: 'conv_frozen', w: 1.5, stack: 3, title: 'MBConv Стадії 1–3', shape: '33×33×48', desc: 'Блоки Mobile Inverted Bottleneck із замороженими вагами. Витягують загальні ознаки форм.', chips: ['Depthwise Conv', 'SE-блоки', 'Заморожено'],
        details: [{title: '1x1 Conv (Expand)', color: 0xe05e5e}, {title: 'BatchNorm + Swish', color: 0x8a96ad}, {title: 'Depthwise Conv 3x3', color: 0xffb454}, {title: 'Squeeze & Excitation', color: 0x5fa8d3}, {title: '1x1 Conv (Project)', color: 0xe05e5e}, {title: 'DropConnect', color: 0x8a96ad}] },
      { id: 'stage4_5', cat: 'conv_frozen', w: 1.3, stack: 2, title: 'MBConv Стадії 4–5', shape: '17×17×120', desc: 'Більш глибокі шари, що розуміють складні патерни.', chips: ['17×17', 'Заморожено'],
        details: [{title: '1x1 Conv (Expand)', color: 0xe05e5e}, {title: 'Depthwise Conv 5x5', color: 0xffb454}, {title: 'Squeeze & Excitation', color: 0x5fa8d3}, {title: '1x1 Conv (Project)', color: 0xe05e5e}] },
      { id: 'stage6_7', cat: 'conv_train', w: 1.4, stack: 2, title: 'MBConv Стадії 6–7', shape: '9×9×352', desc: 'Ці шари були розморожені (fine-tuning) для адаптації під міські специфіки (дахи, транспорт).', chips: ['9×9×352', 'Донавчено'],
        details: [{title: '1x1 Conv (Expand)', color: 0xffb454}, {title: 'Depthwise Conv 5x5', color: 0xffb454}, {title: 'Squeeze & Excitation', color: 0x5fa8d3}, {title: '1x1 Conv (Project)', color: 0xffb454}] },
      { id: 'head_conv', cat: 'conv_train', w: 1.0, title: 'Head Conv 1×1', shape: '9×9×1408', desc: 'Збільшення кількості каналів перед пулінгом.', chips: ['9×9×1408', 'Swish'],
        details: [{title: 'Conv2D 1x1', color: 0xffb454}, {title: 'Batch Normalization', color: 0x8a96ad}, {title: 'Swish Activation', color: 0x00f5d4}] },
      { id: 'pool', cat: 'pool', w: 0.9, title: 'Global Avg Pooling', shape: '1408', desc: 'Усереднення ознак всієї сітки (9×9) в один вектор.', chips: ['1408-D'],
        details: [{title: 'AdaptiveAvgPool2d(1)', color: 0x5fa8d3}, {title: 'Flatten (1408)', color: 0x5fa8d3}] },
      { id: 'dense1', cat: 'dense', w: 1.1, title: 'Linear 1408 → 512', shape: '512', desc: 'Стискання простору ознак до 512.', chips: ['BatchNorm', 'ReLU'],
        details: [{title: 'Linear(1408 → 512)', color: 0xe05e5e}, {title: 'Batch Normalization', color: 0x8a96ad}, {title: 'ReLU Activation', color: 0x00f5d4}, {title: 'Dropout (0.3)', color: 0x8a96ad}] },
      { id: 'cls', cat: 'dense', w: 0.9, title: 'Linear 512 → 3', shape: '3', desc: 'Фінальні логіти для трьох міст.', chips: ['Логіти'],
        details: [{title: 'Linear(512 → 3)', color: 0xe05e5e}] },
      { id: 'out', cat: 'output', w: 0.9, title: 'Softmax', shape: '3 (ймовірності)', desc: 'Перетворення у ймовірності.', chips: ['Варшава', 'Прага', 'Будапешт'],
        details: [{title: 'Softmax(dim=1)', color: 0x00f5d4}] },
    ],
  },

  streetclip: {
    title: 'StreetCLIP — ViT-L/14 @336',
    summary: 'Трансформер-енкодер, дотренований на вуличних фото. Розуміє глобальний контекст та стиль міста за рахунок механізмів уваги. Подвійний клік розкриває мікроархітектуру.',
    blocks: [
      { id: 'in', cat: 'input', w: 0.8, title: 'Вхід 336×336×3', shape: '336×336×3', desc: 'Великий розмір зображення для кращої деталізації текстур.', chips: ['RGB', 'CLIP Norm'],
        details: [{title: 'Bicubic Resize (336x336)', color: 0x21d4a8}, {title: 'CLIP Normalization', color: 0x8a96ad}] },
      { id: 'patch', cat: 'conv_frozen', w: 1.2, title: 'Patch Embed 14×14', shape: '576×1024', desc: 'Нарізка зображення на патчі та лінійна проєкція у токени.', chips: ['576 патчів', '1024-D'],
        details: [{title: 'Conv2D (stride=14) → Patches', color: 0x3b66c4}, {title: 'Flatten (196x3 → 588)', color: 0x5fa8d3}, {title: 'Linear Projection (1024)', color: 0xe05e5e}] },
      { id: 'cls_pos', cat: 'pool', w: 1.2, title: 'CLS + Pos Embed', shape: '577×1024', desc: 'Додавання токену класифікації (CLS) та позиційних векторів.', chips: ['577 токенів'],
        details: [{title: 'Concat [CLS Token]', color: 0x5fa8d3}, {title: 'Add Positional Embedding', color: 0xb07cff}] },
      { id: 'vit_frozen', cat: 'vit_frozen', w: 2.2, stack: 4, title: 'ViT-L: Шари 1–22', shape: '577×1024', desc: 'Гігантські трансформер-блоки (Multi-Head Attention + MLP) із замороженими вагами.', chips: ['MHA 16 голів', 'Заморожено'],
        details: [{title: 'LayerNorm', color: 0x8a96ad}, {title: 'Multi-Head Attention (16 heads)', color: 0x5a3bc4}, {title: 'Residual Add', color: 0xffd166}, {title: 'LayerNorm', color: 0x8a96ad}, {title: 'MLP (Expand → GELU → Project)', color: 0xe05e5e}, {title: 'Residual Add', color: 0xffd166}] },
      { id: 'vit_train', cat: 'vit_train', w: 1.3, stack: 2, title: 'ViT-L: Шари 23–24', shape: '577×1024', desc: 'Останні 2 шари донавчені з дуже малим learning rate.', chips: ['Донавчено', 'lr 2e-5'],
        details: [{title: 'LayerNorm', color: 0x8a96ad}, {title: 'Multi-Head Attention', color: 0xff8c42}, {title: 'Residual Add', color: 0xffd166}, {title: 'LayerNorm', color: 0x8a96ad}, {title: 'MLP (GELU)', color: 0xff8c42}, {title: 'Residual Add', color: 0xffd166}] },
      { id: 'cls_ext', cat: 'pool', w: 1.0, title: 'Вилучення CLS', shape: '1×1024', desc: 'Забір лише одного токена (CLS), який зібрав інформацію з усіх інших патчів.', chips: ['1024-D'],
        details: [{title: 'Slice Tensor [:, 0, :]', color: 0x5fa8d3}] },
      { id: 'proj', cat: 'embed', w: 1.2, title: 'Visual Projection', shape: '768', desc: 'Проєкція з ViT простору у простір ознак CLIP та L2-нормалізація.', chips: ['768-D', 'L2 Norm'],
        details: [{title: 'Linear Projection (1024 → 768)', color: 0xe05e5e}, {title: 'L2 Normalization', color: 0x8a96ad}] },
      { id: 'probe1', cat: 'dense', w: 1.1, title: 'Лінійний Пробник', shape: '256', desc: 'Легка навчена «голова» поверх заморожених ознак.', chips: ['LayerNorm', '768 → 256'],
        details: [{title: 'LayerNorm', color: 0x8a96ad}, {title: 'Linear(768 → 256)', color: 0xe05e5e}, {title: 'GELU Activation', color: 0x00f5d4}, {title: 'Dropout (0.2)', color: 0x8a96ad}] },
      { id: 'cls', cat: 'dense', w: 0.9, title: 'Linear 256 → 3', shape: '3', desc: 'Фінальні логіти.', chips: ['Логіти'],
        details: [{title: 'Linear(256 → 3)', color: 0xe05e5e}] },
      { id: 'out', cat: 'output', w: 0.9, title: 'Softmax', shape: '3 (ймовірності)', desc: 'Фінальні ймовірності міст. OOD перевіряється за 768-D ембедингом.', chips: ['Варшава', 'Прага', 'Будапешт'],
        details: [{title: 'Softmax(dim=1)', color: 0x00f5d4}] },
    ],
  },

  geoclip: {
    title: 'GeoCLIP — CLIP ViT-B/32 + GPS',
    summary: 'Двогілкова архітектура. Під час навчання GPS-гілка допомагає візуальному енкодеру навчитися пов\'язувати фото з географією.',
    blocks: [
      /* Image Branch */
      { id: 'in', cat: 'input', w: 0.8, title: 'Вхід 224×224×3', shape: '224×224×3', desc: 'Фото вулиці (роздільна здатність ViT-B).', chips: ['RGB'],
        details: [{title: 'Resize (224x224)', color: 0x21d4a8}, {title: 'CLIP Normalization', color: 0x8a96ad}] },
      { id: 'patch', cat: 'conv_frozen', w: 1.1, title: 'Patch Embed 32×32', shape: '49×768', desc: 'Розбиття на великі патчі.', chips: ['49 патчів', '768-D'],
        details: [{title: 'Conv2D (stride=32)', color: 0x3b66c4}, {title: 'Flatten', color: 0x5fa8d3}] },
      { id: 'cls', cat: 'pool', w: 1.0, title: 'CLS + Pos Embed', shape: '50×768', desc: 'Токен класифікації + позиції.', chips: ['50 токенів'],
        details: [{title: 'Concat [CLS]', color: 0x5fa8d3}, {title: 'Positional Embedding', color: 0xb07cff}] },
      { id: 'vit_frozen', cat: 'vit_frozen', w: 1.8, stack: 3, title: 'ViT-B: Шари 1–10', shape: '50×768', desc: 'Заморожені блоки базового CLIP ViT-B.', chips: ['MHA 12 голів', 'Заморожено'],
        details: [{title: 'Multi-Head Attention (12 heads)', color: 0x5a3bc4}, {title: 'MLP', color: 0xe05e5e}, {title: 'Residual Connections', color: 0xffd166}] },
      { id: 'vit_train', cat: 'vit_train', w: 1.2, stack: 2, title: 'ViT-B: Шари 11–12', shape: '50×768', desc: 'Донавчання для виділення архітектурних стилів міст.', chips: ['Донавчено'],
        details: [{title: 'Multi-Head Attention', color: 0xff8c42}, {title: 'MLP', color: 0xff8c42}, {title: 'Residual Connections', color: 0xffd166}] },
      { id: 'img_proj', cat: 'embed', w: 1.2, title: 'Image Projection', shape: '512', desc: 'Мультимодальний ембединг фотографії.', chips: ['512-D', 'L2 Norm'],
        details: [{title: 'Linear Projection (768 → 512)', color: 0xe05e5e}, {title: 'L2 Normalization', color: 0x8a96ad}] },
      { id: 'head1', cat: 'dense', w: 1.1, title: 'Linear 512 → 256', shape: '256', desc: 'MLP-голова для інференсу.', chips: ['GELU', 'Drop 0.2'],
        details: [{title: 'Linear(512 → 256)', color: 0xe05e5e}, {title: 'GELU', color: 0x00f5d4}, {title: 'Dropout (0.2)', color: 0x8a96ad}] },
      { id: 'head2', cat: 'dense', w: 0.9, title: 'Linear 256 → 3', shape: '3', desc: 'Логіти класів.', chips: ['Логіти'],
        details: [{title: 'Linear(256 → 3)', color: 0xe05e5e}] },
      { id: 'out', cat: 'output', w: 0.8, title: 'Softmax', shape: '3 (ймовірності)', desc: 'Імовірності для інференсу.', chips: ['3 міста'],
        details: [{title: 'Softmax', color: 0x00f5d4}] },

      /* GPS Branch */
      { id: 'gps_in', cat: 'input', w: 0.8, branch: 'gps', title: 'GPS (Lat, Lon)', shape: '2 (lat, lon)', desc: 'Точні географічні координати фотографії (тільки при навчанні).', chips: ['Lat, Lon'],
        details: [{title: 'Latitude, Longitude Array', color: 0x21d4a8}] },
      { id: 'rff', cat: 'embed', w: 1.2, branch: 'gps', title: 'Fourier Features', shape: '512', desc: 'Проєкція простору у високорозмірні синуси/косинуси.', chips: ['Sin/Cos'],
        details: [{title: 'Random Gaussian Matrix Projection', color: 0xe05e5e}, {title: 'Sin() Activation', color: 0x00f5d4}, {title: 'Cos() Activation', color: 0x00f5d4}, {title: 'Concat [Sin, Cos]', color: 0x5fa8d3}] },
      { id: 'gps_mlp', cat: 'dense', w: 1.4, stack: 3, branch: 'gps', title: 'GPS MLP Енкодер', shape: '512', desc: 'Шари Linear + GELU для перетворення координат у вектор.', chips: ['Linear', 'GELU'],
        details: [{title: 'Linear (512 → 512) + GELU', color: 0xe05e5e}, {title: 'Linear (512 → 512)', color: 0xe05e5e}, {title: 'LayerNorm', color: 0x8a96ad}] },
      { id: 'gps_proj', cat: 'embed', w: 1.2, branch: 'gps', title: 'GPS Projection', shape: '512', desc: 'Кінцевий вектор локації.', chips: ['512-D', 'L2 Norm'],
        details: [{title: 'Linear Projection', color: 0xe05e5e}, {title: 'L2 Normalization', color: 0x8a96ad}] },
      { id: 'contrast', cat: 'loss', w: 1.2, branch: 'gps', linkTo: 'img_proj', title: 'Contrastive Loss', shape: 'loss (скаляр)', desc: 'InfoNCE loss. Під час навчання зіштовхує GPS та Image ембединги.', chips: ['InfoNCE'],
        details: [{title: 'Cosine Similarity (Image ↔ GPS)', color: 0x5fa8d3}, {title: 'Temperature Scaling', color: 0xffb454}, {title: 'Cross Entropy (Log-Sum-Exp)', color: 0xffd166}] },
    ],
  },
};
