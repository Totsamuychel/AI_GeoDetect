/* ============================================================================
   anatomy.js — Інтерактивна 3D-візуалізація будови моделей (Three.js).
   Обертання (OrbitControls), масштаб, клік по блоках → опис, 
   ПОДВІЙНИЙ КЛІК → анімація "розпаду" блоку на мікроструктуру.
   ============================================================================ */

(function () {
  const host = document.getElementById('canvasHost');
  if (!host || typeof THREE === 'undefined') return;

  const UNIT = 1.6;      // масштаб ширини блоку
  const GAP = 0.65;      // проміжок між блоками
  const H = 1.35, D = 1.35;

  let scene, camera, renderer, controls, raycaster, pointer;
  let modelGroup, blockMeshes = [], selectedMesh = null;
  let currentKey = 'streetclip';

  // Змінні для ефекту розпаду (explode)
  let explodedGroup = null;
  let explodingMeshes = [];

  // Forward-pass анімація: упорядкований шлях потоку, «імпульси» та стан
  let mainFlow = [], gpsFlow = [];
  let pulseMesh = null, gpsPulseMesh = null;
  let flowState = { active: false, lanes: [], holdFrames: 0 };

  // ── Текстова мітка над блоком (sprite) ─────────────────────────────────
  function makeLabel(text, fontSize = 44) {
    const pad = 24;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.font = `700 ${fontSize}px Inter, Segoe UI, sans-serif`;
    const w = ctx.measureText(text).width + pad * 2;
    canvas.width = w; canvas.height = fontSize + pad;
    ctx.font = `700 ${fontSize}px Inter, Segoe UI, sans-serif`;
    ctx.fillStyle = 'rgba(10,14,23,0.82)';
    roundRect(ctx, 0, 0, canvas.width, canvas.height, 14); ctx.fill();
    ctx.fillStyle = '#e8edf6';
    ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    const scale = 0.0042;
    sprite.scale.set(canvas.width * scale, canvas.height * scale, 1);
    return sprite;
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // ── Ініціалізація сцени ────────────────────────────────────────────────
  function init() {
    scene = new THREE.Scene();
    const W = host.clientWidth, Ht = host.clientHeight || 460;
    camera = new THREE.PerspectiveCamera(45, W / Ht, 0.1, 100);
    camera.position.set(2, 5, 20);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, Ht);
    host.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.6;
    controls.minDistance = 6;
    controls.maxDistance = 45;
    controls.addEventListener('start', () => { controls.autoRotate = false; });

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(6, 12, 8); scene.add(key);
    const fill = new THREE.DirectionalLight(0x5577ff, 0.4);
    fill.position.set(-8, 4, -6); scene.add(fill);

    raycaster = new THREE.Raycaster();
    pointer = new THREE.Vector2();
    renderer.domElement.addEventListener('pointerdown', onPick);
    renderer.domElement.addEventListener('dblclick', onDoubleClick); // <--- ДОДАНО

    // Fullscreen toggle
    const fsBtn = document.getElementById('fullscreenBtn');
    if (fsBtn) {
      fsBtn.addEventListener('click', () => {
        host.classList.toggle('fullscreen');
        const isFS = host.classList.contains('fullscreen');
        fsBtn.innerHTML = isFS ? '✕' : '⛶';
        fsBtn.title = isFS ? 'Закрити' : 'На весь екран';
        if (!isFS) stopFlow(); // Зупиняємо forward-pass при ВИХОДІ з повноекранного режиму
        onResize();
      });
    }

    // Escape key to exit fullscreen
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && host.classList.contains('fullscreen')) {
        host.classList.remove('fullscreen');
        if (fsBtn) {
          fsBtn.innerHTML = '⛶';
          fsBtn.title = 'На весь екран';
        }
        stopFlow();
        onResize();
      }
    });

    // Кнопка запуску forward-pass + HUD з формою тензора
    const playBtn = document.createElement('button');
    playBtn.id = 'playFlowBtn';
    playBtn.className = 'flow-btn';
    playBtn.innerHTML = '▶ Запустити прохід';
    playBtn.title = 'Показати, як дані проходять крізь мережу';
    host.appendChild(playBtn);
    playBtn.addEventListener('click', () => {
      if (flowState && flowState.active) stopFlow();
      else startFlow();
    });

    const hud = document.createElement('div');
    hud.id = 'flowHud';
    hud.className = 'flow-hud';
    host.appendChild(hud);

    window.addEventListener('resize', onResize);
    if (window.ResizeObserver) new ResizeObserver(onResize).observe(host);

    // Додаємо підказку про подвійний клік
    const hintEl = document.querySelector('.canvas-hint');
    if (hintEl) hintEl.innerHTML += ' · <b>Подвійний клік</b> — мікроархітектура';

    buildModel(currentKey);
    animate();
  }

  // ── Побудова блоків моделі ─────────────────────────────────────────────
  function buildModel(key) {
    currentKey = key;
    if (modelGroup) { scene.remove(modelGroup); disposeGroup(modelGroup); }
    modelGroup = new THREE.Group();
    blockMeshes = []; selectedMesh = null;
    clearExplosion(); // Скидаємо розпад при зміні моделі
    resetFlowState();  // Скидаємо forward-pass анімацію
    mainFlow = []; gpsFlow = [];

    const data = ANATOMY[key];
    const mainBlocks = data.blocks.filter(b => b.branch !== 'gps');
    const gpsBlocks = data.blocks.filter(b => b.branch === 'gps');
    const centersMap = {};

    // Розкладка головного потоку
    const totalW = mainBlocks.reduce((s, b) => s + b.w * UNIT + GAP, -GAP);
    let x = -totalW / 2;
    const centers = [];
    mainBlocks.forEach((b) => {
      const bw = b.w * UNIT;
      const cx = x + bw / 2;
      addBlock(b, cx, 0, 0, bw);
      centersMap[b.id] = new THREE.Vector3(cx, 0, 0);
      centers.push({ x1: x, x2: x + bw, cx, id: b.id });
      mainFlow.push({ id: b.id, block: b, pos: new THREE.Vector3(cx, 0, 0) });
      x += bw + GAP;
    });
    // Стрілки між блоками головного потоку
    for (let i = 0; i < centers.length - 1; i++) {
      addArrow(centers[i].x2, centers[i + 1].x1, 0, 0);
    }

    // GPS-гілка (паралельно, позаду по Z)
    if (gpsBlocks.length) {
      const Z = -5;
      const gW = gpsBlocks.reduce((s, b) => s + b.w * UNIT + GAP, -GAP);
      let gx = -gW / 2;
      const gc = [];
      gpsBlocks.forEach((b) => {
        const bw = b.w * UNIT;
        const cx = gx + bw / 2;
        addBlock(b, cx, 0, Z, bw);
        centersMap[b.id] = new THREE.Vector3(cx, 0, Z);
        gc.push({ x1: gx, x2: gx + bw, cx, id: b.id });
        gpsFlow.push({ id: b.id, block: b, pos: new THREE.Vector3(cx, 0, Z) });
        gx += bw + GAP;
      });
      for (let i = 0; i < gc.length - 1; i++) addArrow(gc[i].x2, gc[i + 1].x1, 0, Z, 0xb07cff);
    }

    // Перехресні зв'язки
    data.blocks.forEach(b => {
      if (b.linkTo && centersMap[b.id] && centersMap[b.linkTo]) {
        const p1 = centersMap[b.id];
        const p2 = centersMap[b.linkTo];
        const dir = new THREE.Vector3().subVectors(p2, p1);
        const len = dir.length() - (b.w * UNIT) / 2 - 0.2;
        dir.normalize();
        
        const start = p1.clone().add(dir.clone().multiplyScalar((b.w * UNIT) / 2));
        const color = ANATOMY_CATEGORIES[b.cat].color;
        const arrow = new THREE.ArrowHelper(dir, start, len, color, 0.4, 0.25);
        modelGroup.add(arrow);
      }
    });

    scene.add(modelGroup);
    createPulses(); // «Імпульси» для forward-pass (після побудови modelGroup)
    document.getElementById('anatomyTabs') && updateSummary(data);
    autoFrame();
  }

  function addBlock(b, x, y, z, bw) {
    const cat = ANATOMY_CATEGORIES[b.cat];
    const group = new THREE.Group();
    group.position.set(x, y, z);
    
    const userData = { block: b, baseEmissive: 0.12 };

    const stacks = b.stack || 1;
    const gap = 0.08;
    const h = (H - gap * (stacks - 1)) / stacks;

    for (let i = 0; i < stacks; i++) {
      const geo = new THREE.BoxGeometry(bw, h, D);
      const mat = new THREE.MeshStandardMaterial({
        color: cat.color, metalness: 0.35, roughness: 0.45,
        emissive: cat.color, emissiveIntensity: 0.12,
      });
      const mesh = new THREE.Mesh(geo, mat);
      
      const yOff = (i - (stacks - 1) / 2) * (h + gap);
      mesh.position.set(0, yOff, 0);
      mesh.userData = userData; 

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 })
      );
      mesh.add(edges);
      
      group.add(mesh);
      blockMeshes.push(mesh);
    }

    const label = makeLabel(b.title);
    label.position.set(0, H / 2 + 0.55, 0);
    group.add(label);

    modelGroup.add(group);
  }

  function addArrow(x1, x2, y, z, color) {
    const dir = new THREE.Vector3(1, 0, 0);
    const origin = new THREE.Vector3(x1, y, z);
    const len = Math.max(x2 - x1, 0.05);
    const arrow = new THREE.ArrowHelper(dir, origin, len, color || 0x4f8cff, 0.3, 0.2);
    modelGroup.add(arrow);
  }

  // ── Вибір блоку (Одинарний клік) ───────────────────────────────────────
  function onPick(ev) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(blockMeshes, false);
    if (hits.length) {
        select(hits[0].object);
    }
  }

  function select(mesh) {
    if (selectedMesh && selectedMesh.userData === mesh.userData) return;

    if (selectedMesh) {
      blockMeshes.forEach(m => {
        if (m.userData === selectedMesh.userData) {
          m.material.emissiveIntensity = m.userData.baseEmissive;
          m.scale.set(1, 1, 1);
        }
      });
    }
    selectedMesh = mesh;
    blockMeshes.forEach(m => {
      if (m.userData === mesh.userData) {
        m.material.emissiveIntensity = 0.55;
        m.scale.set(1.06, 1.06, 1.06);
      }
    });
    
    // Якщо при виборі нового блоку був відкритий розпад іншого - закриваємо
    if (explodedGroup && explodedGroup.userData.blockId !== mesh.userData.block.id) {
        clearExplosion();
    }
    
    showInfo(mesh.userData.block);
  }

  function showInfo(b) {
    const cat = ANATOMY_CATEGORIES[b.cat];
    document.getElementById('blockTitle').textContent = b.title;
    
    let htmlDesc = b.desc;
    if (b.details && b.details.length > 0) {
        htmlDesc += '<br/><br/><span style="color:var(--muted); font-size:12px;"><i>Подвійний клік на блок покаже деталі</i></span>';
    }
    // Якщо блок вже розгорнутий - показуємо деталі в тексті
    if (explodedGroup && explodedGroup.userData.blockId === b.id) {
        htmlDesc = b.desc + '<br/><br/><b style="color:var(--text)">Мікроархітектура:</b><ul style="margin-top:6px; padding-left:18px; color:var(--accent-2); font-size:13px;">';
        b.details.forEach(d => {
           htmlDesc += `<li style="margin-bottom:4px;">${d.title}</li>`;
        });
        htmlDesc += '</ul>';
    }
    document.getElementById('blockDesc').innerHTML = htmlDesc;

    const meta = document.getElementById('blockMeta');
    const catChip = `<span class="chip" style="border-color:#${cat.color.toString(16)}">
        <b style="color:#${cat.color.toString(16)}">${cat.label}</b></span>`;
    const chips = (b.chips || []).map(c => `<span class="chip">${c}</span>`).join('');
    meta.innerHTML = catChip + chips;
  }

  // ── РОЗПАД БЛОКУ (Подвійний клік) ───────────────────────────────────────
  function onDoubleClick(ev) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(blockMeshes, false);
    
    if (hits.length) {
        explodeBlock(hits[0].object);
    } else {
        clearExplosion();
    }
  }

  function clearExplosion() {
    if (explodedGroup) {
      modelGroup.remove(explodedGroup);
      disposeGroup(explodedGroup);
      explodedGroup = null;
      explodingMeshes = [];
      // Відновлюємо опис для вибраного блоку
      if (selectedMesh) showInfo(selectedMesh.userData.block);
    }
  }

  function explodeBlock(mesh) {
    const b = mesh.userData.block;
    if (!b.details || !b.details.length) return; // Немає чого розкладати

    // Якщо клікаємо по вже розкладеному - просто згортаємо
    if (explodedGroup && explodedGroup.userData.blockId === b.id) {
      clearExplosion();
      return;
    }
    
    clearExplosion();

    explodedGroup = new THREE.Group();
    explodedGroup.userData.blockId = b.id;
    modelGroup.add(explodedGroup);

    // Отримуємо позицію блоку у локальних координатах моделі
    const basePos = new THREE.Vector3();
    mesh.getWorldPosition(basePos);
    modelGroup.worldToLocal(basePos);
    explodedGroup.position.copy(basePos);

    // Вертикальна лінія-стебло
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 1.2 + b.details.length * 1.0, 0)
    ]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 });
    explodedGroup.add(new THREE.Line(lineGeo, lineMat));

    // Створюємо мікро-блоки (details)
    b.details.forEach((det, i) => {
      const geo = new THREE.BoxGeometry(b.w * UNIT * 0.85, 0.25, D * 0.85);
      const c = det.color || ANATOMY_CATEGORIES[b.cat]?.color || 0xffffff;
      const mat = new THREE.MeshStandardMaterial({
        color: c, emissive: c, emissiveIntensity: 0.3, transparent: true, opacity: 0
      });
      const slice = new THREE.Mesh(geo, mat);

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 })
      );
      slice.add(edges);

      const lbl = makeLabel(det.title, 26); // Менший шрифт
      lbl.position.set(0, 0.45, 0);
      lbl.material.opacity = 0;
      slice.add(lbl);

      // Початкова позиція (внізу, всередині материнського блоку)
      slice.position.set(0, 0, 0); 
      // Цільова позиція (вгорі)
      const targetY = 1.4 + i * 1.0;

      explodedGroup.add(slice);
      explodingMeshes.push({ mesh: slice, edges, lbl, targetY });
    });

    // Оновлюємо UI панель, щоб показати розширений опис
    showInfo(b);
  }

  // ── FORWARD-PASS: анімація проходження даних крізь мережу ───────────────
  function clearLights() {
    blockMeshes.forEach(m => { m.material.emissiveIntensity = m.userData.baseEmissive; });
  }

  function lightBlock(blockId, on) {
    blockMeshes.forEach(m => {
      if (m.userData.block.id === blockId) {
        m.material.emissiveIntensity = on ? 0.85 : m.userData.baseEmissive;
      }
    });
  }

  function makePulse(color) {
    const g = new THREE.Group();
    g.add(new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 20, 20),
      new THREE.MeshBasicMaterial({ color })
    ));
    g.add(new THREE.Mesh(
      new THREE.SphereGeometry(0.55, 20, 20),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.22, depthWrite: false })
    ));
    g.visible = false;
    modelGroup.add(g);
    return g;
  }

  function createPulses() {
    pulseMesh = makePulse(0x7be0ff);
    gpsPulseMesh = gpsFlow.length ? makePulse(0xc79bff) : null;
  }

  // Повне скидання стану (при перебудові моделі — старі pulse вже знищено разом з групою)
  function resetFlowState() {
    flowState = { active: false, lanes: [], holdFrames: 0 };
    pulseMesh = null;
    gpsPulseMesh = null;
    const btn = document.getElementById('playFlowBtn');
    if (btn) btn.innerHTML = '▶ Запустити прохід';
    const hud = document.getElementById('flowHud');
    if (hud) hud.classList.remove('show');
  }

  function setHud(block) {
    const hud = document.getElementById('flowHud');
    if (!hud) return;
    hud.innerHTML = `${block.title} · <b>${block.shape || ''}</b>`;
    hud.classList.add('show');
  }

  function startFlow() {
    if (!mainFlow.length || !pulseMesh) return;

    // Прибираємо ручне виділення та розпад, щоб не конфліктували з підсвіткою
    clearExplosion();
    if (selectedMesh) {
      blockMeshes.forEach(m => { if (m.userData === selectedMesh.userData) m.scale.set(1, 1, 1); });
      selectedMesh = null;
    }
    clearLights();
    controls.autoRotate = false;

    const lanes = [{ path: mainFlow, pulse: pulseMesh, seg: 0, t: 0, done: false, speed: 0.04, primary: true }];
    if (gpsFlow.length && gpsPulseMesh) {
      lanes.push({ path: gpsFlow, pulse: gpsPulseMesh, seg: 0, t: 0, done: false, speed: 0.04, primary: false });
    }
    lanes.forEach(lane => {
      lane.pulse.visible = true;
      lane.pulse.position.copy(lane.path[0].pos);
      lightBlock(lane.path[0].id, true);
      if (lane.primary) { showInfo(lane.path[0].block); setHud(lane.path[0].block); }
    });
    flowState = { active: true, lanes, holdFrames: 0 };

    const btn = document.getElementById('playFlowBtn');
    if (btn) btn.innerHTML = '⏸ Зупинити';
  }

  function stopFlow() {
    if (pulseMesh) pulseMesh.visible = false;
    if (gpsPulseMesh) gpsPulseMesh.visible = false;
    clearLights();
    if (flowState) { flowState.active = false; flowState.lanes = []; }
    const btn = document.getElementById('playFlowBtn');
    if (btn) btn.innerHTML = '▶ Запустити прохід';
    const hud = document.getElementById('flowHud');
    if (hud) hud.classList.remove('show');
  }

  function updateFlows() {
    if (!flowState || !flowState.active) return;
    let allDone = true;
    flowState.lanes.forEach(lane => {
      const path = lane.path;
      if (!lane.done) {
        allDone = false;
        lane.t += lane.speed;
        if (lane.t >= 1) {
          lane.t = 0;
          lane.seg++;
          const node = path[lane.seg];
          lightBlock(node.id, true);
          if (lane.primary) { showInfo(node.block); setHud(node.block); }
          if (lane.seg >= path.length - 1) lane.done = true;
        }
      }
      const i = Math.min(lane.seg, path.length - 1);
      const j = Math.min(lane.seg + 1, path.length - 1);
      lane.pulse.position.lerpVectors(path[i].pos, path[j].pos, lane.t);
    });

    if (allDone) {
      flowState.holdFrames++;
      if (flowState.holdFrames > 70) stopFlow(); // ~1.2 с фінального підсвічування Softmax
    }
  }

  // ── Легенда + резюме ──────────────────────────────────────────────────
  function buildLegend() {
    const el = document.getElementById('legend');
    if (!el) return;
    el.innerHTML = Object.values(ANATOMY_CATEGORIES).map(c => `
      <div class="legend-row">
        <span class="legend-dot" style="background:#${c.color.toString(16).padStart(6,'0')}"></span>
        ${c.label}
      </div>`).join('');
  }

  function updateSummary(data) {
    const t = document.getElementById('blockTitle');
    const d = document.getElementById('blockDesc');
    const m = document.getElementById('blockMeta');
    if (!selectedMesh) {
      t.textContent = data.title;
      d.textContent = data.summary;
      m.innerHTML = '<span class="chip">Натисніть на блок, щоб дізнатися більше</span>';
    }
  }

  // ── Рендер-цикл та Анімації ───────────────────────────────────────────
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    
    // Анімація розпаду блоку (Spring/Lerp Effect)
    if (explodingMeshes.length > 0) {
      explodingMeshes.forEach(item => {
        // Плавний рух вгору
        item.mesh.position.y += (item.targetY - item.mesh.position.y) * 0.12;
        
        // Плавна поява (Opacity Fade In)
        if (item.mesh.material.opacity < 0.85) {
          item.mesh.material.opacity += 0.04;
          item.edges.material.opacity += 0.04;
          item.lbl.material.opacity += 0.04;
        }
      });
    }

    updateFlows(); // Рух «імпульсів» forward-pass

    renderer.render(scene, camera);
  }

  function autoFrame() {
    const box = new THREE.Box3().setFromObject(modelGroup);
    const center = box.getCenter(new THREE.Vector3());
    controls.target.copy(center);
    const size = box.getSize(new THREE.Vector3());
    const dist = Math.max(size.x, size.z) * 0.9 + size.y;
    camera.position.set(center.x + 2, center.y + size.y + 4, center.z + dist + 7);
    controls.update();
  }

  function onResize() {
    const W = host.clientWidth, Ht = host.clientHeight || 460;
    camera.aspect = W / Ht; camera.updateProjectionMatrix();
    renderer.setSize(W, Ht);
  }

  function disposeGroup(g) {
    g.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        if (o.material.map) o.material.map.dispose();
        o.material.dispose();
      }
    });
  }

  // ── Вкладки моделей ───────────────────────────────────────────────────
  function buildTabs() {
    const tabs = document.getElementById('anatomyTabs');
    if(!tabs) return;
    const labels = { streetclip: 'StreetCLIP', baseline: 'Baseline CNN', geoclip: 'GeoCLIP' };
    tabs.innerHTML = Object.keys(labels).map(k =>
      `<button class="tab${k === currentKey ? ' active' : ''}" data-key="${k}">${labels[k]}</button>`
    ).join('');
    tabs.querySelectorAll('.tab').forEach(btn => {
      btn.addEventListener('click', () => {
        tabs.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        controls.autoRotate = true;
        buildModel(btn.dataset.key);
      });
    });
  }

  // Старт
  buildTabs();
  buildLegend();
  init();

  // Публічний хук: дозволяємо app.js перемикати вкладку
  window.showAnatomy = function (key) {
    const btn = document.querySelector(`.tab[data-key="${key}"]`);
    if (btn) btn.click();
  };
})();
