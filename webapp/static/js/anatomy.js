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
  let modelGroup, blockMeshes = [], selectedMesh = null, hoveredMesh = null;
  let glowMesh = null;         // Спеціальний меш для світіння
  let particles = null;        // Частинки фону
  let blockLabels = [];        // підписи блоків {label, blockId, baseY} — для зсуву при розпаді
  let currentKey = 'streetclip';

  // Змінні для ефекту розпаду (explode)
  let explodedGroup = null;
  let explodingMeshes = [];
  let microMeshes = [];        // мікро-блоки (деталі) для кліку/поповера

  // Forward-pass анімація: упорядкований шлях потоку, «імпульси» та стан
  let mainFlow = [], gpsFlow = [];
  let pulseMesh = null, gpsPulseMesh = null;
  let flowState = { active: false, lanes: [], holdFrames: 0 };
  let walkIndex = -1;          // поточний крок «прогону» (головний потік)
  let modalEl = null;          // модальне вікно детального опису блоку
  let downX = 0, downY = 0;    // позиція натискання (для відрізнення кліку від обертання)

  // ── Текстова мітка над блоком (sprite) ─────────────────────────────────
  function makeLabel(text, fontSize = 44, isMicro = false) {
    const pad = isMicro ? 14 : 24;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.font = `700 ${fontSize}px Inter, Segoe UI, sans-serif`;
    const w = ctx.measureText(text).width + pad * 2;
    canvas.width = w; canvas.height = fontSize + pad * 1.5;
    
    ctx.font = `700 ${fontSize}px Inter, Segoe UI, sans-serif`;
    ctx.fillStyle = isMicro ? 'rgba(28,38,58,0.95)' : 'rgba(10,14,23,0.88)';
    roundRect(ctx, 0, 0, canvas.width, canvas.height, isMicro ? 8 : 14); ctx.fill();
    
    if (!isMicro) {
        ctx.strokeStyle = 'rgba(79,140,255,0.5)';
        ctx.lineWidth = 4;
        ctx.stroke();
    }

    ctx.fillStyle = '#e8edf6';
    ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    
    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: true, depthWrite: false });
    const sprite = new THREE.Sprite(mat);
    sprite.renderOrder = 2;
    const scale = isMicro ? 0.0032 : 0.0042;
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
    controls.maxDistance = 55;
    controls.addEventListener('start', () => { controls.autoRotate = false; });

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(10, 20, 10); scene.add(key);
    const fill = new THREE.DirectionalLight(0x4f8cff, 0.6);
    fill.position.set(-10, 5, -5); scene.add(fill);

    raycaster = new THREE.Raycaster();
    pointer = new THREE.Vector2();
    renderer.domElement.addEventListener('pointerdown', onPick);
    renderer.domElement.addEventListener('pointermove', onHover);
    renderer.domElement.addEventListener('pointerup', onPointerUp);
    renderer.domElement.addEventListener('dblclick', onDoubleClick);

    // Fullscreen toggle
    const fsBtn = document.getElementById('fullscreenBtn');
    if (fsBtn) fsBtn.addEventListener('click', () => {
      host.classList.contains('fullscreen') ? exitFullscreen() : enterFullscreen();
    });

    const exitBtn = document.createElement('button');
    exitBtn.id = 'fsExitBtn';
    exitBtn.className = 'fs-exit';
    exitBtn.innerHTML = '✕ Вийти';
    exitBtn.title = 'Вийти з повного екрана (Esc)';
    exitBtn.addEventListener('click', exitFullscreen);
    host.appendChild(exitBtn);

    const fsModels = document.createElement('div');
    fsModels.className = 'fs-models';
    const fsLabels = { streetclip: 'StreetCLIP', baseline: 'Baseline CNN', geoclip: 'GeoCLIP' };
    fsModels.innerHTML = Object.keys(fsLabels).map(k =>
      `<button class="fs-model-btn${k === currentKey ? ' active' : ''}" data-key="${k}">${fsLabels[k]}</button>`
    ).join('');
    host.appendChild(fsModels);
    fsModels.querySelectorAll('.fs-model-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const outer = document.querySelector(`.anatomy-tabs .tab[data-key="${btn.dataset.key}"]`);
        if (outer) outer.click(); else buildModel(btn.dataset.key);
      });
    });

    // Reset Camera Button
    const resetBtn = document.createElement('button');
    resetBtn.className = 'fs-btn';
    resetBtn.style.top = '12px';
    resetBtn.style.right = '56px';
    resetBtn.innerHTML = '↺';
    resetBtn.title = 'Скинути камеру';
    resetBtn.addEventListener('click', () => { controls.autoRotate = true; autoFrame(); });
    host.appendChild(resetBtn);

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (modalEl && modalEl.classList.contains('open')) { closeModal(); return; }
        if (host.classList.contains('fullscreen')) exitFullscreen();
        return;
      }
      if (hostInView() && !(modalEl && modalEl.classList.contains('open'))) {
        if (e.key === 'ArrowRight') { e.preventDefault(); walkNext(); }
        else if (e.key === 'ArrowLeft') { e.preventDefault(); walkPrev(); }
        else if (e.key === ' ' || e.code === 'Space') { e.preventDefault(); walkToggle(); }
      }
    });

    const panel = document.createElement('div');
    panel.className = 'walk-panel';
    panel.innerHTML = `
      <div class="walk-steps" id="walkSteps"></div>
      <div class="walk-bar">
        <button id="walkPrevBtn" class="walk-ctrl" title="Попередній крок (←)">◀</button>
        <button id="walkPlayBtn" class="walk-ctrl walk-main" title="Запустити прогін (Пробіл)">▶ Прогін</button>
        <button id="walkNextBtn" class="walk-ctrl" title="Наступний крок (→)">▶</button>
        <div class="walk-progress"><div id="walkBar" class="walk-progress-fill"></div></div>
      </div>
      <div id="walkLabel" class="walk-label">Крок 0 / 0</div>`;
    host.appendChild(panel);
    document.getElementById('walkPrevBtn').addEventListener('click', walkPrev);
    document.getElementById('walkNextBtn').addEventListener('click', walkNext);
    document.getElementById('walkPlayBtn').addEventListener('click', walkToggle);

    const hud = document.createElement('div');
    hud.id = 'flowHud';
    hud.className = 'flow-hud';
    host.appendChild(hud);

    buildModal();

    window.addEventListener('resize', onResize);
    if (window.ResizeObserver) new ResizeObserver(onResize).observe(host);

    const hintEl = document.querySelector('.canvas-hint');
    if (hintEl) hintEl.innerHTML = '🖱️ ЛКМ — огляд · <b>2x клік</b> — деталі';

    buildModel(currentKey);
    createParticles();
    animate();
  }

  function createParticles() {
    const count = 1200;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count * 3; i++) {
        pos[i] = (Math.random() - 0.5) * 60;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
        color: 0x4f8cff,
        size: 0.12,
        transparent: true,
        opacity: 0.4,
        blending: THREE.AdditiveBlending
    });
    particles = new THREE.Points(geo, mat);
    scene.add(particles);
  }

  function createGlowMesh() {
    const geo = new THREE.BoxGeometry(1, 1, 1);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x4f8cff,
      transparent: true,
      opacity: 0.15,
      side: THREE.BackSide
    });
    glowMesh = new THREE.Mesh(geo, mat);
    glowMesh.visible = false;
    scene.add(glowMesh);
  }

  // ── Побудова блоків моделі ─────────────────────────────────────────────
  function buildModel(key) {
    currentKey = key;
    if (modelGroup) { scene.remove(modelGroup); disposeGroup(modelGroup); }
    modelGroup = new THREE.Group();
    blockMeshes = []; selectedMesh = null; hoveredMesh = null; blockLabels = [];
    if (!glowMesh) createGlowMesh(); else glowMesh.visible = false;
    clearExplosion(); 
    resetFlowState(); 
    mainFlow = []; gpsFlow = [];

    const data = ANATOMY[key];
    const mainBlocks = data.blocks.filter(b => b.branch !== 'gps');
    const gpsBlocks = data.blocks.filter(b => b.branch === 'gps');
    const centersMap = {};

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
    for (let i = 0; i < centers.length - 1; i++) addArrow(centers[i].x2, centers[i + 1].x1, 0, 0);

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

    data.blocks.forEach(b => {
      try {
        if (b.linkTo && centersMap[b.id] && centersMap[b.linkTo]) {
          const p1 = centersMap[b.id], p2 = centersMap[b.linkTo];
          const dir = new THREE.Vector3().subVectors(p2, p1);
          const len = dir.length() - (b.w * UNIT) / 2 - 0.2;
          dir.normalize();
          const start = p1.clone().add(dir.clone().multiplyScalar((b.w * UNIT) / 2));
          const cat = ANATOMY_CATEGORIES[b.cat];
          const color = cat ? cat.color : 0x4f8cff;
          modelGroup.add(new THREE.ArrowHelper(dir, start, len, color, 0.4, 0.25));
        }
      } catch (err) { console.warn('anatomy link skip', b.id, err); }
    });

    scene.add(modelGroup);
    createPulses();
    buildWalkSteps();
    resetWalk();
    if (document.getElementById('anatomyTabs')) updateSummary(data);
    document.querySelectorAll('.fs-model-btn').forEach(b => b.classList.toggle('active', b.dataset.key === currentKey));
    autoFrame();
  }

  function addBlock(b, x, y, z, bw) {
    const cat = ANATOMY_CATEGORIES[b.cat];
    const group = new THREE.Group();
    group.position.set(x, y, z);
    
    const userData = { block: b, baseEmissive: 0.18 };
    const stacks = b.stack || 1;
    const gap = 0.14;
    const h = (H - gap * (stacks - 1)) / stacks;

    for (let i = 0; i < stacks; i++) {
      const geo = new THREE.BoxGeometry(bw, h, D);
      const mat = new THREE.MeshPhysicalMaterial({
        color: cat.color,
        metalness: 0.3,
        roughness: 0.1,
        transmission: 0.4,
        thickness: 0.5,
        emissive: cat.color,
        emissiveIntensity: 0.18,
        transparent: true,
        opacity: 0.9,
      });
      const mesh = new THREE.Mesh(geo, mat);
      const yOff = (i - (stacks - 1) / 2) * (h + gap);
      mesh.position.set(0, yOff, 0);
      mesh.userData = userData; 

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 })
      );
      mesh.add(edges);
      group.add(mesh);
      blockMeshes.push(mesh);
    }

    const label = makeLabel(b.title);
    label.position.set(0, H / 2 + 0.85, 0);
    group.add(label);
    blockLabels.push({ label, blockId: b.id, baseY: H / 2 + 0.85 });
    modelGroup.add(group);
  }

  function addArrow(x1, x2, y, z, color) {
    const dir = new THREE.Vector3(1, 0, 0);
    const origin = new THREE.Vector3(x1, y, z);
    const len = Math.max(x2 - x1, 0.05);
    const arrow = new THREE.ArrowHelper(dir, origin, len, color || 0x4f8cff, 0.3, 0.2);
    modelGroup.add(arrow);
  }

  function onHover(ev) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(blockMeshes, false);
    
    if (hits.length) {
      const mesh = hits[0].object;
      if (hoveredMesh !== mesh) {
        if (hoveredMesh && hoveredMesh !== selectedMesh) resetMeshVisual(hoveredMesh);
        hoveredMesh = mesh;
        if (hoveredMesh !== selectedMesh) highlightMesh(hoveredMesh, 0.5);
        document.body.style.cursor = 'pointer';
      }
    } else {
      if (hoveredMesh && hoveredMesh !== selectedMesh) resetMeshVisual(hoveredMesh);
      hoveredMesh = null;
      document.body.style.cursor = 'auto';
    }
  }

  function highlightMesh(mesh, intensity, scale = 1.05) {
    blockMeshes.forEach(m => {
      if (m.userData === mesh.userData) {
        m.material.emissiveIntensity = intensity;
        m.scale.set(scale, scale, scale);
      }
    });
  }

  function resetMeshVisual(mesh) {
    blockMeshes.forEach(m => {
      if (m.userData === mesh.userData) {
        m.material.emissiveIntensity = m.userData.baseEmissive;
        m.material.opacity = 0.9;
        m.scale.set(1, 1, 1);
      }
    });
  }

  function onPick(ev) {
    downX = ev.clientX; downY = ev.clientY;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(blockMeshes, false);
    if (hits.length) select(hits[0].object);
  }

  function onPointerUp(ev) {
    if (Math.abs(ev.clientX - downX) + Math.abs(ev.clientY - downY) > 6) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    if (microMeshes.length) {
      const mh = raycaster.intersectObjects(microMeshes, false);
      if (mh.length) {
        const u = mh[0].object.userData;
        openDetailPopover(u.microDetail, u.parentBlock, ev);
        return;
      }
    }
    const hits = raycaster.intersectObjects(blockMeshes, false);
    if (hits.length) openBlockModal(hits[0].object.userData.block, ev);
  }

  function openDetailPopover(det, parent, ev) {
    if (!modalEl || !det) return;
    const dc = '#' + ((det.color || 0xffffff) >>> 0).toString(16).padStart(6, '0');
    const pName = parent ? parent.title : '';
    document.getElementById('blockModalBody').innerHTML = `
      <div class="bm-head">
        <span class="bm-cat" style="border-color:${dc};color:${dc}">Мікрокрок</span>
        <h3>${det.title}</h3>
      </div>
      <p class="bm-lead">Внутрішня операція блоку <b>${pName}</b>.</p>
      <div class="bm-field"><div class="bm-label">🧩 Контекст</div>
        <p>Один із під-кроків, на які розкладається блок «${pName}» у детальному 3D-перегляді.</p></div>`;
    modalEl.classList.add('open');
    positionPopover(ev);
  }

  function select(mesh) {
    if (selectedMesh && selectedMesh.userData === mesh.userData) return;
    if (selectedMesh) resetMeshVisual(selectedMesh);
    selectedMesh = mesh;
    highlightMesh(selectedMesh, 0.8, 1.1);
    
    if (glowMesh) {
        glowMesh.visible = true;
        const box = new THREE.Box3().setFromObject(mesh);
        const size = new THREE.Vector3(); box.getSize(size);
        const center = new THREE.Vector3(); box.getCenter(center);
        glowMesh.scale.set(size.x * 1.3, size.y * 1.3, size.z * 1.3);
        glowMesh.position.copy(center);
        glowMesh.material.color.set(mesh.material.color);
    }

    if (explodedGroup && explodedGroup.userData.blockId !== mesh.userData.block.id) clearExplosion();
    showInfo(mesh.userData.block);
  }

  function showInfo(b) {
    const cat = ANATOMY_CATEGORIES[b.cat];
    const t = document.getElementById('blockTitle');
    const d = document.getElementById('blockDesc');
    const infoPanel = document.getElementById('blockInfo');
    
    if (infoPanel) infoPanel.classList.add('updating');
    
    setTimeout(() => {
      t.textContent = b.title;
      let htmlDesc = b.desc;
      if (b.details?.length > 0) htmlDesc += '<br/><br/><span class="hint-small">💡 Подвійний клік — розгорнути деталі</span>';
      if (explodedGroup && explodedGroup.userData.blockId === b.id) {
          htmlDesc = b.desc + '<br/><br/><b style="color:var(--text)">Мікроархітектура:</b><ul class="micro-list">';
          b.details.forEach(d => { htmlDesc += `<li>${d.title}</li>`; });
          htmlDesc += '</ul>';
      }
      
      if (b.shape) {
          const dims = b.shape.split('×').map(s => parseInt(s.trim()));
          let tensorHtml = `<div class="tensor-visual ${dims.length > 1 ? 'd3' : 'd1'}">`;
          if (dims.length > 1) {
              tensorHtml += `<div class="cube-icon" style="opacity:${Math.min(dims[0]/100, 1)}"></div>`;
          } else {
              tensorHtml += `<div class="bar-icon" style="width:${Math.min(dims[0]/10, 100)}%"></div>`;
          }
          tensorHtml += `<span>${b.shape}</span></div>`;
          htmlDesc = tensorHtml + htmlDesc;
      }

      d.innerHTML = htmlDesc;
      if (infoPanel) infoPanel.classList.remove('updating');
    }, 180);

    const meta = document.getElementById('blockMeta');
    const catChip = `<span class="chip" style="border-color:#${cat.color.toString(16)}"><b style="color:#${cat.color.toString(16)}">${cat.label}</b></span>`;
    const chips = (b.chips || []).map(c => `<span class="chip">${c}</span>`).join('');
    meta.innerHTML = catChip + chips;
  }

  function onDoubleClick(ev) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(blockMeshes, false);
    if (hits.length) explodeBlock(hits[0].object); else clearExplosion();
  }

  function clearExplosion() {
    if (explodedGroup) {
      modelGroup.remove(explodedGroup);
      disposeGroup(explodedGroup);
      explodedGroup = null;
      explodingMeshes = [];
      microMeshes = [];
      if (glowMesh) glowMesh.visible = false;
      if (selectedMesh) showInfo(selectedMesh.userData.block);
    }
  }

  function explodeBlock(mesh) {
    const b = mesh.userData.block;
    if (!b.details?.length) return;
    if (explodedGroup && explodedGroup.userData.blockId === b.id) { clearExplosion(); return; }
    clearExplosion();
    explodedGroup = new THREE.Group();
    explodedGroup.userData.blockId = b.id;
    modelGroup.add(explodedGroup);

    const basePos = new THREE.Vector3();
    mesh.getWorldPosition(basePos);
    modelGroup.worldToLocal(basePos);
    explodedGroup.position.copy(basePos);

    const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1.2 + b.details.length * 1.0, 0)]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0x4f8cff, transparent: true, opacity: 0.6 });
    explodedGroup.add(new THREE.Line(lineGeo, lineMat));

    b.details.forEach((det, i) => {
      const geo = new THREE.BoxGeometry(b.w * UNIT * 0.9, 0.28, D * 0.9);
      const c = det.color || ANATOMY_CATEGORIES[b.cat]?.color || 0xffffff;
      const mat = new THREE.MeshPhysicalMaterial({
        color: c, emissive: c, emissiveIntensity: 0.4, transparent: true, opacity: 0,
        roughness: 0.2, metalness: 0.1
      });
      const slice = new THREE.Mesh(geo, mat);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
      slice.add(edges);
      const lbl = makeLabel(det.title, 32, true);
      lbl.position.set(0, 0.5, 0); lbl.material.opacity = 0;
      slice.add(lbl);
      slice.position.set(0, 0, 0); 
      const targetY = 1.4 + i * 1.0;
      slice.userData = { microDetail: det, parentBlock: b };
      microMeshes.push(slice);
      explodedGroup.add(slice);
      explodingMeshes.push({ mesh: slice, edges, lbl, targetY, currentY: 0, vel: 0 });
    });
    showInfo(b);
  }

  function clearLights() { 
    blockMeshes.forEach(m => { 
        m.material.emissiveIntensity = m.userData.baseEmissive; 
        m.material.opacity = 0.9; 
    }); 
  }
  function lightBlock(blockId, on) {
    blockMeshes.forEach(m => { 
        if (m.userData.block.id === blockId) { 
            m.material.emissiveIntensity = on ? 1.4 : m.userData.baseEmissive; 
            m.material.opacity = 1.0; 
            // Оновлюємо glowMesh лише якщо не йде анімація прогону
            if (on && glowMesh && (!flowState || !flowState.active)) {
                const box = new THREE.Box3().setFromObject(m);
                const size = new THREE.Vector3(); box.getSize(size);
                const center = new THREE.Vector3(); box.getCenter(center);
                glowMesh.visible = true;
                glowMesh.scale.set(size.x * 1.4, size.y * 1.4, size.z * 1.4);
                glowMesh.position.copy(center);
                glowMesh.material.color.set(m.material.color);
            }
        } 
    });
  }

  function resetFlowState() {
    flowState = { active: false, lanes: [], holdFrames: 0 };
    const btn = document.getElementById('walkPlayBtn'); if (btn) btn.innerHTML = '▶ Прогін';
    const hud = document.getElementById('flowHud'); if (hud) hud.classList.remove('show');
  }

  function setHud(block) {
    const hud = document.getElementById('flowHud'); if (!hud) return;
    hud.innerHTML = `${block.title} · <b style="color:var(--accent-2)">${block.shape || ''}</b>`; hud.classList.add('show');
  }

  function startFlow() {
    if (!mainFlow.length) return;
    clearExplosion(); if (selectedMesh) resetMeshVisual(selectedMesh); selectedMesh = null;
    clearLights(); controls.autoRotate = false;
    const SPEED = 0.016;
    const lanes = [{ path: mainFlow, seg: 0, t: 0, done: false, speed: SPEED, pause: 0, primary: true }];
    if (gpsFlow.length) lanes.push({ path: gpsFlow, seg: 0, t: 0, done: false, speed: SPEED, pause: 0, primary: false });
    lanes.forEach(lane => {
      lightBlock(lane.path[0].id, true);
      if (lane.primary) { showInfo(lane.path[0].block); setHud(lane.path[0].block); }
    });
    flowState = { active: true, lanes, holdFrames: 0 };
    walkIndex = 0; updateWalkPanel();
    const btn = document.getElementById('walkPlayBtn'); if (btn) btn.innerHTML = '⏸ Пауза';
  }

  function stopFlow() {
    clearLights(); if (flowState) { flowState.active = false; flowState.lanes = []; }
    const btn = document.getElementById('walkPlayBtn'); if (btn) btn.innerHTML = '▶ Прогін';
    const hud = document.getElementById('flowHud'); if (hud) hud.classList.remove('show');
  }

  function updateFlows() {
    if (!flowState || !flowState.active) return;
    let allDone = true;
    flowState.lanes.forEach(lane => {
      const path = lane.path;
      if (!lane.done) {
        allDone = false;
        if (lane.pause > 0) { lane.pause--; } else {
          lane.t += lane.speed;
          if (lane.t >= 1) {
            lane.t = 0; lane.seg++;
            if (lane.seg >= path.length) { lane.done = true; return; }
            const node = path[lane.seg]; lightBlock(node.id, true);
            lane.pause = 24;
            if (lane.primary) { showInfo(node.block); setHud(node.block); }
          }
        }
      }
    });
    const primary = flowState.lanes.find(l => l.primary);
    if (primary) { walkIndex = Math.min(primary.seg, mainFlow.length - 1); updateWalkPanel(); }
    if (allDone && ++flowState.holdFrames > 90) stopFlow();
  }

  function buildLegend() {
    const el = document.getElementById('legend'); if (!el) return;
    el.innerHTML = Object.entries(ANATOMY_CATEGORIES).map(([id, c]) => `
      <div class="legend-row" data-cat="${id}">
        <span class="legend-dot" style="background:#${c.color.toString(16).padStart(6,'0')}"></span>
        ${c.label}
      </div>`).join('');
    el.querySelectorAll('.legend-row').forEach(row => {
      row.addEventListener('mouseenter', () => {
        const catId = row.dataset.cat;
        blockMeshes.forEach(m => {
          if (m.userData.block.cat === catId) {
              m.material.emissiveIntensity = 0.9;
              m.material.opacity = 1.0;
              m.scale.set(1.04, 1.04, 1.04);
          } else {
              m.material.opacity = 0.2;
          }
        });
      });
      row.addEventListener('mouseleave', () => {
        blockMeshes.forEach(m => {
          m.material.opacity = 0.9;
          resetMeshVisual(m);
        });
        if (selectedMesh) highlightMesh(selectedMesh, 0.75, 1.1);
      });
    });
  }

  function updateSummary(data) {
    if (selectedMesh) return;
    document.getElementById('blockTitle').textContent = data.title;
    document.getElementById('blockDesc').textContent = data.summary;
    document.getElementById('blockMeta').innerHTML = '<span class="chip">💡 Оберіть блок для огляду</span>';
  }

  function buildWalkSteps() {
    const wrap = document.getElementById('walkSteps'); if (!wrap) return;
    wrap.innerHTML = mainFlow.map((n, i) =>
      `<button class="walk-step-chip" data-i="${i}" title="${n.block.title}">
         <span class="wsc-num">${i + 1}</span><span class="wsc-name">${n.block.title}</span>
       </button>`
    ).join('');
    wrap.querySelectorAll('.walk-step-chip').forEach(btn => {
      btn.addEventListener('click', () => walkGoto(parseInt(btn.dataset.i, 10)));
    });
  }
  function resetWalk() { walkIndex = -1; updateWalkPanel(); }
  function updateWalkPanel() {
    const n = mainFlow.length;
    const label = document.getElementById('walkLabel'), bar = document.getElementById('walkBar');
    if (label) label.textContent = walkIndex < 0 ? `Крок 0 / ${n} — натисніть «Прогін»` : `Крок ${walkIndex + 1} / ${n} — ${mainFlow[walkIndex].block.title}`;
    if (bar) bar.style.width = (walkIndex < 0 ? 0 : ((walkIndex + 1) / n) * 100) + '%';
    const chips = document.querySelectorAll('.walk-step-chip');
    chips.forEach((c, i) => c.classList.toggle('active', i === walkIndex));
    if (walkIndex >= 0 && chips[walkIndex]) chips[walkIndex].scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' });
  }
  function walkGoto(i) {
    if (!mainFlow.length || !pulseMesh) return;
    if (flowState && flowState.active) stopFlow();
    clearExplosion();
    i = Math.max(0, Math.min(mainFlow.length - 1, i)); walkIndex = i;
    controls.autoRotate = false; clearLights();
    const node = mainFlow[i]; lightBlock(node.id, true);
    pulseMesh.visible = true; pulseMesh.position.copy(node.pos);
    if (gpsPulseMesh) gpsPulseMesh.visible = false;
    showInfo(node.block); setHud(node.block); updateWalkPanel();
  }
  function walkNext() { walkGoto(walkIndex < 0 ? 0 : walkIndex + 1); }
  function walkPrev() { walkGoto(walkIndex < 0 ? 0 : walkIndex - 1); }
  function walkToggle() { (flowState && flowState.active) ? stopFlow() : startFlow(); }

  function buildModal() {
    modalEl = document.createElement('div');
    modalEl.className = 'block-modal';
    modalEl.innerHTML = `<div class="block-modal-backdrop" data-close></div><div class="block-modal-card" role="dialog" aria-modal="true"><span class="block-modal-arrow"></span><button class="block-modal-close" data-close title="Закрити (Esc)">✕</button><div class="block-modal-body" id="blockModalBody"></div></div>`;
    host.appendChild(modalEl);
    modalEl.addEventListener('click', (e) => {
      if (e.target.closest('[data-close]')) { closeModal(); return; }
      const ex = e.target.closest('[data-explode]');
      if (ex) { const mesh = meshById(ex.dataset.explode); closeModal(); if (mesh) { select(mesh); explodeBlock(mesh); } }
    });
    let drag = null; const getCard = () => modalEl.querySelector('.block-modal-card');
    modalEl.addEventListener('pointerdown', (e) => {
      if (e.target.closest('[data-close]') || e.target.closest('[data-explode]')) return;
      const head = e.target.closest('.bm-head'); if (!head) return;
      const c = getCard(); if (!c) return;
      const arrow = modalEl.querySelector('.block-modal-arrow'); if (arrow) arrow.style.display = 'none';
      drag = { sx: e.clientX, sy: e.clientY, left: c.offsetLeft, top: c.offsetTop, rect: host.getBoundingClientRect(), w: c.offsetWidth, h: c.offsetHeight };
      head.style.cursor = 'grabbing'; e.preventDefault();
    });
    window.addEventListener('pointermove', (e) => {
      if (!drag) return; const c = getCard(); if (!c) return;
      let left = drag.left + (e.clientX - drag.sx); let top = drag.top + (e.clientY - drag.sy);
      left = Math.max(8, Math.min(left, drag.rect.width - drag.w - 8)); top = Math.max(8, Math.min(top, drag.rect.height - drag.h - 8));
      c.style.left = left + 'px'; c.style.top = top + 'px';
    });
    window.addEventListener('pointerup', () => { if (drag) { const h = modalEl.querySelector('.bm-head'); if (h) h.style.cursor = 'grab'; } drag = null; });
  }
  function meshById(id) { return blockMeshes.find(m => m.userData.block.id === id) || null; }
  function openBlockModal(b, ev) {
    if (!modalEl) return; const cat = ANATOMY_CATEGORIES[b.cat]; const hex = '#' + cat.color.toString(16).padStart(6, '0');
    const chips = (b.chips || []).map(c => `<span class="chip">${c}</span>`).join('');
    const micro = (b.details || []).map(d => { const dc = '#' + (d.color || cat.color).toString(16).padStart(6, '0'); return `<li><span class="micro-dot" style="background:${dc}"></span>${d.title}</li>`; }).join('');
    const body = document.getElementById('blockModalBody');
    body.innerHTML = `<div class="bm-head"><span class="bm-cat" style="border-color:${hex};color:${hex}">${cat.label}</span><h3>${b.title}</h3></div><p class="bm-lead">${b.desc || ''}</p>${b.purpose ? `<div class="bm-field"><div class="bm-label">🎯 Для чого</div><p>${b.purpose}</p></div>` : ''}${b.how ? `<div class="bm-field"><div class="bm-label">⚙️ Як працює</div><p>${b.how}</p></div>` : ''}<div class="bm-grid"><div class="bm-chip"><span class="k">Форма виходу</span><span class="v">${b.shape || '—'}</span></div><div class="bm-chip"><span class="k">Категорія</span><span class="v">${cat.label}</span></div></div>${chips ? `<div class="bm-tags">${chips}</div>` : ''}${micro ? `<div class="bm-field"><div class="bm-label">🔬 Мікроархітектура</div><ul class="bm-micro">${micro}</ul></div>` : ''}${(b.details && b.details.length) ? `<button class="bm-explode" data-explode="${b.id}">🧩 Показати у 3D</button>` : ''}`;
    modalEl.classList.add('open'); positionPopover(ev);
  }
  function positionPopover(ev) {
    const card = modalEl.querySelector('.block-modal-card'), arrow = modalEl.querySelector('.block-modal-arrow'); if (!card) return;
    const rect = host.getBoundingClientRect(); const px = ev ? ev.clientX - rect.left : rect.width / 2; const py = ev ? ev.clientY - rect.top : rect.height / 2;
    const cw = Math.min(330, rect.width - 24); card.style.width = cw + 'px'; card.style.left = '-9999px'; card.style.top = '0px';
    const ch = Math.min(card.offsetHeight, rect.height - 24); let side, left;
    if (px + 20 + cw <= rect.width - 8) { side = 'left'; left = px + 20; } else { side = 'right'; left = px - 20 - cw; }
    left = Math.max(8, Math.min(left, rect.width - cw - 8)); let top = Math.max(8, Math.min(py - 46, rect.height - ch - 8));
    card.style.left = left + 'px'; card.style.top = top + 'px';
    if (arrow) { const ay = Math.max(16, Math.min(py - top, ch - 16)); arrow.style.top = ay + 'px'; arrow.className = 'block-modal-arrow ' + side; }
  }
  function closeModal() { if (modalEl) modalEl.classList.remove('open'); }

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    
    if (glowMesh && glowMesh.visible) {
        glowMesh.material.opacity = 0.12 + Math.sin(Date.now() * 0.005) * 0.05;
    }

    if (explodingMeshes.length > 0) {
      explodingMeshes.forEach(item => {
        const stiffness = 0.18, damping = 0.72;
        const force = (item.targetY - item.currentY) * stiffness;
        item.vel = (item.vel + force) * damping;
        item.currentY += item.vel;
        item.mesh.position.y = item.currentY;
        if (item.mesh.material.opacity < 0.9) {
          item.mesh.material.opacity += 0.08; item.edges.material.opacity += 0.08; item.lbl.material.opacity += 0.08;
        }
      });
    }

    if (blockLabels.length) {
      const exId = explodedGroup ? explodedGroup.userData.blockId : null;
      blockLabels.forEach(it => {
        const target = (it.blockId === exId) ? -(H / 2) - 0.5 : it.baseY;
        it.label.position.y += (target - it.label.position.y) * 0.14;
      });
    }

    updateFlows();

    if (particles) {
        const positions = particles.geometry.attributes.position.array;
        for (let i = 1; i < positions.length; i += 3) {
            positions[i] -= 0.025;
            if (positions[i] < -15) positions[i] = 15;
        }
        particles.geometry.attributes.position.needsUpdate = true;
        particles.rotation.y += 0.0008;
    }

    renderer.render(scene, camera);
  }

  function autoFrame() {
    const box = new THREE.Box3().setFromObject(modelGroup);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const dist = Math.max(size.x, size.z) * 1.0 + size.y;
    const startPos = camera.position.clone(), startTarget = controls.target.clone();
    const targetPos = new THREE.Vector3(center.x + 2, center.y + size.y + 5, center.z + dist + 11);
    let t = 0;
    const step = () => {
        t += 0.04;
        if (t >= 1) { camera.position.copy(targetPos); controls.target.copy(center); return; }
        camera.position.lerpVectors(startPos, targetPos, t);
        controls.target.lerpVectors(startTarget, center, t);
        requestAnimationFrame(step);
    };
    step();
  }

  function onResize() {
    const W = host.clientWidth, Ht = host.clientHeight || 460;
    camera.aspect = W / Ht; camera.updateProjectionMatrix();
    renderer.setSize(W, Ht);
  }

  function enterFullscreen() {
    host.classList.add('fullscreen');
    document.body.classList.add('anatomy-fs');
    const b = document.getElementById('fullscreenBtn');
    if (b) { b.innerHTML = '✕'; b.title = 'Закрити'; }
    requestAnimationFrame(onResize);
  }
  function exitFullscreen() {
    if (!host.classList.contains('fullscreen')) return;
    host.classList.remove('fullscreen');
    document.body.classList.remove('anatomy-fs');
    const b = document.getElementById('fullscreenBtn');
    if (b) { b.innerHTML = '⛶'; b.title = 'На весь екран'; }
    stopFlow();
    requestAnimationFrame(onResize);
  }
  function hostInView() {
    if (host.classList.contains('fullscreen')) return true;
    const r = host.getBoundingClientRect();
    return r.top < window.innerHeight * 0.5 && r.bottom > window.innerHeight * 0.5;
  }
  function disposeGroup(g) {
    g.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); }
    });
  }
  function buildTabs() {
    const tabs = document.getElementById('anatomyTabs'); if(!tabs) return;
    const labels = { streetclip: 'StreetCLIP', baseline: 'Baseline CNN', geoclip: 'GeoCLIP' };
    tabs.innerHTML = Object.keys(labels).map(k =>
      `<button class="tab${k === currentKey ? ' active' : ''}" data-key="${k}">${labels[k]}</button>`
    ).join('');
    tabs.querySelectorAll('.tab').forEach(btn => {
      btn.addEventListener('click', () => {
        tabs.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active'); controls.autoRotate = true; buildModel(btn.dataset.key);
      });
    });
  }

  buildTabs(); buildLegend(); init();

  window.showAnatomy = function (key) {
    const btn = document.querySelector(`.tab[data-key="${key}"]`);
    if (btn) btn.click();
  };
})();
