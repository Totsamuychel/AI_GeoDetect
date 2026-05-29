import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import get_registry, ROOT, CACHE_DIR
from PIL import Image

reg = get_registry()
arch = "streetclip"
lm = reg._ensure_loaded(arch)

# Синтетичні «явно не вулиця» зображення
noise = Image.fromarray(np.random.randint(0,255,(400,400,3),dtype=np.uint8))
gray  = Image.fromarray(np.full((400,400,3), 128, dtype=np.uint8))
red   = Image.fromarray(np.dstack([np.full((400,400),200),np.zeros((400,400)),np.zeros((400,400))]).astype('uint8'))

tests = {"noise": noise, "gray": gray, "red": red}
for name, img in tests.items():
    r = reg.predict(arch, img)
    print(f"{name:8s} OOD={r['ood']['is_ood']} sim={r['ood']['max_similarity']}")

# Розподіл власних подібностей train (з кешу прототипів) — перерахуємо швидко
import pandas as pd
df = pd.read_csv(ROOT/"dataset/manifests/train.csv", low_memory=False)
protos = lm.prototypes
sims_all=[]
import torch
idx_of={c:i for i,c in enumerate(lm.class_names)}
for c in lm.class_names:
    sub=df[df["city"].astype(str).str.lower()==c.lower()].head(150)
    batch=[]
    for _,row in sub.iterrows():
        p=ROOT/"dataset"/str(row["filepath"])
        if not p.exists(): continue
        batch.append(lm.transform(Image.open(p).convert("RGB")))
        if len(batch)==16:
            t=torch.stack(batch).to(reg.device)
            with torch.no_grad(): emb=lm.model.encode_image(t)
            s=(emb@protos.T).cpu().numpy()
            sims_all.extend(s[np.arange(len(s)), idx_of[c]].tolist()); batch=[]
    if batch:
        t=torch.stack(batch).to(reg.device)
        with torch.no_grad(): emb=lm.model.encode_image(t)
        s=(emb@protos.T).cpu().numpy()
        sims_all.extend(s[np.arange(len(s)), idx_of[c]].tolist())
a=np.array(sims_all)
print("own-sim percentiles:", {p: round(float(np.percentile(a,p)),3) for p in [1,3,5,10,25,50]})
