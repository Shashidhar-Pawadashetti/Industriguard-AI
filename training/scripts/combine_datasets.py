import shutil, yaml
from pathlib import Path
from collections import Counter

BASE = Path(r"C:\industriguard_training")

REMAP = {
    # Worker-Safety
    "helmet": 0,  "no-helmet": 1,  "vest": 2,  "no-vest": 3,
    # bangga
    "Helmet": 0,  "Without Helmet": 1,  "Vest": 2,  "Without Vest": 3,
    # Snehil Sanyal
    "Hardhat": 0,  "NO-Hardhat": 1,  "Safety Vest": 2,  "NO-Safety Vest": 3,
    # Mendeley
    "NoHelmet": 1,  "NoVest": 3,
    # person, mask, etc → None = drop (not listed = auto-dropped)
}

def load_class_names(yaml_path):
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    names = d.get("names", {})
    return [names[i] for i in sorted(names.keys())] if isinstance(names, dict) else names

def remap_label(src_lbl, dst_lbl, class_names):
    lines_out = []
    for line in open(src_lbl):
        parts = line.strip().split()
        if not parts:
            continue
        cid = int(parts[0])
        if cid >= len(class_names):
            continue
        tgt = REMAP.get(class_names[cid])
        if tgt is not None:
            lines_out.append(f"{tgt} " + " ".join(parts[1:]))
    if lines_out:
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("\n".join(lines_out))
        return True
    return False

def process(raw_dir, splits=None):
    raw_dir = Path(raw_dir)
    yamls = list(raw_dir.rglob("data.yaml"))
    if not yamls:
        print(f"  ⚠ No yaml in {raw_dir.name}"); return 0, 0
    names = load_class_names(yamls[0])
    print(f"  Classes: {names}")

    splits = splits or {"train": "train", "valid": "val"}
    kept = dropped = 0

    for src_split, dst_split in splits.items():
        for img_dir in [raw_dir/src_split/"images",
                        raw_dir/"images"/src_split]:
            if img_dir.exists():
                break
        else:
            continue
        lbl_dir = img_dir.parent.parent / "labels" / img_dir.parent.name \
                  if "images" == img_dir.parent.name else img_dir.parent / "labels"
        # Try sibling labels folder
        lbl_dir2 = img_dir.parent.parent / (img_dir.parent.parent.name + "/../labels") / src_split
        for ld in [img_dir.parent.parent/"labels"/src_split,
                   raw_dir/src_split/"labels"]:
            if ld.exists():
                lbl_dir = ld; break

        dst_i = BASE/"combined"/dst_split/"images"
        dst_l = BASE/"combined"/dst_split/"labels"
        dst_i.mkdir(parents=True, exist_ok=True)
        dst_l.mkdir(parents=True, exist_ok=True)

        for img in img_dir.glob("*.*"):
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            stem = f"{raw_dir.name}_{img.stem}"
            ok = remap_label(lbl, dst_l/(stem+".txt"), names)
            if ok:
                shutil.copy2(img, dst_i/(stem+img.suffix))
                kept += 1
            else:
                dropped += 1
    return kept, dropped

# Clear and rebuild
if (BASE/"combined").exists():
    shutil.rmtree(BASE/"combined")

for name, path in [
    ("Worker-Safety",  BASE/"raw"/"worker_safety"),
    ("bangga PPE-2",   BASE/"raw"/"bangga"),
    ("Snehil Sanyal",  BASE/"raw"/"snehil"),
    ("Mendeley PPE",   BASE/"raw"/"mendeley"),
]:
    print(f"\n🔄 {name}...")
    k, d = process(path)
    print(f"   kept={k}, dropped={d}")

# Stats
names_map = {0:"helmet", 1:"no_helmet", 2:"safety_vest", 3:"no_vest"}
print("\n📊 Final distribution:")
for split in ["train", "val"]:
    c = Counter()
    for f in (BASE/"combined"/split/"labels").glob("*.txt"):
        for ln in open(f):
            p = ln.strip().split()
            if p: c[int(p[0])] += 1
    print(f"\n  {split}:")
    total = sum(c.values())
    for cid in range(4):
        n = c.get(cid, 0)
        bar = "█" * int(30 * n / max(total,1))
        print(f"    {names_map[cid]:15s} {n:6d}  {bar}")

# Write yaml
cfg = {
    "path": str(BASE/"combined").replace("\\","/"),
    "train": "train/images",
    "val":   "val/images",
    "nc": 4,
    "names": {0:"helmet", 1:"no_helmet", 2:"safety_vest", 3:"no_vest"}
}
with open(BASE/"data.yaml","w") as f:
    yaml.dump(cfg, f)
print(f"\n✅ data.yaml written to {BASE/'data.yaml'}")