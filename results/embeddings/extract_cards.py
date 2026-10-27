"""Match the v4 model's leaf vocab to XMage card names and collect each card's metadata."""
import collections, glob, json, re, sys
import numpy as np, torch
sys.path.insert(0, "/home/user/MageZero/src/magezero")
from vocab import feature_id
from model import load_checkpoint

model, vocab, edge_vocab = load_checkpoint("/home/user/MageZero/experiment_models/Pauper-Faeries/ver4/model.pt.gz")
ids = np.asarray(vocab.ids)
row_of = {int(i): r for r, i in enumerate(ids)}
print("leaf vocab rows:", len(ids))

# every card name XMage knows: name -> (class, rarities seen)
info = {}
for p in glob.glob("/home/user/mage/Mage.Sets/src/mage/sets/*.java"):
    for name, rarity, cls in re.findall(r'new SetCardInfo\("((?:[^"\\]|\\.)*)",\s*"?[^,]*"?,\s*Rarity\.(\w+),\s*([\w.]+)\.class', open(p, errors="ignore").read()):
        e = info.setdefault(name.replace('\\"', '"'), [cls, set()])
        e[1].add(rarity)
print("XMage card names:", len(info))
matched = {n: v for n, v in info.items() if feature_id(n) in row_of}
print("card names in the vocab:", len(matched))

def parse(cls):
    if "basiclands" in cls:  # basic lands have their own base class
        return {"types": ["LAND"], "cost": "", "mv": 0, "colors": "L", "subtypes": [], "pt": None, "keywords": []}
    path = "/home/user/mage/Mage.Sets/src/" + cls.replace(".", "/") + ".java"
    try:
        src = open(path, errors="ignore").read()
    except OSError:
        return None
    m = re.search(r"super\(\s*ownerId\s*,\s*setInfo\s*,\s*new CardType\[\]\s*\{([^}]*)\}\s*,\s*(?:\"([^\"]*)\"|null|\"\")", src)
    if not m:
        if "basiclands" in cls:  # basic lands have their own base class
            return {"types": ["LAND"], "cost": "", "mv": 0, "colors": "L", "subtypes": [], "pt": None, "keywords": []}
        return None
    types = [t.strip().replace("CardType.", "") for t in m.group(1).split(",") if t.strip()]
    cost = m.group(2) or ""
    sub = re.findall(r"SubType\.(\w+)", src.split("public " + cls.split(".")[-1] + "(final")[0] if False else src[:src.find("this.addAbility") if "this.addAbility" in src else len(src)])
    pt = re.search(r"this\.power = new MageInt\((\d+)\);\s*this\.toughness = new MageInt\((\d+)\)", src)
    colors = "".join(c for c in "WUBRG" if "{" + c + "}" in cost or re.search(r"\{[^}]*" + c + r"[^}]*\}", cost))
    mv = sum(int(x) for x in re.findall(r"\{(\d+)\}", cost)) + len(re.findall(r"\{[WUBRGC](?:/[WUBRGP])?\}", cost))
    abilities = re.findall(r"this\.addAbility\((?:new )?(\w+)", src)
    return {"types": types, "cost": cost, "mv": mv, "colors": colors or ("C" if "LAND" not in types else "L"),
            "subtypes": sorted(set(sub)), "pt": [int(pt.group(1)), int(pt.group(2))] if pt else None,
            "keywords": sorted({a.replace("Ability", "") for a in abilities if a.endswith("Ability")})}

cards = []
for name, (cls, rarities) in sorted(matched.items()):
    meta = parse(cls)
    if meta is None:
        print("unparsed:", name, cls); continue
    order = ["COMMON", "UNCOMMON", "RARE", "MYTHIC", "SPECIAL", "BONUS", "LAND"]
    meta.update(name=name, row=row_of[feature_id(name)], rarity=min(rarities, key=lambda r: order.index(r) if r in order else 9))
    cards.append(meta)
emb = model.embedding.weight.detach().numpy()
np.save("results/embeddings/card_emb.npy", emb[[c["row"] for c in cards]])
json.dump(cards, open("results/embeddings/cards.json", "w"), indent=0)
print("cards with metadata:", len(cards))
print("types:", collections.Counter(t for c in cards for t in c["types"]).most_common())
print("colors:", collections.Counter(c["colors"] for c in cards).most_common(12))
print("rarity:", collections.Counter(c["rarity"] for c in cards).most_common())
print("sample:", [(c["name"], c["types"], c["cost"], c["colors"], c["rarity"]) for c in cards[:6]])
