import collections, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, "/home/user/MageZero/src/magezero")
from vocab import feature_id, initial_rows
W = "results/embeddings"
cards = json.load(open(f"{W}/cards.json")); E = np.load(f"{W}/card_emb.npy")
I = initial_rows(np.array([feature_id(c["name"]) for c in cards]), E.shape[1])
D = E - I
names = [c["name"] for c in cards]

decks = {}
for p in glob.glob("/home/user/mage/Mage.MageZero/decks/Pauper-*.txt") + glob.glob("/home/user/mage/Mage.MageZero/decks/Standard-Mono*.dck"):
    deck = os.path.basename(p).rsplit(".", 1)[0]
    for line in open(p, errors="ignore"):
        m = re.match(r"\s*(?:SB:\s*)?(\d+)\s+(?:\[[^\]]*\]\s*)?(.+?)\s*$", line)
        if m: decks.setdefault(m.group(2), set()).add(deck)
for c in cards:
    d = decks.get(c["name"], set())
    c["decks"] = sorted(d)
    c["deck"] = d.pop() if len(d) == 1 else ("shared" if d else "none")
    c["main"] = "CREATURE" if "CREATURE" in c["types"] else ("LAND" if "LAND" in c["types"] else c["types"][0])
    c["mvb"] = "land" if c["main"] == "LAND" else str(min(c["mv"], 5)) + ("+" if c["mv"] >= 5 else "")
    c["color1"] = c["colors"] if len(c["colors"]) == 1 else "multi"
print("deck counts:", collections.Counter(c["deck"] for c in cards).most_common())
drift = np.linalg.norm(D, axis=1) / np.linalg.norm(I, axis=1)
print(f"drift |E-I|/|I|: median {np.median(drift):.2f}, min {drift.min():.2f} ({names[drift.argmin()]}), max {drift.max():.2f} ({names[drift.argmax()]})")

def unit(X): return X / np.linalg.norm(X, axis=1, keepdims=True)
def knn_acc(X, labels, k=5, mask=None):
    X = unit(X); S = X @ X.T; np.fill_diagonal(S, -np.inf)
    labels = np.array(labels); idx = np.arange(len(labels)) if mask is None else np.flatnonzero(mask)
    hits = []
    for i in idx:
        nb = [j for j in np.argsort(-S[i]) if mask is None or mask[j]][:k]
        vals, cnt = np.unique(labels[nb], return_counts=True)
        hits.append(vals[np.argmax(cnt)] == labels[i])
    return np.mean(hits)
rng = np.random.default_rng(0)
print("\n5-NN leave-one-out accuracy (cosine). learned E / learned change E-I / initial I / majority class")
for attr in ("deck", "color1", "main", "mvb", "rarity"):
    lab = [c[attr] for c in cards]; maj = collections.Counter(lab).most_common(1)[0][1] / len(lab)
    print(f"  {attr:7s} {knn_acc(E, lab):.2f} / {knn_acc(D, lab):.2f} / {knn_acc(I, lab):.2f} / {maj:.2f}")
# within one deck's cards, does type still cluster? (removes the deck effect)
for attr in ("main", "mvb"):
    within = [knn_acc(D, [c[attr] for c in cards], k=3, mask=np.array([c["deck"] == d for c in cards]))
              for d in set(c["deck"] for c in cards) if sum(c["deck"] == d for c in cards) >= 8]
    print(f"  {attr} within single decks (3-NN, change E-I): mean {np.mean(within):.2f}")

print("\nnearest neighbours (cosine on learned change E-I):")
U = unit(D); S = U @ U.T
for n in ("Faerie Miscreant", "Faerie Seer", "Counterspell", "Spellstutter Sprite", "Ninja of the Deep Hours", "Island", "Lightning Bolt",
          "Galvanic Blast", "Llanowar Elves", "Journey to Nowhere", "Thoughtcast", "Myr Enforcer", "Shock"):
    if n in names:
        i = names.index(n); nb = np.argsort(-S[i])[1:6]
        print(f"  {n:24s} -> " + ", ".join(f"{names[j]} ({S[i, j]:.2f})" for j in nb))
json.dump(cards, open(f"{W}/cards_meta.json", "w"))
np.save(f"{W}/delta.npy", D); np.save(f"{W}/drift.npy", drift)
