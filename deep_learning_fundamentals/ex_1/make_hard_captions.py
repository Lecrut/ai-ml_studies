import pandas as pd
import random
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

# ================= CONFIG =================

INPUT_FILE = 'captions_217k_ultra_dataset.csv'
OUTPUT_FILE = 'captions_6_02_2026_fixed.csv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64

# SEMANTIC MINING (BERT)
# 0.78 - 0.95 to "Sweet Spot" - bardzo podobne, ale nie identyczne
SEMANTIC_MIN = 0.78
SEMANTIC_MAX = 0.95

# NOWY BALANS (Według Twojego życzenia)
TARGET_RATIOS = {
    "semantic": 0.50,   # 50% - Trudne pary z BERTa (Core task)
    "attribute": 0.30,  # 30% - Kolory i Liczby (Precision)
    "logic": 0.20       # 20% - Syntetyczne zamiany (Logic Swaps)
}

# ================= MASSIVE VOCABULARY =================

# 1. Liczby (Rozszerzone)
NUMBERS_TXT = [
    'one','two','three','four','five','six','seven','eight','nine','ten',
    'eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty',
    'thirty', 'forty', 'fifty', 'sixty', 'hundred', 'thousand',
    'single', 'pair', 'couple', 'few', 'many', 'several'
]
NUMBERS_DIGIT = [str(i) for i in range(0, 101)] # 0 do 100

# 2. Kolory (Zniuansowane)
COLORS_MAP = {
    'red': ['blue','green','black','white','yellow','crimson','maroon'],
    'blue': ['red','yellow','green','orange','purple','azure','navy','teal'],
    'green': ['red','blue','yellow','brown','olive','emerald','lime'],
    'yellow': ['purple','blue','red','orange','amber','gold'],
    'black': ['white','grey','red','blue','charcoal','obsidian'],
    'white': ['black','cream','grey','ivory','beige'],
    'brown': ['grey','black','tan','beige','chocolate'],
    'grey': ['brown','white','black','silver','slate'],
    'orange': ['purple','green','blue','red','peach'],
    'pink': ['green','blue','red','purple','magenta','rose'],
    'purple': ['yellow','green','orange','blue','violet','lavender'],
    'golden': ['silver', 'bronze', 'yellow'],
    'silver': ['golden', 'bronze', 'grey']
}

# 3. Klastry Logiczne (Strict Logic Clusters)
# Zamieniamy słowa TYLKO wewnątrz tych list, żeby zachować sens.
LOGIC_CLUSTERS = [
    # --- LUDZIE ---
    ["man", "woman", "guy", "lady", "gentleman", "person"],
    ["boy", "girl", "child", "kid", "toddler"],
    ["baby", "infant", "newborn"],
    ["adult", "teenager", "senior", "elder"],
    ["crowd", "group", "team", "couple", "family"],
    ["police", "soldier", "worker", "doctor", "chef", "player", "athlete"],

    # --- ZWIERZĘTA ---
    ["dog", "cat", "puppy", "kitten"],
    ["horse", "cow", "sheep", "goat", "donkey", "bull"],
    ["lion", "tiger", "bear", "wolf", "fox", "leopard"],
    ["bird", "eagle", "hawk", "pigeon", "duck", "swan", "goose", "parrot"],
    ["fish", "shark", "dolphin", "whale"],
    ["snake", "lizard", "frog", "turtle"],

    # --- TRANSPORT ---
    ["car", "truck", "bus", "van", "taxi", "jeep", "ambulance"],
    ["bike", "bicycle", "motorcycle", "scooter", "moped"],
    ["boat", "ship", "yacht", "ferry", "canoe", "kayak", "raft"],
    ["plane", "helicopter", "jet", "glider"],
    ["train", "subway", "tram", "metro"],

    # --- DOM / MEBLE ---
    ["chair", "couch", "sofa", "bench", "stool", "seat"],
    ["table", "desk", "counter", "shelf"],
    ["bed", "mattress", "cot", "hammock"],
    ["door", "window", "gate", "arch"],
    ["floor", "ceiling", "wall", "roof"],

    # --- KUCHNIA / JEDZENIE ---
    ["cup", "mug", "glass", "bottle", "jar"],
    ["plate", "bowl", "dish", "tray", "pan"],
    ["fork", "spoon", "knife"],
    ["pizza", "burger", "sandwich", "hotdog", "taco", "burrito"],
    ["apple", "banana", "orange", "lemon", "fruit", "berry"],
    ["bread", "cake", "pie", "cookie", "muffin"],
    ["beer", "wine", "soda", "water", "juice", "coffee", "tea"],

    # --- SPORT / ZABAWA ---
    ["ball", "frisbee", "balloon", "kite"],
    ["racket", "bat", "stick", "club"],
    ["skateboard", "surfboard", "snowboard", "skis"],
    ["helmet", "cap", "hat", "mask"],

    # --- UBRANIA ---
    ["shirt", "t-shirt", "jacket", "coat", "sweater", "hoodie", "vest"],
    ["pants", "jeans", "shorts", "trousers", "leggings"],
    ["dress", "skirt", "gown", "robe"],
    ["shoes", "boots", "sneakers", "sandals", "heels"],
    ["bag", "backpack", "purse", "suitcase"],

    # --- MIEJSCA ---
    ["street", "road", "highway", "path", "trail", "sidewalk"],
    ["beach", "desert", "field", "meadow", "plain"],
    ["forest", "woods", "jungle", "park", "garden"],
    ["mountain", "hill", "cliff", "rock"],
    ["ocean", "river", "lake", "pond", "pool", "sea"],
    ["building", "house", "shop", "store", "office", "school"],

    # --- AKCJE (RUCH) ---
    ["running", "walking", "jogging", "sprinting", "marching", "strolling"],
    ["jumping", "hopping", "leaping", "skipping"],
    ["climbing", "hiking", "trekking"],
    ["swimming", "diving", "floating", "surfing"],
    ["riding", "driving", "cycling"],
    ["throwing", "catching", "kicking", "hitting"],

    # --- AKCJE (STATYCZNE) ---
    ["sitting", "standing", "lying", "resting", "kneeling", "crouching"],
    ["sleeping", "napping", "dozing"],
    ["waiting", "staying", "stopping"],
    ["watching", "looking", "staring", "glancing", "observing"],
    ["smiling", "laughing", "grinning"],
    ["crying", "weeping", "frowning"],
    ["eating", "drinking", "tasting"],
    ["talking", "speaking", "shouting", "whispering", "singing"]
]

# Mapowanie dla szybkiego dostępu
WORD_TO_CLUSTER = {}
for cluster in LOGIC_CLUSTERS:
    for word in cluster:
        WORD_TO_CLUSTER[word] = [w for w in cluster if w != word]

# ================= HELPERS =================

def fix_articles(text):
    words = text.split()
    vowels = tuple("aeiouAEIOU")
    for i in range(len(words)-1):
        a = words[i].lower()
        b = words[i+1]
        if a == "a" and b.startswith(vowels): words[i] = "an"
        elif a == "an" and not b.startswith(vowels): words[i] = "a"
    return " ".join(words)

def clean_word(w):
    return re.sub(r'[^\w\s]', '', w).lower()

def generate_logic_negative(caption):
    words = str(caption).split()
    candidates = [] 
    
    for i, word in enumerate(words):
        w_clean = clean_word(word)
        if w_clean in WORD_TO_CLUSTER:
            candidates.append((i, WORD_TO_CLUSTER[w_clean]))
    
    if not candidates:
        return None

    idx, options = random.choice(candidates)
    replacement = random.choice(options)
    
    old_word = words[idx]
    new_word = replacement
    
    if old_word and old_word[0].isupper(): new_word = new_word.capitalize()
    if old_word and not old_word[-1].isalnum(): new_word += old_word[-1]
        
    words[idx] = new_word
    return fix_articles(" ".join(words))

def get_attribute_negative(caption):
    words = str(caption).split()
    changed = False
    out = []
    for word in words:
        clean = clean_word(word)
        replacement = word
        
        if clean in COLORS_MAP:
            replacement = random.choice(COLORS_MAP[clean])
            changed = True
        elif clean in NUMBERS_TXT:
            # Losujemy z tej samej kategorii liczbowej
            cands = [n for n in NUMBERS_TXT if n != clean]
            replacement = random.choice(cands)
            changed = True
        elif clean in NUMBERS_DIGIT:
            cands = [n for n in NUMBERS_DIGIT if n != clean]
            replacement = random.choice(cands)
            changed = True
            
        if word and word[0].isupper(): replacement = replacement.capitalize()
        if word and not word[-1].isalnum(): replacement += word[-1]
        out.append(replacement)

    if changed: return fix_articles(" ".join(out))
    return None

def mean_pooling(output, mask):
    emb = output.last_hidden_state
    mask = mask.unsqueeze(-1).expand(emb.size()).float()
    return torch.sum(emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

# ================= MAIN =================

def main():
    print("Device:", DEVICE)

    # 1. Wczytanie
    df = pd.read_csv(INPUT_FILE)
    positives = df[df['label'] == 1].reset_index(drop=True)
    
    captions = positives['caption'].astype(str).tolist()
    image_paths = positives['image_path'].tolist()
    N = len(positives)
    print("Positives:", N)

    # 2. Przydział Strategii (30/50/20)
    p_attr = TARGET_RATIOS['attribute']
    p_sem = TARGET_RATIOS['semantic']
    p_logic = TARGET_RATIOS['logic']
    
    # Normalizacja gdyby suma nie dawała 1.0 (dla bezpieczeństwa)
    total_p = p_attr + p_sem + p_logic
    probs = [p_attr/total_p, p_sem/total_p, p_logic/total_p]
    
    strategies = np.random.choice(['attribute', 'semantic', 'logic'], size=N, p=probs)

    # 3. BERT
    print("Embedding captions...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(DEVICE)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, N, BATCH_SIZE)):
        batch = captions[i:i+BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        e = mean_pooling(out, enc['attention_mask'])
        e = F.normalize(e, dim=1)
        all_embeddings.append(e)
    all_embeddings = torch.cat(all_embeddings)

    # 4. Mining & Generation
    print(f"Mining Negatives (Semantic Range: {SEMANTIC_MIN} - {SEMANTIC_MAX})...")
    
    negatives = []
    stats = {'attribute': 0, 'semantic': 0, 'logic': 0, 'fallback_soft': 0}

    for i in tqdm(range(0, N, BATCH_SIZE)):
        end = min(i+BATCH_SIZE, N)
        
        batch_emb = all_embeddings[i:end]
        sim = torch.mm(batch_emb, all_embeddings.T)
        top_vals, top_inds = torch.topk(sim, k=50, dim=1)

        for j in range(len(batch_emb)):
            global_idx = i + j
            orig_img = image_paths[global_idx]
            orig_cap = captions[global_idx]
            strategy = strategies[global_idx]
            
            final_neg = None
            final_type = 'pending'

            # --- 1. ATTRIBUTE STRATEGY ---
            if strategy == 'attribute':
                res = get_attribute_negative(orig_cap)
                if res:
                    final_neg = res
                    final_type = 'attribute'
                else:
                    # Jak nie ma atrybutów, próbujemy LOGIC
                    strategy = 'logic'

            # --- 2. LOGIC STRATEGY ---
            if not final_neg and strategy == 'logic':
                res = generate_logic_negative(orig_cap)
                if res:
                    final_neg = res
                    final_type = 'logic'
                else:
                    # Jak nie ma słów do zamiany, próbujemy SEMANTIC
                    strategy = 'semantic'

            # --- 3. SEMANTIC STRATEGY ---
            if not final_neg: # Semantic albo fallback z powyższych
                scores = top_vals[j]
                inds = top_inds[j]
                
                # A. Strict Semantic (Idealny kandydat)
                for s, idx in zip(scores, inds):
                    idx = idx.item()
                    s = s.item()
                    if image_paths[idx] != orig_img and SEMANTIC_MIN <= s <= SEMANTIC_MAX:
                        final_neg = captions[idx]
                        final_type = 'semantic'
                        break
                
                # B. Soft Fallback (Jeśli nie znaleziono idealnego)
                # Bierzemy najlepsze dopasowanie, które nie jest tym samym obrazkiem
                if not final_neg:
                    for s, idx in zip(scores, inds):
                        idx = idx.item()
                        if image_paths[idx] != orig_img:
                            final_neg = captions[idx]
                            final_type = 'fallback_soft'
                            break
            
            negatives.append({
                "image_path": orig_img,
                "caption": final_neg,
                "label": 0
            })
            stats[final_type] += 1

    # 5. Zapis
    df_neg = pd.DataFrame(negatives)
    df_final = pd.concat([
        positives[['image_path','caption','label']],
        df_neg[['image_path','caption','label']]
    ])
    df_final = df_final.sample(frac=1).reset_index(drop=True)

    print("\nFinal Stats:")
    print(f"Positives: {N}")
    print(f"Attribute: {stats['attribute']} ({stats['attribute']/N:.1%})")
    print(f"Semantic:  {stats['semantic']} ({stats['semantic']/N:.1%})")
    print(f"Logic:     {stats['logic']} ({stats['logic']/N:.1%})")
    print(f"Fallback:  {stats['fallback_soft']} ({stats['fallback_soft']/N:.1%})")
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("Saved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()