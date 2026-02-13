import os
import random
import re
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
import gc

# ================= CONFIG =================

IMG_DIR = "datasets/flickr8k/Images"
OUTPUT_FILE = "captions_8_02_2026_big.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_BLIP = 16    
CHUNK_SIZE_MINING = 512 

# USTAWIENIA GENERATORA
LITERARY_PROB = 0.35   # 35% szans na trudne słownictwo (pod Zestaw 3)
SEMANTIC_MIN = 0.50    # Dolny próg podobieństwa dla negatywów
SEMANTIC_MAX = 0.88    # Górny próg (żeby nie brać parafraz)

# Słowa do usunięcia z początku zdania (artefakty BLIPa)
PREFIXES_TO_REMOVE = [
    "a detailed scene of", "a detailed photo of", "a close up of",
    "a picture of", "an image of", "arafed view of"
]

# ================= SŁOWNIKI (ZESTAW 3 READY) =================

# Rozszerzona mapa literacka - Klucz do zdania tych egzaminów
LITERARY_MAP = {
    "eating": ["consuming", "devouring", "ingesting", "partaking of"],
    "drinking": ["sipping", "imbibing"],
    "holding": ["clutching", "grasping", "clasping", "bearing"],
    "giving": ["dispensing", "bestowing", "handing", "providing"],
    "looking": ["gazing", "observing", "surveying", "examining", "peering"],
    "walking": ["strolling", "ambling", "traversing", "wandering"],
    "running": ["sprinting", "dashing", "hastening", "bolting"],
    "sitting": ["seated", "residing", "perched", "resting"],
    "standing": ["stationed", "positioned", "upright"],
    "while": ["whilst", "as", "during the time that"],
    "with": ["accompanied by", "alongside", "featuring"],
    "near": ["adjacent to", "in proximity to", "neighboring"],
    "on": ["upon", "atop"],
    "big": ["massive", "colossal", "enormous", "substantial"],
    "small": ["diminutive", "petite", "tiny", "minute"],
    "child": ["juvenile", "youngster", "adolescent"],
    "group": ["gathering", "assembly", "collection"]
}

LOGIC_CLUSTERS = [
    ["man","woman","boy","girl","child","adult","senior","lady","gentleman"],
    ["dog","cat","horse","cow","sheep","bear","bird","fox","wolf"],
    ["car","truck","bus","bike","boat","train","plane"],
    ["run","walk","sit","stand","sleep","jump","lie"],
    ["smile","cry","laugh","frown"],
    ["red","blue","green","yellow","black","white","orange","purple","pink","brown","grey"],
    ["grass","sand","snow","water","street","floor","field","beach"],
    ["sunny","rainy","cloudy","foggy"],
    ["one","two","three","four","five","six","ten","many"]
]

WORD_MAP = {}
for c in LOGIC_CLUSTERS:
    for w in c:
        WORD_MAP[w] = [x for x in c if x != w]

# ================= HELPERS =================

def clean(w):
    return re.sub(r"[^\w\s]", "", w).lower()

def fix_articles(text):
    words = text.split()
    vowels = tuple("aeiouAEIOU")
    for i in range(len(words)-1):
        if words[i].lower()=="a" and words[i+1].startswith(vowels): words[i]="an"
        elif words[i].lower()=="an" and not words[i+1].startswith(vowels): words[i]="a"
    return " ".join(words)

def clean_caption_output(caption):
    """Usuwa frazy startowe typu 'a detailed scene of'."""
    cap_lower = caption.lower()
    for prefix in PREFIXES_TO_REMOVE:
        if cap_lower.startswith(prefix):
            caption = caption[len(prefix):].strip()
            break
    if len(caption) > 0:
        caption = caption[0].upper() + caption[1:]
    return caption

def apply_literary_style(caption):
    """Wstrzykuje trudne słownictwo."""
    if random.random() > LITERARY_PROB: return None
    words = str(caption).split()
    new_words = []
    changed = False
    for word in words:
        clean_w = clean(word)
        if clean_w in LITERARY_MAP and random.random() < 0.7:
            fancy = random.choice(LITERARY_MAP[clean_w])
            # Proste zachowanie formy (ing -> ing)
            if word.endswith("ing") and not fancy.endswith("ing"): pass
            
            if word[0].isupper(): fancy = fancy.capitalize()
            if not word[-1].isalnum(): fancy += word[-1]
            new_words.append(fancy)
            changed = True
        else:
            new_words.append(word)
    return " ".join(new_words) if changed else None

def logic_negative(text):
    words = text.split()
    cands = [i for i,w in enumerate(words) if clean(w) in WORD_MAP]
    if not cands: return None
    i = random.choice(cands)
    rep = random.choice(WORD_MAP[clean(words[i])])
    if words[i][0].isupper(): rep = rep.capitalize()
    if not words[i][-1].isalnum(): rep += words[i][-1]
    words[i] = rep
    return fix_articles(" ".join(words))

# ================= EMBEDDING =================

def embed_captions(captions, model, tokenizer):
    all_emb = []
    for i in range(0, len(captions), 256):
        batch = captions[i:i+256]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        emb = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).expand(emb.size()).float()
        emb = torch.sum(emb * mask, 1) / torch.clamp(mask.sum(1), 1e-9)
        emb = F.normalize(emb, p=2, dim=1)
        all_emb.append(emb)
    return torch.cat(all_emb)

# ================= MAIN =================

def main():
    print(f"Device: {DEVICE}")

    # --- 1. BLIP GENERATION ---
    print("Loading BLIP Large...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)

    files = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg",".png",".jpeg"))]
    print(f"Found {len(files)} images.")

    dataset_positives = []
    
    print("Step 1: Generating Smart Captions...")
    for i in tqdm(range(0, len(files), BATCH_SIZE_BLIP)):
        batch_files = files[i:i+BATCH_SIZE_BLIP]
        batch_paths = [os.path.join(IMG_DIR, f) for f in batch_files]
        imgs = []
        valid_files = []
        for f, path in zip(batch_files, batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(img)
                valid_files.append(f)
            except: continue
        
        if not imgs: continue
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)

        # A. SHORT (Standard)
        out_short = blip.generate(**inputs, max_new_tokens=25, min_length=5)
        caps_short = processor.batch_decode(out_short, skip_special_tokens=True)

        # B. LONG (Enriched - wymuszamy detale)
        prompt = ["a detailed scene of"] * len(imgs)
        inputs_long = processor(images=imgs, text=prompt, return_tensors="pt").to(DEVICE)
        out_long = blip.generate(**inputs_long, max_new_tokens=70, min_length=20)
        caps_long = processor.batch_decode(out_long, skip_special_tokens=True)

        for f, s, l in zip(valid_files, caps_short, caps_long):
            rel_path = f"datasets/flickr8k/Images/{f}"
            l_clean = clean_caption_output(l)
            
            # Zbieramy 2-3 pozytywy na start
            dataset_positives.append({"image_path": rel_path, "caption": s, "label": 1})
            dataset_positives.append({"image_path": rel_path, "caption": l_clean, "label": 1})
            
            # Wariant literacki (pod Zestaw 3) - tylko z długiego opisu
            lit = apply_literary_style(l_clean)
            if lit:
                dataset_positives.append({"image_path": rel_path, "caption": lit, "label": 1})

    print(f"Generated {len(dataset_positives)} positive captions.")
    
    del blip, processor
    torch.cuda.empty_cache()
    gc.collect()

    # --- 2. MINING ---
    print("Loading BERT for Semantic Mining...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    bert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)
    bert.eval()

    unique_caps = list(set([d['caption'] for d in dataset_positives]))
    cap_to_idx = {c: i for i, c in enumerate(unique_caps)}
    embeddings = embed_captions(unique_caps, bert, tokenizer)

    final_dataset = []
    cap_idx_to_rows = {}
    for row in dataset_positives:
        c_idx = cap_to_idx[row['caption']]
        if c_idx not in cap_idx_to_rows: cap_idx_to_rows[c_idx] = []
        cap_idx_to_rows[c_idx].append(row)

    print("Step 2: Negative Mining...")
    num_embs = len(embeddings)
    
    # Mining w chunkach dla bezpieczeństwa pamięci
    for i in tqdm(range(0, num_embs, CHUNK_SIZE_MINING)):
        end = min(i + CHUNK_SIZE_MINING, num_embs)
        sim_chunk = torch.mm(embeddings[i:end], embeddings.T)
        
        for j in range(len(sim_chunk)):
            global_idx = i + j
            cap_orig = unique_caps[global_idx]
            sims = sim_chunk[j]
            associated_rows = cap_idx_to_rows.get(global_idx, [])
            
            for row in associated_rows:
                # 1. Dodajemy Pozytyw
                final_dataset.append(row)
                img = row["image_path"]

                # 2. Logic Negative (Detale: Kolor/Obiekt) - Zawsze 1
                ln = logic_negative(cap_orig)
                if ln and ln != cap_orig:
                    final_dataset.append({"image_path": img, "caption": ln, "label": 0})
                
                # 3. Dodatkowy Logic Negative (dla długich opisów)
                if len(cap_orig.split()) > 10:
                     ln2 = logic_negative(cap_orig) # Spróbuj wylosować inne słowo
                     if ln2 and ln2 != cap_orig and ln2 != ln:
                        final_dataset.append({"image_path": img, "caption": ln2, "label": 0})

                # 4. Semantic Negative (Kontekst) - 1 lub 2 sztuki
                valid_indices = torch.where((sims > SEMANTIC_MIN) & (sims < SEMANTIC_MAX))[0]
                if len(valid_indices) > 0:
                    perm = torch.randperm(len(valid_indices))[:2]
                    for cand_idx in valid_indices[perm]:
                        candidate_cap = unique_caps[cand_idx.item()]
                        if candidate_cap != cap_orig:
                             final_dataset.append({"image_path": img, "caption": candidate_cap, "label": 0})

    # --- ZAPIS ---
    df = pd.DataFrame(final_dataset)
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    print("="*30)
    print("FINAL DATASET STATS:")
    print(df["label"].value_counts())
    print(f"Total rows: {len(df)}")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()