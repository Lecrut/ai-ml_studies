import pandas as pd
import re
import numpy as np

input_file = 'captions_flickr8k.csv'
output_file = 'captions_flickr8k_with_false.csv'

try:
    df = pd.read_csv(input_file)
    print("Plik wczytany pomyślnie.")
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku {input_file}. Prześlij go do plików po lewej stronie!")
    raise

df = df[df['image_path'].str.contains('.jpg', case=False, na=False)]
df['label'] = 1

swap_dict = {
    'dog': 'cat', 'cat': 'dog', 'dogs': 'cats', 'cats': 'dogs',
    'man': 'woman', 'woman': 'man', 'men': 'women', 'women': 'men',
    'boy': 'girl', 'girl': 'boy', 'child': 'adult', 'adult': 'child',
    'black': 'white', 'white': 'black', 'red': 'blue', 'blue': 'red',
    'green': 'yellow', 'yellow': 'green', 'grass': 'snow', 'snow': 'sand',
    'running': 'sleeping', 'sleeping': 'running', 'standing': 'sitting',
    'sunny': 'rainy', 'rainy': 'sunny', 'day': 'night', 'night': 'day',
    'car': 'bus', 'bus': 'truck', 'bike': 'motorcycle',
    'happy': 'sad', 'sad': 'happy', 'big': 'tiny', 'small': 'huge',
    'one': 'two', 'two': 'three', 'three': 'four'
}

def generate_false_caption(row, all_data):
    original_caption = str(row['caption'])
    current_image = row['image_path']
    words = original_caption.split()
    new_words = []
    replaced = False
    
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word).lower()
        if clean_word in swap_dict and not replaced:
            replacement = swap_dict[clean_word]
            if word[0].isupper(): replacement = replacement.capitalize()
            if not word[-1].isalnum(): replacement += word[len(clean_word):]
            new_words.append(replacement)
            replaced = True
        else:
            new_words.append(word)
    
    modified_caption = " ".join(new_words)
    
    if replaced and modified_caption != original_caption:
        return modified_caption
    
    while True:
        random_sample = all_data.sample(1).iloc[0]
        if random_sample['image_path'] != current_image:
            return random_sample['caption']

print("Generowanie fałszywych opisów...")
df_reference = df.copy()
negative_rows = []

for index, row in df.iterrows():
    false_caption = generate_false_caption(row, df_reference)
    negative_rows.append({
        'image_path': row['image_path'],
        'caption': false_caption,
        'label': 0
    })

df_final = pd.concat([df, pd.DataFrame(negative_rows)], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv(output_file, index=False)
print(f"GOTOWE! Pobierz plik: {output_file} (ok. {len(df_final)} wierszy)")