import pandas as pd

# Wczytaj oba pliki
improved = pd.read_csv('dataset_true_false_improved.csv')
with_false = pd.read_csv('captions_flickr8k_with_false.csv')

print(f"Improved: {len(improved)} wierszy")
print(f"With false: {len(with_false)} wierszy")

# Sprawdź czy pliki mają tę samą długość
if len(improved) != len(with_false):
    print("UWAGA: Pliki mają różną liczbę wierszy!")
else:
    print("OK: Pliki mają tę samą liczbę wierszy")

# Połącz dane: image_path z with_false, caption i label z improved
result_df = pd.DataFrame({
    'image_path': with_false['image_path'],
    'caption': improved['caption'],
    'label': improved['is_true']
})

# Zapisz wynik
result_df.to_csv('captions_flickr8k_with_false_fixed.csv', index=False)

print(f"\nZapisano {len(result_df)} wierszy do captions_flickr8k_with_false_fixed.csv")
print("\nPorównanie pierwszych 5 wierszy:")
print("\n=== NOWY PLIK ===")
print(result_df.head())
print("\n=== IMPROVED (dla porównania) ===")
print(improved[['caption', 'is_true']].head())
