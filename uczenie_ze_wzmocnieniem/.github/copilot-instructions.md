Prosty model środowiska do testowania algorytmów:
- SARSA,
- SARSA(lambda), 
- Q-learning, 
- Expected SARSA, 
- Deep Q-learning

Implementacja w Pythonie
Wszystko w plikach .py i folderze /myEnv_lab_4
wzoruj się na plikach z /env/
Wyświetlacz: tekstowy lub graficzny (np. Pygame)

Środowisko ma:. 
- Wyznaczamy losowy labirynt 
- Labirynt musi mieć 4 wejścia/wyjścia (np. w rogach)
- W rozgrywce biorą udział 2 agenty
- Na początku rozlosowujemy 2 wejścia jako start dla agentów, a trzecie jako cel z nagrodą
- Agenci poruszają się na zmianę
- Po spotkaniu agentów w tym samym polu, obaj są przenoszeni na swoje pola startowe i otrzymują karę
- Za dotarcie do celu agent otrzymuje nagrodę i jest koniec epizodu
- Za każdy ruch agent otrzymuje małą karę
- plansza ma mieć rozmiar 10x10
- labirynt ma mieć około 4 ścieżek od startu do celu