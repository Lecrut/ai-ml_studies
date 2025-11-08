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
- Wyznaczamy labirynt w zmiennej 2D (np. lista list)
- Labirynt zawiera ściany i ścieżki które są ze sobą połączone
- Labirynt musi mieć 4 wejścia/wyjścia (np. w rogach)
- W rozgrywce biorą udział minimum 2 agenty
- Na początku rozlosowujemy wejścia jako start dla agentów, a ostatnie jako cel z nagrodą
- Agenci poruszają się na zmianę
- Po spotkaniu agentów w tym samym polu, obaj są przenoszeni na swoje pola startowe i otrzymują karę
- Za dotarcie do celu agent otrzymuje nagrodę i jest koniec epizodu
- Za każdy ruch agent otrzymuje małą karę
- plansza ma mieć rozmiar 10x10
- labirynt ma mieć około 4 ścieżek od startu do celu