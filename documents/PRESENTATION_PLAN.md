# Plan Prezentacji Obrony Pracy Dyplomowej

**Temat:** Wykorzystanie filtru Kalmana i sieci LSTM w predykcji cen akcji

**Czas trwania:** 15 minut

---

## Slajd 1: Tytuł i powitanie (1 min)

- **Treść:** Tytuł pracy, imię i nazwisko autora, imię i nazwisko promotora, nazwa uczelni.
- **Wizualizacja:** Logotypy uczelni, profesjonalne tło powiązane z giełdą/AI.
- **Komentarz:** Krótkie przedstawienie się i nakreślenie tematyki.

## Slajd 2: Wprowadzenie i dziedzina problemowa (1.5 min)

- **Treść:** Charakterystyka rynków finansowych (szum, zmienność, nieliniowość). Dlaczego predykcja jest trudna?
- **Wizualizacja:** Wykres "surowych" cen akcji (np. AMZN) z widocznym szumem dziennym.
- **Komentarz:** Podkreślenie, że tradycyjne metody statystyczne często zawodzą w obliczu tak dużej dynamiki.

## Slajd 3: Cel pracy (1 min)

- **Treść:** Porównanie trzech podejść: modelu bazowego, modelu wzbogaconego o wskaźniki techniczne oraz modelu opartego na filtracji Kalmana.
- **Wizualizacja:** Diagram trzech ścieżek badawczych (Base vs. Enriched vs. Kalman).
- **Komentarz:** Nakreślenie hipotezy – czy filtracja danych/dodanie wskaźników poprawia stabilność predykcji?

## Slajd 4: Filtr Kalmana – Serce projektu (2 min)

- **Treść:** Definicja filtru jako estymatora stanu. Zastosowanie w projekcie: filtracja danych "niefiltrowanych" do postaci "czystszej".
- **Wizualizacja:** Wykres porównujący: Original Price vs Kalman Filtered Price.
- **Komentarz:** Wyjaśnienie, że filtr Kalmana pozwala nam "zobaczyć" trend pod powierzchnią dziennych fluktuacji.

## Slajd 5: Architektura sieci LSTM (1.5 min)

- **Treść:** Dlaczego LSTM? Pamięć długo- i krótkotrwała w kontekście szeregów czasowych.
- **Wizualizacja:** Schemat komórki LSTM (wejście, bramki: zapominania, wejściowa, wyjściowa).
- **Komentarz:** Wyjaśnienie zdolności modelu do wyłapywania zależności historycznych.

## Slajd 6: Metodologia i Narzędzia (1.5 min)

- **Treść:** Wykorzystanie frameworka Kedro (pipeline-y) oraz Optuna (automatyczne strojenie hiperparametrów).
- **Wizualizacja:** Graf pipeline-u z Kedro (Preprocessing -> Tuning -> Training -> Evaluation).
- **Komentarz:** Podkreślenie profesjonalnego podejścia do inżynierii danych i powtarzalności eksperymentów (150 prób Optuna dla każdego modelu).

## Slajd 7: Przygotowanie danych i cechy (1 min)

- **Treść:** Dane z Yahoo Finance (AMZN). Podział na zbiory. Wskaźniki techniczne (RSI, Bollinger Bands).
- **Wizualizacja:** Tabela z zestawieniem cech dla każdego z trzech modeli (Base=1, Enriched=4, Kalman=1).
- **Komentarz:** Wspomnienie o uniknięciu "wycieku danych" (data leakage) poprzez poprawne skalowanie po podziale.

## Slajd 8: Wyniki eksperymentalne – Metryki (2 min)

- **Treść:** Porównanie R^2, RMSE, MAE dla wszystkich modeli.
- **Wizualizacja:** Wykres słupkowy porównujący R^2 na zbiorze testowym.
- **Komentarz:** Omówienie, który model statystycznie poradził sobie najlepiej.

## Slajd 9: Analiza reaktywności i anomalii (1.5 min)

- **Treść:** Dlaczego model Enriched (mimo gorszych metryk) jest ciekawy? Reakcja na gwałtowne spadki/wzrosty.
- **Wizualizacja:** Wykres predykcji w momencie rynkowej anomalii (porównanie jak modele "nadążały" za zmianą).
- **Komentarz:** Interpretacja wyników – wybór modelu zależy od celu (dokładność matematyczna vs gotowość na zmiany trendu).

## Slajd 10: Wnioski (1.5 min)

- **Treść:** Potwierdzenie skuteczności filtru Kalmana. Wpływ strojenia hiperparametrów na wynik. Refleksja nad przydatnością wskaźników technicznych.
- **Wizualizacja:** Podsumowanie w punktach "Lessons Learned".
- **Komentarz:** Co można by zrobić dalej? (np. inne filtry, większa ilość spółek).

## Slajd 11: Zakończenie i pytania (0.5 min)

- **Treść:** "Dziękuję za uwagę. Czy mają Państwo jakieś pytania?"
- **Wizualizacja:** Slajd z danymi kontaktowymi lub tytułem pracy.
