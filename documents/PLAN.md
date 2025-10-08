# Tytuł pracy inżynierskiej

_Zastosowanie filtrów Kalmana do poprawy predykcji cen giełdowych_

# Główny cel pracy

Zbadanie czy zastosowanie filtru Kalmana na zbiorze cen giełdowych spółki AMZN poprawi przewidzenie dokładnej wartości CLOSE w oparciu o sieci **LSTM**.

# Co zrobiłem dotychczas?

1. Przygotowałem oryginalny zbiór danych

   - Pobrałem przy pomocy yfinance
   - Przeanalizowałem pod kątem zawartości i poprawności
   - Przygotowałem algorytm do konwersji zbioru do formatu, który będzie kompatybilny z sieciami neuronowymi na bazie warstw **LSTM**
   - Podzieliłem zbiór danych na zbiór treningowy, walidacyjny i testowy
   - Otrzymany zbiór danych ustandaryzowałem przy użyciu opakowania Dataset z biblioteki PyTorch

2. Stworzyłem model bazowy

   - Przy użyciu biblioteki PyTorch przygotowałem architekturę modelu bazowego
   - Wytrenowałem model uwzględniając walidację pod koniec każdej epoki
   - Przetestowałem model na danych wcześniej niewidocznych dla modelu
   - Model podczas treningu miał za zadanie minimalizować funkcję straty MSE (z powodu natury regresyjnej problemu)
   - Przygotowałem wykresy:
     - Cena realna vs przewidziana
     - Dystrybucja błędu
     - Funkcja błędu poprzez epoki podczas treningu
     - Prezentujące cały zbiór danych

3. Przygotowałem wzbogacony zbiór danych

   - Przerobiłem oryginalny zbiór danych tak, aby zawierał dodatkowe wskaźniki techniczne:
     - **RSI**
     - **Bandwidth** (Bollinger Bands)
     - **%B** (Bollinger Bands)
   - Przygotowałem algorytm do konwersji zbioru do formatu, który będzie kompatybilny z sieciami neuronowymi na bazie warstw **LSTM**
   - Podzieliłem zbiór danych na zbiór treningowy, walidacyjny i testowy
   - Otrzymany zbiór danych ustandaryzowałem przy użyciu opakowania Dataset z biblioteki PyTorch

4. Stworzyłem model wzbogacony

   - Przy użyciu biblioteki PyTorch przygotowałem architekturę modelu wzbogaconego
   - Wytrenowałem model uwzględniając walidację pod koniec każdej epoki
   - Przetestowałem model na danych wcześniej niewidocznych dla modelu
   - Model podczas treningu miał za zadanie minimalizować funkcję straty MSE (z powodu natury regresyjnej problemu)
   - Przygotowałem wykresy:
     - Cena realna vs przewidziana
     - Dystrybucja błędu
     - Funkcja błędu poprzez epoki podczas treningu
     - Prezentujące cały zbiór danych

# Co jest jeszcze do zrobienia?

1. Przygotowanie zbioru danych przefiltrowanego przy użyciu filtru Kalmana
2. Stworzenie modelu w oparciu o zbiór przefiltrowany przez filtr Kalmana
3. Dla każdego utworzonego modelu znaleźć jak najlepsze hiperparametry (**_Optuna_**)

# Jakie problemy napotkałem po drodze? Jak je rozwiązałem?

1. Przy pierwszym podejściu tworzenia modelu bazowego, model przewidywał stałą wartość (był niestabilny). Po głębszej analizie algorytmu uczenia **_Gradient descent_** zauważyłem, że przy aktualizacji wag model uwzględnia wartość wejścia, która w przypadku cen giełdowych potrafiła być ogromna, prowadząc do **_Eksplozji gradientu_**. Problem rozwiązałem normalizując do zakresu \[0,1\] zbiór danych przy użyciu MinMaxScaler z biblioteki scikit-learn.

# Co chciałbym uwzględnić w artykule naukowym?

1. Proces postępu pracy inżynierskiej:
   - Co zrobiłem i jak zrobiłem
2. Koncepcje architektoniczne lub matematyczne stojące za zagadnieniami takimi jak:
   - Warstwa LSTM
   - Jak uczy się sieć neuronowa (**_Gradient descent_**)?
   - Wskaźniki techniczne (**RSI**, **Bandwidth**, **%B**)
3. Wykresy, które otrzymałem w wyniku prowadzenia badania
4. Małe kawałki kodu pokazujące ważne elementy pracy:
   - algorytm konwersji zbioru danych
   - trening modelu
   - ewaluacja modelu / testowanie
