# Propozycja struktury pracy inżynierskiej

Poniżej znajduje się propozycja struktury Twojej pracy inżynierskiej, która jest zgodna ze standardami pisania artykułów naukowych. Każda sekcja zawiera opis tego, co powinno się w niej znaleźć.

---

### 1. Strona tytułowa

- Tytuł pracy: _Zastosowanie filtrów Kalmana do poprawy predykcji cen giełdowych przy użyciu sieci LSTM_.
- Twoje imię i nazwisko, numer albumu.
- Nazwa uczelni, wydziału, kierunku studiów.
- Imię i nazwisko promotora.
- Data i miejsce złożenia pracy.

---

### 2. Streszczenie (Abstrakt)

- **Cel:** Krótkie (zazwyczaj 200-300 słów) podsumowanie całej pracy. Pisze się je na samym końcu, ale umieszcza na początku.
- **Co zawrzeć:**
  - **Wprowadzenie:** Jedno lub dwa zdania wprowadzające w problematykę predykcji cen giełdowych.
  - **Cel pracy:** Jasno określony główny cel, czyli zbadanie wpływu filtru Kalmana na dokładność predykcji sieci LSTM.
  - **Metodyka:** Skrócony opis użytych metod – wspomnij o zbiorze danych (AMZN), modelach LSTM (bazowy, wzbogacony, z filtrem Kalmana) i metrykach oceny (np. MSE, RMSE).
  - **Wyniki:** Główne ustalenia – czy filtr Kalmana poprawił wyniki? Jak wypadały poszczególne modele w porównaniu?
  - **Wnioski:** Jedno zdanie podsumowujące znaczenie Twoich wyników.
- **Słowa kluczowe:** 5-7 słów kluczowych, np. _predykcja cen giełdowych, sieci LSTM, filtr Kalmana, analiza techniczna, uczenie maszynowe, szeregi czasowe_.

---

### 3. Wstęp

- **Cel:** Wprowadzenie czytelnika w temat, przedstawienie motywacji, problemu badawczego i celu pracy.
- **Struktura:**
  - **Tło problemu:** Zacznij od ogólnego wprowadzenia w zagadnienie predykcji rynków finansowych. Podkreśl, dlaczego jest to trudne (chaotyczna natura rynków) i ważne (potencjalne zyski, zarządzanie ryzykiem).
  - **Problem badawczy:** Przejdź do konkretnego problemu – niedoskonałości istniejących modeli predykcyjnych. Wspomnij o popularności sieci LSTM w tym zadaniu.
  - **Proponowane rozwiązanie:** Przedstaw swoją hipotezę – że zastosowanie filtru Kalmana do "odszumienia" danych wejściowych może poprawić jakość predykcji modelu LSTM.
  - **Cel i zakres pracy:** Precyzyjnie sformułuj cel główny (jak w `PLAN.md`). Określ zakres pracy: analiza dotyczy akcji spółki AMZN, porównane zostaną trzy modele, a do optymalizacji użyjesz Optuny.
  - **Struktura pracy:** Na końcu wstępu opisz krótko, co znajduje się w kolejnych rozdziałach artykułu.

---

### 4. Podstawy teoretyczne i przegląd literatury

- **Cel:** Wyjaśnienie wszystkich kluczowych koncepcji i technologii, których używasz. To pokazuje, że rozumiesz teoretyczne podstawy swojej pracy.
- **Co opisać (każdy punkt jako osobny podrozdział):**
  - **4.1. Predykcja szeregów czasowych na rynkach finansowych:** Ogólne wprowadzenie, modele statystyczne (np. ARIMA) vs. modele uczenia maszynowego.
  - **4.2. Sieci neuronowe i uczenie głębokie:**
    - Podstawy działania sieci neuronowych.
    - Proces uczenia: **algorytm spadku gradientu (Gradient Descent)**, funkcja straty (MSE), problem **eksplodującego gradientu**, który napotkałeś i rozwiązałeś.
  - **4.3. Rekurencyjne sieci neuronowe i warstwy LSTM:**
    - Ograniczenia standardowych sieci neuronowych w przetwarzaniu sekwencji.
    - Architektura warstwy **LSTM** (komórka, bramki: zapominania, wejściowa, wyjściowa) – możesz tu użyć schematu, który masz w notatkach.
  - **4.4. Filtr Kalmana:**
    - Intuicja i cel działania: estymacja stanu ukrytego systemu na podstawie zaszumionych pomiarów.
    - Matematyczne podstawy: równania predykcji i korekcji (możesz przedstawić wzory, ale skup się na wyjaśnieniu ich sensu).
    - Zastosowanie w Twojej pracy: "odszumianie" szeregu czasowego cen akcji.
  - **4.5. Wskaźniki analizy technicznej:**
    - Wyjaśnij, czym są i dlaczego mogą pomóc w predykcji.
    - Opisz te, których użyłeś: **RSI**, **Bollinger Bands (%B, Bandwidth)**. Możesz dołączyć wykresy ilustrujące te wskaźniki.

---

### 5. Metodyka badań

- **Cel:** Szczegółowy opis krok po kroku, jak przeprowadziłeś swoje badanie. Ktoś, czytając ten rozdział, powinien być w stanie powtórzyć Twój eksperyment.
- **Struktura:**
  - **5.1. Środowisko badawcze:** Opisz narzędzia: Python, biblioteki (yfinance, PyTorch, scikit-learn, Optuna, Pandas), ewentualnie specyfikacja sprzętowa.
  - **5.2. Przygotowanie danych:**
    - **Źródło danych:** Pobieranie danych dla AMZN przy użyciu `yfinance` (określ zakres dat).
    - **Zbiór bazowy:** Opisz jego strukturę (OHLCV).
    - **Zbiór wzbogacony:** Opisz proces dodawania wskaźników technicznych.
    - **Zbiór filtrowany:** Opisz, jak zastosowałeś filtr Kalmana na oryginalnym zbiorze danych.
    - **Przetwarzanie wstępne:** Opisz proces konwersji danych do sekwencji (formatu dla LSTM), normalizację `MinMaxScaler` (wspomnij, że to rozwiązanie problemu eksplozji gradientu) oraz podział na zbiory: treningowy, walidacyjny i testowy (podaj proporcje, np. 80-10-10).
  - **5.3. Architektura modeli:**
    - Opisz architekturę sieci LSTM, której użyłeś (liczba warstw, liczba neuronów w warstwach, funkcja aktywacji, warstwa `Dropout` jeśli była używana).
    - Wspomnij, że dla modelu bazowego i przefiltrowanego architektura była taka sama, a dla wzbogaconego inna (jeśli tak było, np. inna liczba cech wejściowych).
  - **5.4. Proces treningu i oceny:**
    - **Funkcja straty:** MSE.
    - **Optymalizator:** np. Adam.
    - **Proces uczenia:** Opisz pętlę treningową, rolę zbioru walidacyjnego (monitorowanie przeuczenia).
    - **Metryki oceny:** RMSE (do interpretacji wyników, bo jest w tej samej jednostce co cena), MAE.
  - **5.5. Optymalizacja hiperparametrów:**
    - Opisz, jak użyłeś biblioteki **Optuna** do znalezienia najlepszych hiperparametrów (np. learning rate, liczba neuronów, rozmiar batcha). Określ przestrzeń przeszukiwania.

---

### 6. Wyniki i dyskusja

- **Cel:** Przedstawienie wyników eksperymentów i ich interpretacja.
- **Struktura:**
  - **6.1. Wyniki modelu bazowego:**
    - Przedstaw metryki (RMSE, MSE) na zbiorze testowym.
    - Pokaż wykresy: cena rzeczywista vs. przewidywana, dystrybucja reszt, krzywa uczenia (błąd w funkcji epok).
  - **6.2. Wyniki modelu wzbogaconego:**
    - Analogicznie przedstaw metryki i wykresy.
  - **6.3. Wyniki modelu z filtrem Kalmana:**
    - Analogicznie przedstaw metryki i wykresy.
  - **6.4. Porównanie modeli i dyskusja:**
    - Stwórz tabelę porównującą kluczowe metryki dla wszystkich trzech modeli.
    - Zinterpretuj wyniki. Który model okazał się najlepszy i dlaczego? Czy Twoja hipoteza się potwierdziła?
    - Omów wykresy. Czy modele dobrze naśladują trendy? Gdzie popełniają największe błędy?
    - Odnieś się do napotkanych problemów (eksplodujący gradient) i pokaż, jak Twoje rozwiązanie (normalizacja) zadziałało w praktyce.

---

### 7. Podsumowanie i wnioski

- **Cel:** Zebranie wszystkiego w całość, podsumowanie najważniejszych ustaleń i zaproponowanie dalszych kierunków badań.
- **Struktura:**
  - **Powtórzenie celu i metod:** Krótko przypomnij, co chciałeś osiągnąć i jak to zrobiłeś.
  - **Główne wnioski:** W punktach lub w zwięzłym akapicie przedstaw najważniejsze wyniki. Np. "Zastosowanie filtru Kalmana pozwoliło na redukcję błędu RMSE o X% w stosunku do modelu bazowego".
  - **Ograniczenia pracy:** Wskaż słabości Twojego podejścia (np. badanie tylko jednej spółki, testowanie w określonym horyzoncie czasowym).
  - **Propozycje dalszych badań:** Zaproponuj, co można by zrobić dalej. Np. przetestować inne filtry, zastosować inne architektury (np. GRU, Transformers), zbadać inne rynki lub dodać więcej wskaźników.

---

### 8. Bibliografia

- **Cel:** Stworzenie listy wszystkich źródeł (książek, artykułów naukowych, dokumentacji), na które powoływałeś się w tekście.
- **Format:** Użyj jednolitego stylu cytowania (np. APA, IEEE). LaTeX i menedżery bibliografii (BibTeX) bardzo to ułatwiają.

---

### 9. Spis rysunków i tabel

- LaTeX wygeneruje je automatycznie, jeśli będziesz poprawnie używać środowisk `figure` i `table` oraz dodawać do nich etykiety (`\label`) i podpisy (`\caption`).

---

### 10. Załączniki (Opcjonalnie)

- Możesz tu umieścić większe fragmenty kodu, które uznałeś za zbyt obszerne, by umieszczać je w głównym tekście, ale które są ważne dla zrozumienia implementacji.
