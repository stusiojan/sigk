# Pola wektorowe

**Projekt:** SIGK - Projekt 4
**Data:** 22.12.2025
**Zespół:**
[1. Kasperczak Jonatan]
[2. Stusio Jan]

---

## 1. Wstęp i Cel Projektu
Celem projektu było stworzenie modeli głębokiego uczenia (pól wektorowych), które potrafią przekształcić dowolny kształt wejściowy w zadany kształt docelowy (**Utah Teapot**). Zadanie wymagało wytrenowania trzech niezależnych modeli dla obiektów: **Bunny**, **Dragon** oraz **Armadillo**, a następnie przetestowania ich zdolności do generalizacji na nieznanym obiekcie **Asian Dragon**.

## 2. Metodologia

### 2.1. Architektura Modelu
Zastosowano sieć neuronową typu MLP (Multilayer Perceptron) działającą jako **Neuronowe Pole Deformacji** (Neural Deformation Field).
- **Wejście:** Współrzędne punktu w przestrzeni 3D $(x, y, z)$.
- **Wyjście:** Wektor przesunięcia $(\Delta x, \Delta y, \Delta z)$.
- **Zasada działania:** Nowa pozycja punktu obliczana jest jako $P_{final} = P_{initial} + MLP(P_{initial})$. Dzięki temu model uczy się ciągłego mapowania przestrzeni.

### 2.2. Preprocessing i Inwariancja
Zgodnie z wymogami projektu, aby pole działało niezależnie od położenia obiektu, zaimplementowano:
1. **Centrowanie:** Przesunięcie środka masy chmury do punktu $(0,0,0)$.
2. **PCA Alignment:** Wykorzystanie analizy składowych głównych do wyrównania osi obiektu. Rozwiązuje to problem różnej orientacji wejściowych plików .obj.
3. **Normalizację skali:** Przeskalowanie obiektów do jednostkowego boxa, co stabilizuje proces uczenia.

### 2.3. Funkcja Straty
Głównym składnikiem funkcji straty była **Odległość Chamfera (Chamfer Distance)**. Jest ona kluczowa w zadaniach chmur punktów, ponieważ nie wymaga ona znania korespondencji między punktem $i$ a punktem $j$ – mierzy ona ogólne dopasowanie dwóch kształtów.

---

## 3. Wyniki Eksperymentów

Poniższa tabela przedstawia wyniki metryk uzyskane po zakończeniu procesu uczenia oraz w fazie testowej.

| Metoda | Obiekt testowy | Chamfer Distance ↓ | IoU ↑ | Dice Coefficient ↑ |
| :--- | :--- | :--- | :--- | :--- |
| **bunny-flow** | Bunny | 0.000111 | 0.9010 | 0.9479 |
| **dragon-flow** | Dragon | 0.000106 | 0.8953 | 0.9448 |
| **armadillo-flow** | Armadillo | 0.000113 | 0.8817 | 0.9371 |
| **bunny-flow** | **Asian Dragon** | 0.041678 | 0.0979 | 0.1784 |
| **dragon-flow** | **Asian Dragon** | 0.029329 | 0.1622 | 0.2791 |
| **armadillo-flow** | **Asian Dragon** | 0.021598 | 0.1842 | 0.3110 |

---

## 4. Analiza i Wnioski

### 4.1. Efektywność Procesu Uczenia
Modele dedykowane (**Bunny, Dragon, Armadillo**) osiągnęły wysoką precyzję. Wartości IoU oscylujące wokół **0.88 - 0.90** wskazują na niemal idealne pokrycie objętościowe z modelem czajnika. Bardzo niska odległość Chamfera (ok. 0.0001) potwierdza, że powierzchnie obiektów są gładkie i dobrze dopasowane.



### 4.2. Zdolność do Generalizacji (Test na Asian Dragon)
Wprowadzenie nowego obiektu (**Asian Dragon**) do wytrenowanych pól wektorowych pozwoliło na wyciągnięcie następujących wniosków:
1. **Brak inwariancji kształtu:** Modele wykazują niskie wyniki IoU (średnio 15%) dla nowych obiektów. Wynika to z faktu, że sieć uczy się mapowania konkretnych współrzędnych "lokalnych" na przesunięcia. Jeśli geometria smoka różni się od geometrii królika, przesunięcia są aplikowane w "puste" lub niepoprawne miejsca.
2. **Transfer wiedzy:** Najlepszy wynik dla Asian Dragon uzyskał model `armadillo-flow` (IoU: 0.18). Może to sugerować, że chmura punktów Armadillo bardziej "pokrywała" przestrzeń zajmowaną przez punkty Smoka Azjatyckiego niż mniejszy model Królika.

