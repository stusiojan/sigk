# Generacja Animacji Stickman za pomocą Sieci Dyfuzyjnych

Kasperczak Jonatan
Stusio Jan

---

## 1. Wstęp
Celem niniejszego projektu było opracowanie i implementacja modelu generatywnego opartego na procesie dyfuzji,
zdolnego do tworzenia animacji patyczaków na podstawie etykiet tekstowych.
Projekt zakładał generowanie dwóch rodzajów ruchu: chodu (walk) oraz skoku (jump),
przy zachowaniu parametrów wyjściowych 256x256 pikseli oraz długości animacji wynoszącej 48 klatek (2 sekundy).

## 2. Rozwiązanie

### 2.1. Zbiór danych
Wykorzystano bazę danych motion capture **CMU Mocap**, filtrując pliki BVH odpowiadające kategoriom ruchu "walk" (grupa 118) oraz "jump" (grupa 013).
Dane zostały przetworzone do formatu współrzędnych 3D dla 9 kluczowych stawów: biodra, kolana, ramiona i łokcie.
Zastosowano centrowanie bioder w każdej klatce oraz skalowanie wartości do przedziału [-1, 1], co miało wyeliminować moonwalking.

### 2.2. Architektura Modelu
Zaimplementowaliśmy model dyfuzyjny oparty na architekturze 1D U-Net z blokami rezydualnymi:
- **Enkoder/Dekoder:** Wykorzystuje warstwy splotowe 1D do przetwarzania sekwencji czasowej ruchu.
- **Iniekcja czasu i klasy:** Informacja o kroku dyfuzji (timestep) oraz typie ruchu (embedding klasy) jest wstrzykiwana do każdego bloku rezydualnego.
- **Harmonogram dyfuzji:** Cosine Beta Schedule zapewnia płynniejsze usuwanie szumu w porównaniu do harmonogramu liniowego.

### 2.3. Funkcja Straty
Oprócz standardowej straty błędu średniokwadratowego (MSE) dla przewidywanego szumu, wprowadziliśmy stratę geometryczną, czyli Bone Loss.
Oblicza ona euklidesową odległość między stawami (długość kości) i wymusza ich stałość w czasie, co zapobiega nienaturalnemu "rozciąganiu się" postaci podczas generacji.

## 3. Eksperymenty i Trenowanie
Model był trenowany przez 3000 epok przy użyciu optymalizatora Adam z współczynnikiem uczenia $2 \times 10^{-4}$. 
- **Liczba klatek:** 48 (24 FPS)
- **Liczba stawów:** 9 (koordynaty X, Y, Z)
- **Timesteps:** 500 kroków dyfuzji

W celu poprawy jakości wizualnej, do wygenerowanych próbek zastosowano filtr wygładzający Savitzky-Golay, który zredukował drgania kończyn wynikające ze specyfiki procesu samplowania stochastycznego.

## 4. Wyniki
Wyniki generacji zostały poddane ewaluacji ilościowej za pomocą trzech metryk:
1. **Frechet Motion Distance (FMD):** Mierzy podobieństwo dystrybucji wygenerowanych ruchów do danych rzeczywistych.
2. **Mean Per Joint Position Error (MPJPE):** Określa stabilność modelu i spójność wygenerowanych pozy.
3. **Variancja (Var):** Wskazuje na kreatywność modelu i różnorodność próbek dla tej samej etykiety.

### Tabela Wyników (Przykładowa):
| Ruch | FMD | MPJPE | Wariancja |
| :--- | :--- | :--- | :--- |
| Walk | ~0.12 | ~0.06 | ~0.005 |
| Jump | ~0.15 | ~0.08 | ~0.008 |

## 5. Wnioski
Zaimplementowany system dyfuzyjny skutecznie generuje płynne animacje stickmana dla obu zadanych klas ruchu.
Mimo, że stickman zachowuje kształt ludzika to animacje są nieidealne. W skoku model odtwarza ruch kończyn, ale nie radzi sobie z przesunięciami globalnymi
(translacją środka masy). Chód ma ten sam problem, a dodatkowo jest konwulsyjny.

Przy eksperymentach z innymi parametrami punkty nie tworzyły postaci, a raczej kłębek,
chód przypominał trochę animację pająka, a skok nie skakał pomimo zastosowania sieci 1D-ResNet, ograniczeń geometrycznych i uwzględnieniu bone loss w funkcji straty.
