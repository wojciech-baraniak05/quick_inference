# Instrukcja: quick_inference

Ten dokument tłumaczy, **co robi projekt, jak przeprowadzać w nim badania, w jakim formacie są dane
i wyniki oraz dlaczego kod jest napisany tak, a nie inaczej**. Krótszy opis instalacji i struktury
katalogów jest w [README.md](README.md). Tutaj chodzi o zrozumienie całości.

---

## 1. O co chodzi w projekcie

Las losowy z setkami drzew przy predykcji odpytuje **każde** drzewo, nawet gdy po kilku
drzewach wynik jest już przesądzony. Projekt mierzy, ile da się zaoszczędzić, kończąc
głosowanie wcześniej, i ile kosztuje to w jakości.

Główna badana metoda to **statystyczny instance-based (IB) pruning**
(Hernández-Lobato, Martínez-Muñoz, Suárez, *IEEE TPAMI* 31(2), 2009):

- drzewa są pytane po kolei, a głosy zliczane,
- po każdym głosie sprawdzamy, czy klasa prowadząca **wygra także po zebraniu wszystkich T głosów**
  z prawdopodobieństwem ≥ `pi`,
- jeśli tak, przerywamy i zwracamy klasę prowadzącą.

Decyzja zależy **tylko od liczby oddanych głosów**, a nie od danych, więc da się ją policzyć
raz, z góry, dla wszystkich możliwych układów głosów. Wynik to **tablica progów**
(`data/tablice/*.npz`), a w czasie predykcji sprawdzenie warunku stopu to jeden odczyt z tablicy.

Każdą metodę porównujemy z dwiema metodami odniesienia:

| Metoda | Co robi | Po co jest |
|---|---|---|
| `rf_sklearn` | `rf.predict` ze scikit-learn: **soft voting**, czyli średnia prawdopodobieństw drzew | to, czego używa „normalny” użytkownik; punkt odniesienia dla czasu |
| `full_voting` | **hard voting**: każde drzewo oddaje jeden głos, wygrywa większość | to jest dokładnie to, co IB pruning przybliża; punkt odniesienia dla zgodności |
| `IB_pi98` itd. | IB pruning z poziomem `pi` | badana metoda |
| `HYPER_pi98` itd. | IB pruning z a priori rozkładu głosów szacowanym z danych (rozdz. 2a) | badana metoda |
| `GLEE_pi98` itd. | wczesny stop z gaussowskim przedziałem ufności, z artykułu COMET (rozdz. 2b) | badana metoda |
| `MARGIN_k0` itd. | ranking drzew i stała liczba drzew wyznaczona z marginesu lasu (rozdz. 2c) | badana metoda |

> **Ważne:** IB pruning przybliża *hard voting*, a nie `rf.predict`. Soft i hard voting
> zwykle dają ten sam wynik, ale nie zawsze (`agree_rf` między nimi to ok. 0,99). Dlatego to
> `agree_full` mówi, czy IB pruning „dotrzymuje obietnicy” `pi`, a `agree_rf` mówi, jak
> blisko jest tego, czego używa się w praktyce.

---

## 2. Jak działa IB pruning (intuicja i matematyka)

### Model

Mamy T drzew i l klas. Po odpytaniu części drzew mamy wektor głosów `t = (t_1, …, t_l)`.
Pytanie brzmi: jakie jest prawdopodobieństwo, że po zebraniu wszystkich T głosów
wygra ta sama klasa?

Artykuł modeluje pozostałe głosy jako **urnę Polyi** (rozkład Dirichleta-wielomianowy
z płaskim a priori): następny głos trafia do klasy `i` z prawdopodobieństwem

```
(t_i + 1) / (t_1 + … + t_l + l)
```

Im więcej głosów ma klasa, tym bardziej prawdopodobne, że dostanie kolejny. To założenie
ma sens, bo drzewa w lesie są wymienialne: każde jest uczone niezależnie na bootstrapie.

### Rekurencja (`wygrana` w [src/generate_table_IB.py](src/generate_table_IB.py))

```
wygrana(t) = [t_1 > max(reszta)]                        jeśli sum(t) == T
wygrana(t) = Σ_i P(głos do i) · wygrana(t + e_i)        w przeciwnym razie
```

Każdy stan zależy od stanów o jeden głos dalej, więc `lru_cache` liczy każdy stan tylko raz.
Liczy się **ścisła** wygrana, a remis na końcu traktujemy jak przegraną. To ostrożne
podejście: przy remisie nie przerywamy.

### Tablica progów

Pełna tablica „stop / nie stop” dla wszystkich układów głosów miałaby `(T+1)^l` komórek. Kod
korzysta z dwóch własności, żeby ją zmniejszyć:

1. **Symetria klas.** Wynik nie zależy od tego, *która* klasa prowadzi, więc głosy sortujemy
   malejąco: `t_1 ≥ t_2 ≥ … ≥ t_l`.
2. **Monotoniczność.** Przy ustalonych głosach pozostałych klas prawdopodobieństwo rośnie
   razem z `t_1`. Wystarczy więc zapamiętać **próg**:

```
prog[t_2, …, t_l] = najmniejsze t_1, przy którym P(wygrana) ≥ pi
                    (T + 1, jeśli przy tych t_2..t_l nie da się przerwać)
```

Tablica ma więc `l − 1` wymiarów zamiast `l`, a warunek stopu to:

```python
t = sorted(glosy, reverse=True)
stop = t[0] >= prog[tuple(t[1:])]
```

**Przykład** (`T501_L2_pi98.npz`, las 501 drzew, 2 klasy, `pi = 0.98`):
`prog[:6] = [5, 7, 9, 11, 13, 14]`. Jeśli pierwsze 5 drzew zagłosowało jednogłośnie,
przerywamy. Przy jednym głosie przeciw trzeba 7 głosów za itd. Na zbiorze `binary_check`
mediana liczby odpytanych drzew to **5**, a średnia ok. 23 (ok. 4,7% lasu).

---

## 2a. HYPER: a priori z danych

Metoda z artykułu Soto, Suárez, Martínez-Muñoz, *An urn model for majority voting in
classification ensembles* (NIPS 2016), kod w [src/methods/hyper.py](src/methods/hyper.py).

Las dla jednej próbki to **urna** z T kulami (drzewa) w l kolorach (klasy) o nieznanym
składzie `T = (T_1, …, T_l)`, a odpytywanie drzew to losowanie bez zwracania (rozkład
hipergeometryczny). Po głosach `t`:

```
P(T | t)  ∝  C(T_1, t_1) · … · C(T_l, t_l) · P(T)
P*(t, T)  =  suma P(T | t) po składach T, w których wygrywa klasa prowadząca w t
stop, gdy P*(t, T) ≥ pi
```

Przy jednostajnym `P(T)` to dokładnie IB pruning z rozdz. 2 (Stwierdzenie 3 w artykule).
HYPER szacuje `P(T)` z danych:

1. **Głosy out-of-bag** (`glosy_oob`): każdą próbkę uczącą oceniają tylko drzewa, które jej
   nie widziały. Stąd `fit(rf, X_train)` potrzebuje danych uczących i lasu z `bootstrap=True`.
2. **A priori** (`prior_z_glosow`): głosy próbki skalujemy do T drzew
   (`T_i ≈ T · głosy_i / suma`), robimy histogram składów i wygładzamy go oknem o szerokości 5.
3. **Tablica progów** (`tablica_hyper`): rekurencja wstecz po liczbie głosów,
   `f(t) = Σ_i f(t + e_i)`, liczona warstwami w NumPy. A priori nie jest symetryczne, więc
   **każda klasa prowadząca ma własną tablicę**: `prog[c, głosy pozostałych klas]`.

Różnice względem IB pruning w praktyce:

- tablica zależy od danych, więc jest liczona w `fit()` dla każdego lasu i **nie ma pliku**
  w `data/tablice/` (`T=501`: ok. 0,03 s dla 2 klas, ok. 7 s dla 3 klas; 4+ klasy przy
  `T=501` są pomijane z ostrzeżeniem, limit to 2 mln komórek `(T+1)^(l-1)`),
- remis na końcu jest liczony tak, jak rozstrzyga go `full_voting` (`argmax`, niższy indeks
  klasy), a nie jako przegrana,
- stany, które według a priori są niemożliwe, nie przerywają głosowania (chyba że pozostałe
  głosy nie mogą już zmienić wyniku),
- `agree_full` wypada **blisko** `pi`, a nie dużo powyżej, i w pojedynczych przebiegach może
  spaść nieco poniżej (a priori jest oszacowaniem, zwłaszcza na małych zbiorach); w zamian
  metoda odpytuje mniej drzew,
- `Hyper.expected_trees()` szacuje średnią liczbę drzew metodą Monte Carlo z samego a priori,
  bez danych testowych.

Parametry w configu: `pi`, `n_prior` (ile losowych próbek uczących użyć do głosów OOB,
domyślnie 20000) i `smooth` (okno wygładzania, 0 = bez).

---

## 2b. GLEE: gaussowski przedział ufności

Reguła stopu z artykułu Basilico, Munson, Kolda, Dixon, Kegelmeyer, *COMET: A Recipe for
Learning and Using Large Ensembles on Massive Data* (ICDM 2011, rozdz. II-C), kod w
[src/methods/glee.py](src/methods/glee.py). GLEE to skrót od *Gaussian Lazy Ensemble Evaluation*.

Głos drzewa to zmienna Bernoulliego, więc po `n` głosach odsetek głosów klasy prowadzącej `p̂`
ma w przybliżeniu rozkład normalny. Zamiast liczyć prawdopodobieństwo a posteriori (IB, HYPER),
GLEE buduje jednostronny przedział ufności `1 − α` dla prawdziwego odsetka `p` i przerywa
głosowanie, gdy jego dolna granica leży powyżej 0,5:

```
p̂ − ρ·δ > 0,5
δ = z_α · sqrt(p̂ (1 − p̂) / n)
ρ = sqrt((m − n) / (m − 1))   jeśli n > 0,05·m,   inaczej 1
```

- `m` to liczba drzew, `α = 1 − pi` to dopuszczalna częstość niezgodności z pełnym głosowaniem,
- `ρ` to **poprawka na populację skończoną**: drzewa losujemy bez zwracania, więc niepewność
  maleje, gdy zostaje ich mało (przy `n = m` przedział ma szerokość 0),
- przybliżenie normalne jest słabe dla małych `n`, więc stop jest dozwolony dopiero po
  `min_votes` głosach; artykuł podaje 15, 30 i 45 dla `α` = 10⁻², 10⁻³ i 10⁻⁴ i takie są
  wartości domyślne (dla `pi = 0.98` jest to 15),
- dla wielu klas klasa prowadząca jest traktowana jak klasa 1, a druga w kolejności jak
  klasa 0: `p̂ = t_1 / (t_1 + t_2)`, `n = t_1 + t_2`.

Tak jak w IB, regułę zamieniamy na **tablicę progów** (`tablica_glee`), tutaj jednowymiarową:
`prog[t_2]` = najmniejsze `t_1`, przy którym wolno przerwać. Dla `T = 501`, `pi = 0.98`:
`prog[:6] = [15, 14, 13, 12, 12, 13]`. Początkowe wartości wynikają z `min_votes = 15`
(15 głosów jednogłośnych albo 14 do 1 itd.), dalej decyduje już przedział ufności.

Różnice względem IB i HYPER w praktyce:

- tablica zależy tylko od `T` i parametrów, liczy się w `fit()` w kilka milisekund,
  **nie potrzeba pliku ani danych uczących**,
- działa dla **dowolnej liczby klas** (tablica nie rośnie z `l`), więc można ją badać także
  na zbiorach, na których IB i HYPER są pomijane,
- przez `min_votes` nigdy nie przerywa przed 15. drzewem, więc na łatwych zbiorach odpytuje
  więcej drzew niż IB (które potrafi przerwać po 5 jednogłośnych głosach), a `agree_full`
  wypada wyraźnie powyżej `pi`,
- `fpc = false` i `one_sided = false` dają warianty G1, G2 i G2-FPC z rys. 1 artykułu;
  domyślne ustawienia to G1-FPC, czyli reguła, którą artykuł nazywa GLEE.

Z COMET wzięta jest tylko ta reguła. Część o uczeniu (MapReduce, lasy IVoting) dotyczy budowy
zespołu, a projekt bada inferencję na gotowym lesie ze scikit-learn.

Parametry w configu: `pi`, `min_votes`, `fpc`, `one_sided`.

---

## 2c. MARGIN: ranking drzew i liczba drzew z marginesu

Metoda z artykułu Barbareschi, Emmanuele, *A Margin Based Early-Stopping Approach for Random
Forest Classifiers* (DSN-W 2025), kod w [src/methods/margin.py](src/methods/margin.py).

**Margines** próbki mówi, jak pewne jest głosowanie:

```
M = (v_c − v_c*) / N
```

`v_c` to głosy klasy prawdziwej, `v_c*` głosy najlepszej innej klasy, `N` liczba drzew.
`M > 0` oznacza poprawną klasyfikację, `M` bliskie 1 to prawie jednogłośny las.

**Faza offline** (`fit(rf, X_train, y_train)`), na zbiorze kalibracyjnym z etykietami:

1. **Ranking drzew** (`ranking_drzew`). Domyślnie `mean_d_m`, czyli kryterium MeanD-M użyte
   w eksperymentach artykułu: ocena drzewa to średni spadek marginesu próbek po usunięciu
   tego jednego drzewa z zespołu (`spadek_marginesu`). Drzewa o największym spadku idą
   pierwsze. Inne opcje: `accuracy` (trafność pojedynczego drzewa) i `none` (kolejność lasu).
2. **Statystyki marginesu** (`statystyki_marginesu`): średnia `mu` i odchylenie standardowe
   `sigma` marginesu próbek **poprawnie sklasyfikowanych**, a z nich liczba drzew:

   ```
   N* = N · (mu + k · sigma)          zaokrąglone, w przedziale [1, N]
   ```

**Faza po wdrożeniu**: każdą próbkę ocenia `N*` pierwszych drzew rankingu, wynik to
głosowanie większościowe tych drzew. To jedyna badana metoda, w której liczba drzew jest
**taka sama dla każdej próbki** (`trees_mean = trees_median = trees_max = N*`).

Różnice względem IB, HYPER i GLEE w praktyce:

- nie ma poziomu ufności `pi`; kompromis ustawia `k` (`k > 0`: więcej drzew i większa zgodność,
  `k < 0`: mniej drzew). Końcowe sprawdzenie `agree_full ≥ pi` w `testing.py` tej metody nie
  obejmuje,
- zbiór kalibracyjny to **dane uczące oceniane out-of-bag** (jak w HYPER), a nie połowa zbioru
  testowego jak w artykule. Dzięki temu cały zbiór testowy zostaje do pomiaru. Stąd `fit`
  potrzebuje `X_train`, `y_train` i lasu z `bootstrap=True`,
- `N*` rośnie ze średnim marginesem, więc na **łatwych** zbiorach (duże marginesy) metoda
  zostawia **więcej** drzew niż na trudnych. Przy `k = 0` i lesie 501 drzew to ok. 310–350
  drzew na sprawdzonych zbiorach binarnych, czyli wielokrotnie więcej niż IB przy zbliżonym
  `agree_full`,
- ranking `mean_d_m` wybiera drzewa najlepiej wspierające klasę prawdziwą: podzespół bywa
  **trafniejszy** od pełnego lasu, ale przez to **mniej zgodny** z pełnym głosowaniem niż
  tyle samo drzew w kolejności lasu (`ranking = "none"`),
- działa dla dowolnej liczby klas, bez pliku z tablicą; `fit` trwa poniżej sekundy,
- `set_k(k)` zmienia `k` po `fit` bez ponownej kalibracji. Artykuł wspomina o strojeniu `k`
  na urządzeniu na podstawie obserwowanej trafności, ale nie podaje algorytmu, więc tej
  części nie ma w kodzie.

Parametry w configu: `k`, `ranking`, `n_calib` (ile losowych próbek uczących użyć do
kalibracji, domyślnie 20000).

---

## 3. Szybki start

```bat
setup_windows.bat              :: venv + pakiety + zbiory syntetyczne + zbiory UCI
.venv\Scripts\activate
cd src
python testing.py
```

Albo w VS Code: otwórz [src/testing.py](src/testing.py) i uruchamiaj komórki `# %%`
(Run Cell), wtedy obiekt `suite` zostaje w pamięci i można go dalej analizować.

**Skrypty uruchamiamy z katalogu `src/`.** Moduły importują się nawzajem jako moduły
najwyższego poziomu (`import config`, `from methods import …`, `from generate_table_IB import …`),
więc `src/` musi być w `sys.path`. Wyjątkiem są skrypty w `src/data_generate/`, które same
dopisują ścieżkę i można je uruchamiać z katalogu głównego.

---

## 4. Co się dzieje po uruchomieniu `testing.py`

```
config.toml ──► config.py (stałe) ──► testing.py
                                         │
   dla każdego pliku z benchmark.datasets:
       wczytaj(plik) → X, y
       dla każdego seeda z benchmark.seeds:
           train_test_split(test_size, random_state=seed)
           train_forest(n_trees, max_depth, seed)        ← trening na wszystkich rdzeniach
           build_methods([[methods]])                    ← NOWE instancje dla tego lasu
           methods_rf.add_method(m) → m.fit(rf, X_train, y_train)
                                                         ← np. dobór tablicy IB do T i l, a priori HYPER,
                                                           ranking i N* metody MARGIN
           benchmark.run(modes)                          ← batch i/lub online, pomiar czasu
           suite.add(benchmark)
   suite.summary()                                       ← tabele na ekran
   sprawdzenie: czy agree_full ≥ pi w każdym przebiegu
   suite.save(data/<out_dir>)                            ← CSV + NPZ
```

Warstwy kodu:

| Warstwa | Pliki | Odpowiedzialność |
|---|---|---|
| konfiguracja | [config.toml](config.toml), [src/config.py](src/config.py) | jedyne miejsce z parametrami badania |
| metody | [src/methods/](src/methods/) | *jak* przewidywać; każda metoda zwraca predykcję i liczbę odpytanych drzew |
| benchmark | [src/benchmarking/](src/benchmarking/) | *jak mierzyć*; nie wie nic o `config.toml` |
| skrypt | [src/testing.py](src/testing.py) | łączy config z benchmarkiem |
| tablice IB | [src/generate_table_IB.py](src/generate_table_IB.py) | liczenie i zapis tablic progów |
| dane | [src/data_generate/](src/data_generate/) | generowanie i pobieranie zbiorów |

---

## 5. Konfiguracja (`config.toml`)

Wszystkie parametry badania są w [config.toml](config.toml). Nie trzeba niczego zmieniać w kodzie.

```toml
seed = 1                              # tylko dla generate_datasets.py

[benchmark]
datasets = ["generated/binary"]       # foldery (wszystkie *.parquet) lub pojedyncze pliki, względem data/
out_dir  = "results/binary"           # względem data/; NADPISYWANY przy każdym uruchomieniu
modes    = ["batch", "online"]
seeds    = [1, 2, 3]                  # seed = podział train/test ORAZ las
test_size = 0.3
n_online  = 500                       # pierwsze n_online próbek X_test w trybie online
n_repeats = 1                         # pomiar czasu: mediana z n_repeats uruchomień
sklearn_soft_voting = true            # czy dodać rf_sklearn (i kolumny *_rf)

[forest]
n_trees   = 501
max_depth = 5                         # usuń klucz → drzewa bez limitu głębokości

[[methods]]                           # jeden blok = jedna metoda
type = "full_voting"

[[methods]]
type = "ib_pruning"
pi = 0.98                             # → tablica T{n_trees}_L{l}_pi98.npz
# table = "T501_L2_pi98.npz"          # albo konkretny plik
# name = "IB_moja_nazwa"              # opcjonalna nazwa w wynikach
# enabled = false                     # wyłącza blok bez kasowania

[[methods]]
type = "hyper"                        # a priori z głosów out-of-bag, bez pliku z tablicą
pi = 0.98

[[methods]]
type = "glee"                         # gaussowski przedział ufności (COMET), bez pliku i bez danych
pi = 0.98

[[methods]]
type = "margin"                       # ranking drzew + N* = n_trees · (mu + k · sigma)
k = 0.0                               # k > 0: więcej drzew, k < 0: mniej
# ranking = "mean_d_m"                # albo "accuracy", "none"
```

Zasady:

- Wszystkie pola bloku `[[methods]]` poza `type`, `enabled` przekazywane są do konstruktora
  klasy jako argumenty (`IBPruning(pi=0.98)`). Literówka w nazwie parametru daje czytelny błąd.
- Nazwy metod muszą być unikalne. Dwa bloki `ib_pruning` z tym samym `pi` wymagają różnych `name`.
- `rf_sklearn` **nie** dodaje się przez `[[methods]]`, tylko przez `sklearn_soft_voting`.
- Nowy parametr: dopisz go do `config.toml` **i** wystaw jako stałą w [src/config.py](src/config.py).

---

## 6. Formaty danych

### 6.1. Zbiory wejściowe (`.parquet`)

Każdy plik to jedna tabela:

| Kolumny | Typ | Znaczenie |
|---|---|---|
| `f0, f1, …, f{p-1}` | liczby (float) | cechy |
| `target` | int (klasy) / float (regresja) | etykieta |
| *(opcjonalnie)* inne nazwy, np. `year`, `CO` w `gas_turbine` | dowolny | kolumny dodatkowe |

`wczytaj()` ([src/benchmarking/data.py](src/benchmarking/data.py)) bierze **wszystkie kolumny
poza `target`** jako cechy, a kolumny nieliczbowe zamienia na kody całkowite
(`astype("category").cat.codes`). Zwraca `X` typu `float` i `y`.

> Uwaga: w `gas_turbine` kolumny `year` i `CO` *też* trafią do cech, jeśli ten zbiór kiedyś
> zostanie użyty w benchmarku. Trzeba je wtedy usunąć ręcznie.

Własny zbiór dodajesz, zapisując DataFrame w tym formacie:

```python
df = pd.DataFrame(X).add_prefix("f")
df["target"] = y
df.to_parquet("data/moje/moj_zbior.parquet", index=False)
```

i wpisując `datasets = ["moje"]` w configu.

Dostępne zbiory i ich liczba klas (liczba klas jest istotna, patrz rozdz. 8):

| Zbiór | Klasy | Wiersze |
|---|---|---|
| `generated/binary/*` (11 plików) | 2 | 500 – 500 000 |
| `multi_check`, `multi_rare_class` | 3 | 100 000 / 200 000 |
| `multi_small`, `multi_imbalanced`, `multi_hard_nonlinear`, `multi_wide_p_gt_n`, `multi_collinear_unscaled`, `multi_outliers` | 4 | 500 – 20 000 |
| `multi_large`, `multi_multi_features`, `multi_noisy_labels` | 5 | 20 000 – 500 000 |
| `multi_many_classes` | 20 | 50 000 |
| `real_data/binary/susy` | 2 | **5 000 000** |
| `real_data/multiclass/shuttle` | 7 | 58 000 |
| `generated/regression/*`, `real_data/regression/gas_turbine` | regresja | nie dla tego benchmarku |

Folder `data/real_data/extra_time_data/` (MetroPT, WISDM, czujniki gazu) zawiera surowe pliki
dodane ręcznie. Kod ich nie używa.

### 6.2. Tablice progów (`data/tablice/T{T}_L{l}_pi{pi·100}.npz`)

| Klucz | Kształt / typ | Znaczenie |
|---|---|---|
| `prog` | `(T+1,) * (l-1)`, `uint8`/`uint16` | tablica progów z rozdz. 2; `T+1` = „nie da się przerwać” |
| `T`, `l`, `pi` | skalary | parametry, z którymi ją policzono |
| `time` | float | czas generowania w sekundach |

Ważne są tylko komórki o indeksach posortowanych malejąco (`t_2 ≥ t_3 ≥ …`). Pozostałe
mają wartość `T+1` i nigdy nie są czytane, bo głosy zawsze się sortuje przed odczytem.
Typ jest najmniejszym typem całkowitym mieszczącym `T+1` (`np.min_scalar_type`), co
zmniejsza rozmiar pliku.

Generowanie (z `src/`):

```python
from generate_table_IB import generuj_tablice
generuj_tablice(T=501, l=2, pi=0.95)       # → data/tablice/T501_L2_pi95.npz
```

Czasy dla odniesienia: `T=501, l=2` ok. 0,2 s; `T=501, l=3` ok. 83 s.

### 6.3. Wyniki (`data/<out_dir>/`)

```
results_batch.csv      jeden wiersz na (dataset, seed, method)
results_online.csv
aggregate_batch.csv    średnia i odch. std po seedach, wiersz na (dataset, method)
aggregate_online.csv
runs/<dataset>_seed<N>.npz   surowe predykcje jednego przebiegu
```

**Kolumny `results_*.csv`:**

| Kolumna | Znaczenie |
|---|---|
| `dataset`, `seed`, `method` | indeks |
| `accuracy`, `precision`, `recall`, `f1` | jakość względem prawdziwych etykiet (precision/recall/F1 makro-uśrednione) |
| `agree_rf` | odsetek próbek z tą samą predykcją co `rf_sklearn` (tylko gdy `sklearn_soft_voting = true`) |
| `agree_full` | to samo względem `full_voting` (tylko gdy `full_voting` jest w `[[methods]]`) |
| `time_s` | łączny czas predykcji całego zbioru (mediana z `n_repeats`) |
| `ms_per_sample` | `time_s / liczba próbek · 1000` |
| `speedup_vs_rf`, `speedup_vs_full` | ile razy szybciej niż metoda odniesienia (>1 = szybciej) |
| `trees_mean`, `trees_median`, `trees_max` | liczba odpytanych drzew na próbkę |
| `trees_frac` | `trees_mean / T`, czyli jaka część lasu była potrzebna |

`aggregate_*.csv` ma **dwa wiersze nagłówka** (metryka, statystyka), więc czytamy go tak:

```python
pd.read_csv("aggregate_batch.csv", header=[0, 1], index_col=[0, 1])
```

**Zawartość `runs/*.npz`** (wiersz `i` macierzy = metoda `names[i]`):

| Klucz | Kształt | Znaczenie |
|---|---|---|
| `metadata` | str (JSON) | dataset, seed, T, l, n_test, n_online, lista metod, data uruchomienia… |
| `names` | `(n_metod,)` | nazwy metod |
| `classes` | `(l,)` | etykiety klas |
| `y_test` | `(n_test,)` | prawdziwe etykiety całego zbioru testowego |
| `Y_pred_{mode}` | `(n_metod, n_próbek)` | predykcje |
| `N_used_{mode}` | `(n_metod, n_próbek)` | liczba odpytanych drzew dla każdej próbki |
| `times_{mode}` | `(n_metod,)` | czas w sekundach |

W trybie `online` próbek jest `n_online`, czyli to pierwsze `n_online` elementów `y_test`.

```python
import json, numpy as np
d = np.load("data/results/binary/runs/binary_small_seed1.npz")
meta = json.loads(str(d["metadata"]))
names = list(d["names"])
n_ib = d["N_used_batch"][names.index("IB_pi98")]     # rozkład liczby drzew per próbka
```

`N_used` pozwala analizować np. histogram liczby drzew albo zależność „trudności” próbki
od tego, czy IB pomylił się względem pełnego głosowania.

---

## 7. Jak prowadzić badania

### 7.1. Typowe eksperymenty

| Pytanie | Co zmienić w `config.toml` |
|---|---|
| Wpływ poziomu ufności | kilka bloków `ib_pruning` z różnym `pi` (0,9 / 0,95 / 0,98 / 0,99); najpierw wygeneruj tablice |
| Wpływ rozmiaru lasu | `forest.n_trees` (+ tablice dla nowego T); osobne `out_dir` dla każdego T |
| Wpływ głębokości drzew | `forest.max_depth` (np. 5, 10, brak) |
| Zbiory wieloklasowe | `datasets = ["generated/multiclass/multi_check.parquet", …]` i tablica `L3` |
| Duże/rzeczywiste dane | `datasets = ["real_data/binary"]`; uwaga na rozmiar SUSY (rozdz. 8) |
| Rzetelne czasy | `n_repeats = 3–5`, większe `n_online` |

**Zawsze zmieniaj `out_dir`** przy nowym eksperymencie. Wyniki są nadpisywane bez pytania.

### 7.2. Jak czytać wyniki

- **Zgodność:** metoda MARGIN nie ma `pi`; jej `agree_full` zależy od `k` i trzeba je po prostu
  odczytać z wyników. Dla IB pruning `agree_full` powinno być ≥ `pi`. Na końcu `testing.py` wypisuje
  przebiegi, w których tak nie jest. Zwykle `agree_full` jest *znacznie* wyższe niż `pi`
  (np. 0,997 przy `pi = 0.98`), bo `pi` to (w ramach modelu urny Polyi) minimalne
prawdopodobieństwo zgodności w momencie przerwania dla pojedynczej próbki, a większość
  próbek jest daleko od granicy decyzyjnej.
- **Jakość:** `accuracy` IB jest bardzo blisko `full_voting`. Różnica w stosunku do `rf_sklearn`
  wynika głównie z soft vs hard voting, a nie z przycinania.
- **Oszczędność pracy:** `trees_frac` to zysk „teoretyczny”, niezależny od implementacji.
  Ta wartość najlepiej nadaje się do porównań między metodami.
- **Oszczędność czasu:** `speedup_vs_*` zależy od trybu.

### 7.3. Batch a online: dlaczego wyniki tak się różnią

Przykład z `binary_check` / `hard_nonlinear` (501 drzew, `max_depth=5`, `pi=0.98`):

| | `trees_frac` | `speedup_vs_full` online | `speedup_vs_full` batch |
|---|---|---|---|
| binary_check | 0,043 | **23,8×** | 6,1× |
| hard_nonlinear | 0,065–0,072 | **17,3×** | 2,4× |

- **Online** (jedna próbka na raz): koszt to praktycznie `liczba wywołań tree.predict × stały
  narzut sklearn` (walidacja wejścia itp., ok. 0,1 ms na wywołanie). Czas spada więc prawie
  proporcjonalnie do `trees_frac`. To scenariusz, dla którego IB pruning jest najbardziej
  naturalny (np. serwowanie modelu na żądanie).
- **Batch** (cały zbiór naraz): każde drzewo i tak trzeba wywołać, dopóki *jakakolwiek*
  próbka jest aktywna. Zysk bierze się z tego, że kolejne drzewa dostają coraz mniejszy podzbiór.
  Narzut wywołań i rzadkie „trudne” próbki (`trees_max = 501`) ograniczają przyspieszenie.

Z tego samego powodu `full_voting` jest ok. 2× wolniejsze od `rf_sklearn`, choć odpytuje
tyle samo drzew. `rf.predict` sumuje `predict_proba` drzew w jednym przebiegu, a
`tree.predict` to wewnętrznie `predict_proba` + `argmax` + zamiana na etykiety, do tego
dochodzi nasze zliczanie głosów w NumPy. Dlatego przyspieszenie podajemy względem obu metod
odniesienia.

### 7.4. Analiza w notebooku / interaktywnie

Po uruchomieniu komórek `testing.py` obiekt `suite` jest w pamięci:

```python
suite.results("batch")                      # pełna tabela (MultiIndex: dataset, seed, method)
suite.aggregate("online")                   # mean/std po seedach
suite.aggregate("batch", overall=True)      # jeden wiersz na metodę, średnia po wszystkim
suite.runs[0].N_used["batch"]               # surowe liczby drzew pierwszego przebiegu
```

Notebooki w [src/analysis/](src/analysis/) to eksploracja *zbiorów danych* (rozkłady cech,
wizualizacja konfiguracji generatora), a nie wyników benchmarku.
[generated_configs_viz.ipynb](src/analysis/generated_configs_viz.ipynb) miejscami opisuje
wcześniejszą wersję generatora (np. `CATEGORICAL_FEATURES`), której w
[generate_datasets.py](src/data_generate/generate_datasets.py) już nie ma.

---

## 8. Ograniczenia i pułapki

1. **Tylko klasyfikacja.** Benchmark używa `RandomForestClassifier`. Zbiory regresyjne
   (`generated/regression`, `gas_turbine`) są przygotowane na przyszłość. Wpisanie ich do
   `datasets` zrobi z każdej unikalnej wartości `target` osobną klasę. Nie rób tego.

2. **Liczba klas a tablice IB.** Rekurencja odwiedza rzędu `C(T+l, l)` stanów:

   | T | l | stany | praktycznie |
   |---|---|---|---|
   | 501 | 2 | ~126 tys. | ułamek sekundy |
   | 501 | 3 | ~21 mln | ok. 1,5 min |
   | 501 | 4 | ~2,7 mld | **niewykonalne** (pamięć cache) |
   | 51 | 4 | ~341 tys. | szybko |
   | 51 | 5 | ~3,8 mln | do zrobienia |

   Przy `T = 501` IB pruning działa więc tylko dla 2 i 3 klas. Dla zbiorów z 4+ klasami
   (`multi_small`, `shuttle`, `multi_many_classes`, …) metoda zostanie **pominięta
   z ostrzeżeniem** (`pomijam IB_pi98: brak tablicy …`), a reszta benchmarku pójdzie dalej.
   Żeby badać więcej klas, trzeba zmniejszyć `n_trees` albo zmienić algorytm liczenia tablicy.

3. **SUSY ma 5 mln wierszy.** Zbiór testowy to 1,5 mln próbek × 501 drzew × kilka metod × kilka
   seedów. Najpierw ustaw jeden seed i sprawdź czas.

4. **Pomiar czasu jest zaszumiony.** Domyślnie `n_repeats = 1`. Do wyników końcowych zwiększ
   tę wartość i nie obciążaj komputera w trakcie. Wszystkie metody działają w jednym wątku (rozdz. 9).

5. **Parzyste T.** Przy parzystej liczbie drzew możliwy jest remis. `full_voting` wybiera wtedy
   klasę o niższym indeksie, a IB traktuje remis jako „nie wygrana”. Dla klasyfikacji binarnej
   używaj nieparzystego T (501, 51).

6. **`requirements.txt` jest w UTF-16** (efekt `pip freeze > requirements.txt` w PowerShell).
   `pip` sobie z tym radzi, ale `git diff` i niektóre edytory pokazują go jako plik binarny.
   Przy kolejnej aktualizacji warto zapisać go w UTF-8:
   `pip freeze | Out-File -Encoding utf8 requirements.txt`.

---

## 9. Dlaczego kod wygląda tak, a nie inaczej

**`tree.predict` zwraca indeks klasy, nie etykietę.** `RandomForestClassifier` wewnętrznie
koduje `y` jako `0..l-1` i na tym uczy drzewa, więc `rf.estimators_[k].predict(X)` zwraca
indeksy (jako float). Dlatego w metodach jest `.astype(int)` do indeksowania wektora głosów,
a na końcu `self.classes[votes.argmax()]`, które zamienia indeks z powrotem na etykietę.

**Własne głosowanie zamiast `rf.predict`.** `rf.predict` zawsze odpytuje wszystkie drzewa
i uśrednia prawdopodobieństwa. Żeby przerwać w połowie, musimy sami iterować po
`rf.estimators_`. `full_voting` robi to samo bez przerywania, więc różnica czasu
`full_voting` vs `IB` pokazuje czysty zysk z przycinania, a nie różnice implementacji.

**Kolejność drzew to po prostu `rf.estimators_`.** Model urny Polyi zakłada, że kolejne
głosy są wymienialne. Drzewa lasu losowego są uczone niezależnie, więc ich kolejność jest
losowa. Sortowanie drzew np. po trafności złamałoby to założenie, a `pi` przestałoby
oznaczać to, co ma oznaczać.
Opcja `order = "random"` w bloku `ib_pruning` (z opcjonalnym `seed`) daje każdej próbce
własną losową permutację drzew, więc kolejność jest losowa niezależnie od tego, jak las je
zbudował. W trybie batch próbki pytające w danym kroku to samo drzewo są grupowane w jedno
wywołanie `predict`, ale wywołań jest dużo więcej niż przy `"fixed"`, więc batch jest
wyraźnie wolniejszy (ok. 10× w teście na 501 drzewach). W trybie online narzut jest
niewielki (jedna permutacja na próbkę).

**Tablica liczona offline, nie w locie.** Progi zależą tylko od `(T, l, pi)`, a nie od danych.
Liczenie rekurencji przy każdej predykcji byłoby zbyt wolne, a odczyt z tablicy to O(1).
Stąd osobny moduł [generate_table_IB.py](src/generate_table_IB.py) i pliki `.npz` w repozytorium.

**Tablica progów zamiast tablicy True/False.** Dzięki monotoniczności jeden wymiar mniej
(z `(T+1)^l` do `(T+1)^(l-1)`). Dzięki sortowaniu głosów korzystamy tylko z jednej
„permutacji”, a `lru_cache` w `wygrana` ma więcej trafień. `sys.setrecursionlimit` jest
potrzebne, bo głębokość rekurencji to ok. T. `cache_clear()` na końcu zwalnia pamięć
po wygenerowaniu tablicy.

**Wektorowy IB w trybie batch (`predict_batch`).** Zamiast pętli po próbkach jest pętla po
drzewach z malejącym zbiorem `aktywne`. Każde drzewo dostaje jednym wywołaniem wszystkie
jeszcze niezdecydowane próbki, co ogranicza narzut wywołań sklearn. Pętla kończy się,
gdy wszystkie próbki są zdecydowane. Pętle `predict_batch` i `predict_one` są wspólne dla
IB, HYPER i GLEE (klasa `SequentialPruning` w [base.py](src/methods/base.py)); metody różnią się
tylko regułą stopu. `czy_stop_wektorowo` sortuje głosy w każdym wierszu
i czyta progi wektorowo przez `prog[tuple(t[:, 1:].T)]`.

**`FixedVoting`: głosowanie stałej listy drzew.** `full_voting` i MARGIN robią to samo:
każdą próbkę ocenia ta sama lista drzew `self.voters` (wszystkie drzewa albo `N*` pierwszych
z rankingu). Pętle głosowania są więc raz, w klasie `FixedVoting` w
[base.py](src/methods/base.py), bez sprawdzania stopu po każdym drzewie. Dzięki temu czas
MARGIN i `full_voting` różni się tylko liczbą drzew.

**Głosy out-of-bag w osobnym module.** HYPER i MARGIN kalibrują się na danych uczących tak,
żeby każdą próbkę oceniały tylko drzewa, które jej nie widziały. Wspólny kod (indeksy
bootstrapu, losowa podpróbka, sprawdzenie danych) jest w [oob.py](src/methods/oob.py).

**Dwa tryby: `batch` i `online`.** Odpowiadają dwóm realnym zastosowaniom (przetwarzanie
wsadowe vs serwowanie pojedynczych zapytań), w których ten sam algorytm daje bardzo różne
przyspieszenie (rozdz. 7.3). `predict_one` dostaje `x[None, :]` o kształcie `(1, n_cech)`,
bo sklearn wymaga macierzy 2D.

**Trening na wszystkich rdzeniach, predykcja w jednym wątku** (`train_forest`).
`n_jobs=-1` przyspiesza trening. Potem `set_params(n_jobs=None)`, bo inaczej `rf.predict`
liczyłby równolegle, a nasze metody sekwencyjnie, i porównanie czasu byłoby nieuczciwe.

**Nowe instancje metod dla każdego lasu.** `build_methods` jest wołane w pętli po seedach,
bo `fit(rf)` wiąże metodę z konkretnym lasem (np. `IBPruning` dobiera tablicę do `T` i `l`
tego lasu). Na początku `testing.py` jest dodatkowe wywołanie `build_methods` wyłącznie
po to, by **błąd w configu wyszedł od razu**, a nie po godzinie liczenia.

**Pomijanie metody zamiast przerywania.** Gdy dla danego zbioru brakuje tablicy (np. 5 klas),
`testing.py` łapie `FileNotFoundError`/`ValueError` z `add_method`, wypisuje ostrzeżenie
i liczy pozostałe metody. Jedno wywołanie może więc objąć zbiory o różnej liczbie klas.

**`methods_rf(rf, methods=[])`.** Domyślnie `methods_rf` dodaje `rf_sklearn` i `full_voting`
(wygodne przy pracy w notebooku). Skrypt startuje od pustej listy, żeby zestaw metod
wynikał wyłącznie z configu.

**Kolumny porównawcze tylko, gdy jest punkt odniesienia.** `agree_rf`/`speedup_vs_rf` są
liczone tylko, jeśli `rf_sklearn` jest w benchmarku, a `*_full` tylko z `full_voting`.
Brak kolumny jest bezpieczniejszy niż kolumna z wartością, która nic nie znaczy.

**Jeden seed na przebieg.** Ten sam seed steruje podziałem train/test i losowością lasu,
więc przebieg `(dataset, seed)` jest w pełni powtarzalny. Kilka seedów daje średnią i odchylenie.

**Podział `methods/` i `benchmarking/`.** Metody nie wiedzą nic o mierzeniu, a benchmark nie
wie nic o `config.toml`. Nową metodę dodaje się, nie dotykając kodu benchmarku. Benchmarku
można też używać bez configu, np. w notebooku.

**Zapis CSV + NPZ.** CSV to tabele gotowe do Excela lub pandas. NPZ trzyma surowe predykcje
i `N_used` dla każdej próbki, więc później można policzyć nowe metryki **bez ponownego
uruchamiania** benchmarku.

**Polskie i angielskie nazwy.** Kod powstawał stopniowo: nowsze, „publiczne” elementy mają
nazwy angielskie (`benchmark`, `methods_rf`, `InferenceMethod`), a funkcje pomocnicze
i obliczeniowe polskie (`wczytaj`, `znajdz_zbiory`, `czy_stop`, `prog`, `generuj_tablice`).
Klasy `benchmark`, `methods_rf`, `benchmark_suite` mają nazwy małymi literami z przyczyn
historycznych.

---

## 10. Dodawanie własnej metody

1. Utwórz plik w [src/methods/](src/methods/), np. `moja.py`:

   ```python
   import numpy as np
   from .base import InferenceMethod

   class PierwszeK(InferenceMethod):
       """Głosowanie tylko k pierwszych drzew (prosty baseline)."""
       name = "first_k"

       def __init__(self, k=50, name=None):
           self.k = k
           super().__init__(name or f"first_{k}")

       def predict_batch(self, X):
           votes = np.zeros((len(X), self.l), dtype=int)
           rows = np.arange(len(X))
           for tree in self.trees[: self.k]:
               votes[rows, tree.predict(X).astype(int)] += 1   # indeks klasy, nie etykieta!
           return self.classes[votes.argmax(axis=1)], np.full(len(X), min(self.k, self.T))

       def predict_one(self, x):
           y, n = self.predict_batch(x)
           return y[0], n[0]
   ```

   Kontrakt: `predict_batch(X) -> (y_pred, n_used)` i `predict_one(x) -> (y, n_used)`,
   gdzie `n_used` to liczba odpytanych drzew. `fit(rf)` z klasy bazowej ustawia
   `self.trees`, `self.classes`, `self.T`, `self.l`. Nadpisz go (wołając `super().fit(rf)`),
   jeśli metoda potrzebuje przygotowania pod konkretny las. Sygnatura to
   `fit(rf, X_train=None, y_train=None)`: benchmark przekazuje dane uczące lasu.
   Metoda z wczesnym stopem może dziedziczyć po `SequentialPruning` i podać samo
   `_stop_batch(votes)`. Metoda głosująca stałą listą drzew (jak powyższy przykład) może
   dziedziczyć po `FixedVoting` i w `fit` ustawić samo `self.voters`.

2. Zarejestruj ją w [src/methods/registry.py](src/methods/registry.py):

   ```python
   from .moja import PierwszeK
   METHOD_TYPES = {..., "first_k": PierwszeK}
   ```

3. Użyj w configu:

   ```toml
   [[methods]]
   type = "first_k"
   k = 25
   ```

---

## 11. Ściąga

| Chcę… | Zrób |
|---|---|
| uruchomić benchmark | `cd src && python testing.py` |
| zmienić zbiory / las / metody | edytuj [config.toml](config.toml) |
| nową tablicę IB | `from generate_table_IB import generuj_tablice; generuj_tablice(T, l, pi)` (z `src/`) |
| sprawdzić, czy IB dotrzymuje `pi` | kolumna `agree_full` i komunikat na końcu `testing.py` |
| zobaczyć, ile drzew trzeba | `trees_mean`, `trees_frac`; rozkład w `runs/*.npz` → `N_used_*` |
| wczytać wyniki zbiorcze | `pd.read_csv(f, header=[0, 1], index_col=[0, 1])` |
| nie stracić wyników | zmień `out_dir` przed każdym nowym eksperymentem |
| odtworzyć zbiory | `python src/data_generate/generate_datasets.py` (seed z `config.toml`) |
