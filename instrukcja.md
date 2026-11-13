# Instrukcja: quick_inference

Ten dokument tłumaczy, **co robi projekt, jak przeprowadzać w nim badania, w jakim formacie są dane
i wyniki oraz dlaczego kod jest napisany tak, a nie inaczej**. Krótszy opis instalacji i struktury
katalogów jest w [README.md](README.md). Tutaj chodzi o zrozumienie całości. Szczegóły samego
kodu (konwencje tablic, wyprowadzenia, nieoczywiste miejsca) są w
[implementacja.md](implementacja.md).

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
| `OWN_lam0.5_pi98` itd. | metoda własna: HYPER z a priori zmieszanym z jednostajnym, opcjonalnie z regułą GLEE (rozdz. 2e) | badana metoda |
| `IBREG_…`, `GLEEREG_…`, `OWNREG_…` | regresja: wczesny stop, gdy średnia jest już w tolerancji `delta` (rozdz. 2f); odniesieniem jest `full_averaging` | badane metody |

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

## 2d. Wiele klas: reguła pary i kolejność drzew

Te dwa ustawienia pochodzą z dokumentu projektu i dotyczą kilku metod naraz.

**`multiclass = "pair"`** (`ib_pruning`, `hyper`, `own`). Domyślne `"full"` to tablica dla
wszystkich klas naraz: dokładna, ale przy 501 drzewach wykonalna tylko dla 2–3 klas
(rozdz. 8). `"pair"` sprawdza warunek stopu dla **klasy prowadzącej i drugiej w kolejności**
na tablicach binarnych, tak jakby pozostałe głosy miały trafić tylko do tej pary. Działa
dla dowolnej liczby klas:

- IB używa tablicy binarnej `T{T}_L2_pi….npz` (jest w repozytorium dla 501 drzew),
- HYPER i metoda własna budują binarne a priori dla każdej klasy prowadzącej z próbek
  out-of-bag, w których ta klasa była w pierwszej dwójce,
- GLEE działa tak zawsze, więc nie ma tego parametru.

To przybliżenie: nie uwzględnia, że wygrać może klasa dalsza niż druga. Dlatego
`agree_full` może spaść poniżej `pi` w zbiorach z wieloma klasami o podobnej liczbie
głosów; trzeba to sprawdzać w wynikach. Nazwy dostają przyrostek `_pair`.

**`order`** (wszystkie metody z wczesnym stopem, także regresyjne):

| Wartość | Kolejność drzew | Przyrostek nazwy |
|---|---|---|
| `"fixed"` (domyślnie) | `rf.estimators_` | brak |
| `"shuffled"` | jedna losowa permutacja lasu z `seed`, wspólna dla próbek | `_shuf` |
| `"random"` | osobna permutacja dla każdej próbki (wolny batch, rozdz. 9) | `_rand` |

`"shuffled"` to kolejność opisana w dokumencie projektu. Kosztuje tyle co `"fixed"`.

---

## 2e. Metoda własna (klasyfikacja): `own`

Kod w [src/methods/own.py](src/methods/own.py), klasa `Own`. Punkt wyjścia to HYPER: jego
zaletą jest a priori z danych, a słabością wrażliwość na złe oszacowanie tego a priori
(mały zbiór, inne dane testowe niż uczące). Metoda miesza więc a priori z głosów
out-of-bag z jednostajnym:

```
P(T) = lam · P_OOB(T) + (1 − lam) · P_jednostajny(T)
```

- `lam = 1` to HYPER, `lam = 0` to IB pruning, wartości pośrednie dają kompromis między
  szybkością HYPER a ostrożnością IB,
- żaden układ głosów nie ma zerowego prawdopodobieństwa a priori,
- reszta (model urny, osobne tablice dla klas prowadzących, pętla predykcji) jest z HYPER,
  więc metoda ma te same wymagania: `X_train` i las z `bootstrap=True`.

**Wariant rozszerzony** (`glee = true`): głosowanie kończy się, gdy pozwala na to reguła
HYPER **albo** reguła GLEE. Tablice są łączone w `fit`, więc predykcja nie jest droższa.
Dwie reguły naraz to więcej okazji do błędu, więc porównuj warianty przy tym samym
`agree_full`, a nie przy tym samym `pi`; `pi_glee` ustawia osobny poziom reguły GLEE.

Parametry w configu: `pi`, `lam` (domyślnie 0.5), `glee`, `pi_glee`, `min_votes`,
`n_prior`, `smooth`, `multiclass`, `order`, `seed`.

W próbnym przebiegu (5000 próbek, 2 klasy, 501 drzew, `pi = 0.98`, 2 seedy) HYPER
odpytywał średnio 14,5 drzewa przy `agree_full` 0,981 (jeden przebieg poniżej `pi`),
IB 19,0 drzewa przy 0,996, a `own` z `lam = 0.5` 17,7 drzewa przy 0,994. To jeden mały
zbiór, nie wynik badania.

---

## 2f. Regresja: `ib_reg`, `glee_reg`, `own_reg`

Kod w [src/methods/regression.py](src/methods/regression.py) i
[src/methods/own.py](src/methods/own.py). Las to `RandomForestRegressor`, a jego predykcją
jest **średnia** wyjść drzew. Drzewa są pytane po kolei i po każdym liczona jest średnia
`y_t` oraz odchylenie standardowe `s_t` dotychczasowych wyjść. Głosowanie kończy się, gdy
z ufnością `pi` średnia wszystkich `T` drzew leży nie dalej niż `delta` od bieżącej:

```
c[t] · s_t ≤ delta
```

Metody różnią się tylko mnożnikiem `c[t]`, liczonym raz w `fit`:

| Metoda | `c[t]` | Skąd |
|---|---|---|
| `ib_reg` | `t_{t−1,(1+pi)/2} · sqrt((T − t) / (T · t))` | przedział predykcyjny t-Studenta (wariant bayesowski z nieinformatywnym a priori) |
| `glee_reg` | `z_{(1+pi)/2} · ρ_t / sqrt(t)` | przedział normalny z poprawką na populację skończoną `ρ_t` |
| `own_reg` | `t_{nu0+t−1,(1+pi)/2} · sqrt((T − t) / (T · t))`, a zamiast `s_t` rozrzut ważony z a priori | metoda własna |

**`own_reg`** przenosi pomysł HYPER na regresję: z danych out-of-bag szacuje typową
wariancję wyjść drzew dla jednej próbki (`sigma0²`, mediana po próbkach uczących)
i traktuje ją jak `nu0` dodatkowych obserwacji:

```
sigma_t² = (nu0 · sigma0² + (t − 1) · s_t²) / (nu0 + t − 1)
```

Przy małym `t` dominuje wiedza z danych uczących, potem rozrzut tej konkretnej próbki.
`nu0 = 0` to dokładnie `ib_reg`. Rozrzut nie spada do zera, więc metoda nie potrzebuje
minimalnej liczby drzew i nie staje tylko dlatego, że dwa pierwsze drzewa dały ten sam
wynik.

**Tolerancja.** Podaj dokładnie jedno z:

- `delta`: w jednostkach zmiennej celu (tak definiuje ją dokument projektu),
- `delta_std`: w odchyleniach standardowych `y_train`; wygodne, gdy jeden config obejmuje
  zbiory o różnej skali celu. Przeliczona wartość jest w metadanych przebiegu (`deltas`).

**Jak czytać wyniki.** `agree_full` to odsetek próbek, dla których predykcja różni się od
`full_averaging` o nie więcej niż `delta` tej metody, czyli odpowiednik zgodności
z pełnym głosowaniem; porównuje się go z `pi`. Model zakłada, że wyjścia drzew dla jednej
próbki są w przybliżeniu normalne i niezależne, więc `pi` nie jest gwarancją. W próbnych
przebiegach (2 zbiory syntetyczne po 4000 próbek, 501 drzew, `pi = 0.98`,
`delta_std = 0.05`) `agree_full` wyszło średnio 0,970 dla `ib_reg`, 0,978 dla `glee_reg`
i 0,980 dla `own_reg` przy ok. 160 drzewach z 501, a pojedyncze przebiegi były poniżej
`pi`.

**`t_min`.** `ib_reg`: domyślnie 2. `glee_reg`: domyślnie jak `min_votes` w GLEE (15, 30
albo 45). Z `t_min = 2` reguła normalna staje już po 2–3 drzewach, gdy ich wyjścia są
przypadkiem bliskie, i `agree_full` spada wyraźnie poniżej `pi` (0,80–0,94 w tych samych
przebiegach). `own_reg`: domyślnie 1.

Żeby uruchomić regresję, ustaw w `[benchmark]`: `task = "regression"`, `datasets` na
zbiory regresyjne i inny `out_dir` (rozdz. 5).

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
           benchmark.release()                           ← zwalnia las i dane, wyniki zostają
           suite.add(benchmark)                          ← od razu zapisuje NPZ przebiegu i CSV
   suite.summary()                                       ← tabele na ekran
   sprawdzenie: czy agree_full ≥ pi w każdym przebiegu
   suite.save()                                          ← CSV + NPZ (komplet, na koniec)
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
task = "classification"               # albo "regression": las regresyjny i metody *_reg
datasets = ["generated/binary"]       # foldery (wszystkie *.parquet) lub pojedyncze pliki, względem data/
out_dir  = "results/binary"           # względem data/; NADPISYWANY przy każdym uruchomieniu
drop_columns = ["year", "CO"]         # kolumny zbiorów, które nie są cechami
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

[[methods]]
type = "own"                          # metoda własna (rozdz. 2e)
pi = 0.98
lam = 0.5                             # 1 = HYPER, 0 = IB pruning
# glee = true                         # wariant rozszerzony: HYPER lub GLEE
# multiclass = "pair"                 # też w ib_pruning i hyper (rozdz. 2d)
# order = "shuffled"                  # też w ib_pruning, hyper, glee i metodach *_reg

[[methods]]                           # --- regresja: używane tylko przy task = "regression"
type = "full_averaging"               # średnia wszystkich drzew (odniesienie dla agree_full)

[[methods]]
type = "ib_reg"                       # albo "glee_reg", "own_reg" (rozdz. 2f)
pi = 0.98
delta_std = 0.05                      # albo delta = ... w jednostkach zmiennej celu
```

Zasady:

- `task` wybiera rodzaj lasu. Bloki `[[methods]]` drugiego zadania są pomijane, więc oba
  zestawy metod mogą stać w jednym pliku. Przełączenie na regresję to zmiana `task`,
  `datasets` i `out_dir`.
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
poza `target`** i poza wymienionymi w `benchmark.drop_columns` jako cechy, a kolumny
nieliczbowe zamienia na kody całkowite (`astype("category").cat.codes`). Zwraca `X` typu
`float` i `y`.

> Uwaga: w `gas_turbine` kolumny `year` i `CO` nie są cechami (`CO` to drugi cel). Domyślne
> `drop_columns = ["year", "CO"]` je pomija. Własne kolumny dodatkowe trzeba dopisać do tej
> listy, inaczej trafią do cech.

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
| `generated/regression/*` (9 plików), `real_data/regression/gas_turbine` | regresja | 500 – 500 000; wymagają `task = "regression"` |

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

Pliki powstają na bieżąco: po każdym przebiegu (zbiór, seed) zapisywany jest jego `.npz`,
a tabele `.csv` są pisane od nowa ze wszystkich dotychczasowych przebiegów. Przerwany
benchmark zostawia więc wyniki policzonych przebiegów (nie ma jeszcze wznawiania: ponowne
uruchomienie liczy wszystko od początku).

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

W regresji (`task = "regression"`) kolumny jakości i zgodności są inne:

| Kolumna | Znaczenie |
|---|---|
| `mae`, `rmse`, `r2` | jakość względem prawdziwych wartości |
| `agree_rf`, `agree_full` | odsetek próbek z predykcją nie dalej niż `delta` metody od `rf_sklearn` / `full_averaging`; dla metod bez `delta` odsetek predykcji równych z dokładnością arytmetyki |
| `mad_rf`, `mad_full` | średnia różnica bezwzględna względem metody odniesienia |
| `maxdiff_rf`, `maxdiff_full` | największa różnica bezwzględna |

Kolumny czasu i liczby drzew są te same. W `runs/*.npz` `classes` jest puste, `Y_pred_*`
to liczby, a metadane mają dodatkowo `task` i `deltas` (nazwa metody → tolerancja po `fit`).

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

- **Online** (jedna próbka na raz): koszt to praktycznie `liczba odpytanych drzew × stały
  koszt jednego wywołania`. Czas spada więc prawie proporcjonalnie do `trees_frac`. To
  scenariusz, dla którego IB pruning jest najbardziej naturalny (np. serwowanie modelu na
  żądanie).
- **Batch** (cały zbiór naraz): każde drzewo i tak trzeba wywołać, dopóki *jakakolwiek*
  próbka jest aktywna. Zysk bierze się z tego, że kolejne drzewa dostają coraz mniejszy podzbiór.
  Narzut wywołań i rzadkie „trudne” próbki (`trees_max = 501`) ograniczają przyspieszenie.

Liczby w tabeli pochodzą sprzed wprowadzenia `FastTree` (rozdz. 9). Każde wywołanie
`tree.predict` płaciło wtedy ok. 0,07–0,1 ms za walidację wejścia w scikit-learn, przez co
`full_voting` było w batchu ok. 2× wolniejsze od `rf_sklearn`, a w online wszystkie metody
były o rząd wielkości wolniejsze niż teraz. Proporcje `speedup_vs_full` zostają podobne
(obie strony płaciły ten sam narzut), ale `speedup_vs_rf` jest teraz wyraźnie wyższe:
`rf.predict` waliduje wejście i przechodzi przez joblib przy każdym wywołaniu, więc dla
jednej próbki kosztuje kilkanaście ms, wielokrotnie więcej niż własna pętla po drzewach.
Dlatego przyspieszenie podajemy względem obu metod odniesienia.

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

1. **Zadanie ustawia się ręcznie.** `benchmark.task` decyduje, czy powstaje
   `RandomForestClassifier`, czy `RandomForestRegressor`; kod nie zgaduje tego z danych.
   Zbiór regresyjny przy `task = "classification"` zrobi z każdej unikalnej wartości
   `target` osobną klasę. Jeden przebieg `testing.py` to jedno zadanie. HYPER i MARGIN nie
   mają wariantów regresyjnych (dokument projektu uzasadnia dlaczego).

2. **Liczba klas a tablice IB.** Rekurencja odwiedza rzędu `C(T+l, l)` stanów:

   | T | l | stany | praktycznie |
   |---|---|---|---|
   | 501 | 2 | ~126 tys. | ułamek sekundy |
   | 501 | 3 | ~21 mln | ok. 1,5 min |
   | 501 | 4 | ~2,7 mld | **niewykonalne** (pamięć cache) |
   | 51 | 4 | ~341 tys. | szybko |
   | 51 | 5 | ~3,8 mln | do zrobienia |

   Przy `T = 501` IB pruning z tablicą dla wszystkich klas działa więc tylko dla 2 i 3 klas
   (HYPER i `own` mają ten sam limit). Dla zbiorów z 4+ klasami (`multi_small`, `shuttle`,
   `multi_many_classes`, …) metoda zostanie **pominięta z ostrzeżeniem**
   (`pomijam IB_pi98: brak tablicy …`), a reszta benchmarku pójdzie dalej.
   Żeby badać więcej klas, ustaw w bloku metody `multiclass = "pair"` (rozdz. 2d) albo
   zmniejsz `n_trees`.

3. **SUSY ma 5 mln wierszy.** Zbiór testowy to 1,5 mln próbek × 501 drzew × kilka metod × kilka
   seedów. Najpierw ustaw jeden seed i sprawdź czas.

4. **Pomiar czasu jest zaszumiony.** Domyślnie `n_repeats = 1`. Do wyników końcowych zwiększ
   tę wartość i nie obciążaj komputera w trakcie. Wszystkie metody działają w jednym wątku (rozdz. 9).

5. **Parzyste T.** Przy parzystej liczbie drzew możliwy jest remis. `full_voting` wybiera wtedy
   klasę o niższym indeksie, a IB traktuje remis jako „nie wygrana”. Dla klasyfikacji binarnej
   używaj nieparzystego T (501, 51).

6. **Dwa pliki z pakietami.** `requirements.txt` to tylko pakiety potrzebne do benchmarku
   (numpy, pandas, pyarrow, scikit-learn, scipy), a `requirements-dev.txt` dokłada do nich
   ipykernel, matplotlib i seaborn (notebooki i komórki `# %%`). Przypięte są tylko te
   pakiety, zależności pośrednie dobiera `pip`. Wymagany jest Python 3.12 lub nowszy
   (przypięte numpy i scipy nie wspierają 3.11). Nie nadpisuj tych plików przez
   `pip freeze > requirements.txt`: w PowerShell daje to plik UTF-16 z całym środowiskiem.

---

## 9. Dlaczego kod wygląda tak, a nie inaczej

**Drzewa głosują indeksem klasy, nie etykietą.** `RandomForestClassifier` wewnętrznie
koduje `y` jako `0..l-1` i na tym uczy drzewa. Metody zliczają więc głosy po indeksach,
a na końcu `self.classes[votes.argmax()]` zamienia indeks z powrotem na etykietę.

**`FastTree` zamiast `tree.predict`.** `tree.predict` ze scikit-learn przy każdym wywołaniu
waliduje wejście i zamienia je na `float32`. Dla jednej próbki to ok. 95% czasu wywołania
(ok. 65 µs z 68), a metody z wczesnym stopem wołają drzewa pojedynczo, więc benchmark
mierzył głównie ten narzut. Dlatego `fit()` klasy bazowej ustawia `self.trees` na listę
`FastTree` ([src/methods/base.py](src/methods/base.py)): klasa każdego liścia jest liczona
raz (`argmax` wartości liścia, tak jak w `tree.predict`), a `tree.predict(X)` to samo
zejście po drzewie (`tree_.apply`) i odczyt z tablicy. `X` jest zamieniane na `float32`
raz na wywołanie predykcji (`self._X32(X)`, które sprawdza też liczbę cech). Wyniki są
identyczne z `tree.predict`. Lista `FastTree` jest wspólna dla wszystkich metod
dopasowanych do tego samego lasu (`fast_trees(rf)`).

**Własne głosowanie zamiast `rf.predict`.** `rf.predict` zawsze odpytuje wszystkie drzewa
i uśrednia prawdopodobieństwa. Żeby przerwać w połowie, musimy sami iterować po
`rf.estimators_`. `full_voting` robi to samo bez przerywania, więc różnica czasu
`full_voting` vs `IB` pokazuje czysty zysk z przycinania, a nie różnice implementacji.

**Kolejność drzew to po prostu `rf.estimators_`.** Model urny Polyi zakłada, że kolejne
głosy są wymienialne. Drzewa lasu losowego są uczone niezależnie, więc ich kolejność jest
losowa. Sortowanie drzew np. po trafności złamałoby to założenie, a `pi` przestałoby
oznaczać to, co ma oznaczać.
Opcja `order = "shuffled"` losuje jedną permutację lasu w `fit` i nic nie kosztuje przy
predykcji (rozdz. 2d).
Opcja `order = "random"` w bloku metody z wczesnym stopem (z opcjonalnym `seed`) daje każdej próbce
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
           X = self._X32(X)                           # float32 + kontrola kształtu, raz
           votes = np.zeros((len(X), self.l), dtype=int)
           rows = np.arange(len(X))
           for tree in self.trees[: self.k]:          # self.trees to lista FastTree
               votes[rows, tree.predict(X)] += 1      # indeks klasy, nie etykieta!
           return self.classes[votes.argmax(axis=1)], np.full(len(X), min(self.k, self.T))

       def predict_one(self, x):
           y, n = self.predict_batch(x)
           return y[0], n[0]
   ```

   Kontrakt: `predict_batch(X) -> (y_pred, n_used)` i `predict_one(x) -> (y, n_used)`,
   gdzie `n_used` to liczba odpytanych drzew. `fit(rf)` z klasy bazowej ustawia
   `self.trees` (lista `FastTree`, rozdz. 9), `self.classes`, `self.T`, `self.l`. Nadpisz go (wołając `super().fit(rf)`),
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
| badać zbiory z 4+ klasami | `multiclass = "pair"` w blokach `ib_pruning`, `hyper`, `own` |
| uruchomić regresję | `task = "regression"`, `datasets = ["generated/regression"]`, nowy `out_dir` |
| sprawdzić, czy IB dotrzymuje `pi` | kolumna `agree_full` i komunikat na końcu `testing.py` |
| zobaczyć, ile drzew trzeba | `trees_mean`, `trees_frac`; rozkład w `runs/*.npz` → `N_used_*` |
| wczytać wyniki zbiorcze | `pd.read_csv(f, header=[0, 1], index_col=[0, 1])` |
| nie stracić wyników | zmień `out_dir` przed każdym nowym eksperymentem |
| odtworzyć zbiory | `python src/data_generate/generate_datasets.py` (seed z `config.toml`) |
