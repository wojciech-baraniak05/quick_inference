# Implementacja: jak czytać kod quick_inference

Kod w `src/` ma krótkie docstringi i adnotacje typów. Ten plik zbiera to, czego z samych
nazw i typów nie widać: konwencje indeksowania tablic, uzasadnienia wzorów i miejsca,
w których kod robi coś nieoczywistego.

Podział dokumentacji:

- [README.md](README.md): instalacja, konfiguracja, uruchamianie (po angielsku),
- [instrukcja.md](instrukcja.md): co robi projekt i metody, jak prowadzić badania, formaty
  danych i wyników,
- ten plik: szczegóły kodu, w kolejności modułów.

Oznaczenia używane w kodzie i poniżej: `T` to liczba drzew lasu, `l` liczba klas, `pi`
poziom ufności, `t = (t_1, …, t_l)` głosy zebrane do tej pory, `prog` tablica progów.

---

## 1. Typy i kontrakt metody (`src/methods/base.py`)

Aliasy typów zdefiniowane na górze `base.py` i używane w całym kodzie:

| Alias | Co oznacza |
|---|---|
| `Features` | cechy w formacie drzew scikit-learn: `float32`, kształt `(n, n_cech)` |
| `IntArray` | tablica liczb całkowitych: głosy `(n, l)`, indeksy klas, liczby drzew, progi |
| `FloatArray` | tablica `float64` (a priori, marginesy, czasy) |
| `BoolArray` | maska logiczna |
| `Labels` | etykiety klas w typie `rf.classes_` (dowolny dtype) |
| `Prediction` | `(y_pred, n_used)`: etykiety i liczba odpytanych drzew dla każdej próbki |
| `Order` | `"fixed"` albo `"random"` |

Inne aliasy są przy kodzie, który ich używa: `Ranking` (`margin.py`), `Mode`
(`benchmarking/modes.py`), `MethodSpec` (`registry.py`), `RunResult`
(`benchmarking/forest_methods.py`), `Votes` (`generate_table_IB.py`).

**Kontrakt `InferenceMethod`:**

```
fit(rf, X_train=None, y_train=None) -> self
predict_batch(X) -> (y_pred, n_used)     X: (n, n_cech)
predict_one(x)   -> (y, n_used)          x: (1, n_cech), jedna próbka
```

- `n_used` to liczba drzew odpytanych dla danej próbki.
- `fit` wiąże metodę z lasem i ustawia `rf`, `trees`, `classes`, `T`, `l`, `n_features`.
  `X_train` i `y_train` (dane, na których uczono las) dostaje każda metoda, ale używają
  ich tylko te, które coś z nich szacują: `Hyper` potrzebuje `X_train`, `MarginEarlyStop`
  obu.
- `pi` jest atrybutem klasy bazowej o wartości `None`. Metody z poziomem ufności (IB,
  HYPER, GLEE) ustawiają go w konstruktorze, a benchmark po tym atrybucie rozpoznaje,
  które metody sprawdzać względem `pi`.
- Drzewa głosują **indeksem klasy** (`0..l-1`), nie etykietą: `RandomForestClassifier`
  koduje `y` wewnętrznie. Zamiana na etykietę to `self.classes[votes.argmax()]` na końcu
  predykcji. Remis rozstrzyga `argmax`, czyli wygrywa klasa o niższym indeksie.

**Dwie klasy pośrednie**, żeby nowa metoda nie pisała własnych pętli:

- `FixedVoting`: hard voting stałej listy drzew `self.voters`, tej samej dla każdej
  próbki. Po `fit` są to wszystkie drzewa (`FullVoting`). Podklasa może w swoim `fit`
  wybrać podzbiór i kolejność (`MarginEarlyStop`).
- `SequentialPruning`: drzewa głosują po kolei, a próbka kończy, gdy reguła stopu uzna
  wynik za pewny. Podklasa podaje tylko `_stop_batch(votes) -> maska (n,)` dla `votes`
  o kształcie `(n, l)`. Opcjonalnie może podać szybszą wersję skalarną `_stop_one(votes)`
  dla `votes` o kształcie `(l,)`; domyślna woła `_stop_batch` na jednym wierszu.

Żeby metodę dało się wybrać w `config.toml`, trzeba ją dopisać do `METHOD_TYPES`
w `src/methods/registry.py` (rozdz. 10).

---

## 2. `FastTree`: drzewo bez narzutu scikit-learn

`tree.predict` ze scikit-learn przy każdym wywołaniu waliduje wejście i zamienia je na
`float32`. Dla jednej próbki to ok. 65 µs z 68 µs całego wywołania, a metody z wczesnym
stopem wołają drzewa pojedynczo, więc bez `FastTree` benchmark mierzył głównie tę
walidację.

`FastTree(estimator)`:

- `leaf_class[liść]` to klasa liścia, liczona raz jako `argmax` wartości liścia
  (`tree_.value`). To ten sam `argmax` co w `tree.predict`, więc remisy w liściu
  rozstrzygają się identycznie (niższy indeks klasy),
- `predict(X)` to `leaf_class[tree_.apply(X)]`: zejście po drzewie i odczyt z tablicy.

Skutki dla reszty kodu:

- `X` trzeba raz zamienić na `float32` (`to_float32`). `tree_.apply` odrzuca inne typy.
- `FastTree` nie sprawdza kształtu. Robi to `InferenceMethod._X32(X)`, wołane raz na
  początku `predict_batch` i `predict_one`.
- `fast_trees(rf)` pamięta listę `FastTree` dla lasu (słownik ze słabymi referencjami,
  więc nie przedłuża życia lasu). Wszystkie metody dopasowane do tego samego lasu dzielą
  jedną listę. Wpis zawiera też `rf.estimators_`: gdy las zostanie nauczony ponownie,
  lista jest budowana od nowa.
- Obsługiwane są tylko drzewa z jednym wyjściem (`n_outputs = 1`).

`SklearnRF` (`rf_sklearn`) celowo nie używa `FastTree`: to pomiar oryginalnego
`rf.predict`.

---

## 3. Pętle wczesnego stopu (`SequentialPruning`)

### 3.1. Batch, kolejność `fixed`

```
aktywne = indeksy próbek jeszcze niezdecydowanych
X_akt   = ich cechy (X[aktywne])
dla każdego drzewa k:
    dodaj głosy drzewa dla aktywnych próbek
    stop = reguła stopu dla ich głosów
    n_used[te, które stanęły] = k + 1
    usuń je z aktywne i z X_akt
```

`n_used` startuje od `T`, więc próbki, które nie stanęły do końca, mają `T`. `X_akt` jest
kopiowane tylko wtedy, gdy jakaś próbka odpadła, a nie przy każdym drzewie.

W batchu każde drzewo trzeba wywołać, dopóki choć jedna próbka jest aktywna. Zysk bierze
się z tego, że kolejne drzewa dostają coraz mniej próbek.

### 3.2. Batch, kolejność `random`

Każda próbka dostaje własną losową permutację drzew: `perm[i, k]` to drzewo, które próbka
`i` pyta w kroku `k`. W jednym kroku różne próbki pytają różne drzewa, więc nie da się
ich obsłużyć jednym wywołaniem. Kod sortuje aktywne próbki po numerze drzewa i dzieli je
na grupy pytające to samo drzewo; każda grupa to jedno wywołanie `predict`.

Koszty, o których warto wiedzieć:

- liczba wywołań drzew na krok rośnie do `T` (zamiast 1), więc ten tryb jest wyraźnie
  wolniejszy od `fixed`,
- macierz `perm` ma `n × T` elementów (typ dobrany do `T`, dla 501 drzew 2 bajty na
  element) i jest kopiowana przy losowaniu: ok. 2 kB na próbkę przy 501 drzewach.

### 3.3. Online

`predict_one` pyta drzewa po kolei i po każdym głosie woła `_stop_one`. Przy
`order = "random"` losuje jedną permutację drzew na próbkę.

### 3.4. Generator losowy

`self.rng` jest tworzony od nowa w `fit()` z `seed`. Jest jeden na metodę i zużywany
przez kolejne wywołania predykcji, więc wyniki w kolejności `random` zależą od tego, ile
predykcji wykonano wcześniej (np. od kolejności scenariuszy i od `n_repeats`).

---

## 4. IB pruning: tablica progów (`src/generate_table_IB.py`, `src/methods/ib_pruning.py`)

Model i rekurencja są opisane w [instrukcja.md, rozdz. 2](instrukcja.md). Tutaj to, co
dotyczy kodu.

### 4.1. `wygrana(t, T)` i `p_same(t, T)`

`wygrana(t, T)` to prawdopodobieństwo, że klasa na pozycji 0 **ściśle** wygra po `T`
głosach, jeśli dotychczasowe głosy to `t`. W modelu urny Polyi następny głos trafia do
klasy `i` z prawdopodobieństwem `(t[i] + 1) / (sum(t) + l)`, więc wynik dla stanu `t`
składa się z wyników stanów o jeden głos dalej. `lru_cache` sprawia, że każdy stan jest
liczony raz.

`p_same(t, T)` to wzór (8) z artykułu: prawdopodobieństwo, że klasa prowadząca w `t`
wygra też pełne głosowanie. Sortuje `t` malejąco, żeby klasa prowadząca była na pozycji
0. Wynik nie zależy od kolejności klas, a sortowanie zmniejsza liczbę stanów w cache.

Dwie rzeczy techniczne:

- głębokość rekurencji to ok. `T`, dlatego `p_same` podnosi `sys.setrecursionlimit`,
- cache trzyma wszystkie odwiedzone stany (dla `T = 501`, `l = 3` ok. 21 mln, szacunkowo
  kilka GB pamięci), dlatego `generuj_tablice` na końcu woła `cache_clear()`.

### 4.2. Dlaczego tablica ma `l - 1` wymiarów

Przy ustalonych głosach pozostałych klas `P(t, T)` rośnie z liczbą głosów klasy
prowadzącej. Zamiast pełnej tablicy `(T+1)^l` wartości prawda/fałsz wystarczy próg:

```
prog[t_2, ..., t_l] = najmniejsze t_1, dla którego P(t, T) >= pi
                      (T + 1, jeśli przy tych t_2..t_l nie da się przerwać)
```

gdzie `t_1 >= t_2 >= ... >= t_l` to głosy **posortowane malejąco**. `generuj_tablice`
wypełnia tylko komórki o indeksach posortowanych malejąco; pozostałe zostają na `T + 1`
i nigdy nie są czytane. Dlatego przed odczytem głosy trzeba posortować:

- `czy_stop(glosy, prog)`: jedna próbka, zwykłe `sorted`,
- `czy_stop_wektorowo(votes, prog)`: wiele próbek naraz, `votes` o kształcie `(n, l)`.

Typ tablicy to najmniejszy typ całkowity mieszczący `T + 1` (`np.min_scalar_type`).

### 4.3. Skąd `IBPruning` bierze tablicę

```python
IBPruning(pi=0.98)                    # plik T{T}_L{l}_pi98.npz dobierany w fit() do lasu
IBPruning(table="T501_L2_pi98.npz")   # konkretny plik (ścieżka względem data/tablice/)
IBPruning(prog, pi=0.98)              # gotowa tablica
```

W pierwszym wariancie `T` i `l` są znane dopiero w `fit()`, więc tam jest szukany plik;
gdy go nie ma, komunikat błędu podaje gotowe wywołanie `generuj_tablice`. `fit()` zawsze
sprawdza, że tablica pasuje do lasu (`T` z rozmiaru pierwszego wymiaru, `l` z liczby
wymiarów).

Pliki są wczytywane raz (`_wczytaj_npz` z `lru_cache`), więc kolejne lasy o tym samym
rozmiarze dzielą tę samą tablicę w pamięci. Nie należy jej modyfikować.

`pi_tag(pi)` zamienia poziom ufności na tekst do nazw plików i metod: `0.98 -> "98"`,
`0.995 -> "99.5"`. Używają go też HYPER i GLEE do domyślnych nazw.

---

## 5. HYPER (`src/methods/hyper.py`)

Soto, Suárez, Martínez-Muñoz, *An urn model for majority voting in classification
ensembles*, NIPS 2016. Opis metody i różnic względem IB:
[instrukcja.md, rozdz. 2a](instrukcja.md).

### 5.1. Model

Las dla jednej próbki to urna z `T` kulami (drzewa) w `l` kolorach (klasy). Skład urny
`T = (T_1..T_l)` jest nieznany, a odpytywanie drzew to losowanie bez zwracania. Po
zebraniu głosów `t` rozkład a posteriori składu (wzór (2) artykułu) to

```
P(T | t)  ~  C(T_1, t_1) * ... * C(T_l, t_l) * P(T)
```

`P*(t, T)` (wzór (3)) to suma `P(T | t)` po składach, w których wygrywa ta sama klasa co
w `t`. Głosowanie przerywamy, gdy `P*(t, T) >= pi`. IB pruning to ten sam model
z jednostajnym `P(T)`. Tutaj `P(T)` jest szacowane z głosów out-of-bag, więc tablica
progów zależy od danych i jest liczona w `fit()`, a nie czytana z pliku.

### 5.2. `prior_z_glosow(votes, T, smooth)`: a priori z głosów

Rozdz. 2.2 artykułu. Wynik to tablica o kształcie `(T+1,)^(l-1)`:
`prior[T_1, ..., T_{l-1}]`, gdzie `T_l = T - (T_1 + ... + T_{l-1})`. Komórki o sumie
większej niż `T` mają 0.

Kroki:

1. Próbki bez żadnego głosu out-of-bag są pomijane.
2. Głosy każdej próbki są skalowane do zespołu `T` drzew: `T_i ~ T * votes_i / sum(votes)`.
   Zaokrąglenie jest metodą **największych reszt**: najpierw `floor`, potem brakujące
   głosy (od 0 do `l - 1`) dostają klasy o największej części ułamkowej. Dzięki temu suma
   wychodzi dokładnie `T`.
3. Z przeskalowanych składów powstaje histogram.
4. Histogram jest wygładzany oknem przesuwnym o szerokości `smooth` (artykuł: 5; 0 lub 1
   wyłącza wygładzanie). Filtr liczy sumy kroczące i potrafi zostawić wartości rzędu
   `-1e-20` zamiast 0. Ujemne „prawdopodobieństwo” dawało `nan` w całej tablicy progów,
   stąd `np.clip(..., 0, None)`.
5. Składy niemożliwe (suma > `T`) są zerowane, a całość normalizowana do sumy 1.

### 5.3. `tablica_hyper(prior, pi)`: tablica progów

```
prog[c, <głosy pozostałych klas w kolejności klas>] =
    najmniejsza liczba głosów klasy prowadzącej c, od której P*(t, T) >= pi
```

A priori nie jest symetryczne, więc klasy nie są wymienne i **każda klasa prowadząca ma
własną tablicę** (pierwszy wymiar `prog`). Głosy pozostałych klas są w kolejności klas,
bez sortowania.

Sposób liczenia. Niech

- `h(t)` = suma po składach `T` z `P(T) * P(konkretna kolejność głosów o liczbach t | T)`,
- `g_c(t)` = ta sama suma, ale tylko po składach, w których wygrywa klasa `c`.

Wtedy `P*(t, T) = g_c(t) / h(t)` dla `c` = klasa prowadząca w `t`. Obie funkcje spełniają
tę samą rekurencję, bo następny głos trafia do którejś klasy:

```
f(t) = suma_i f(t + e_i)
```

z warunkiem brzegowym dla kompletu głosów: `f(T) = P(T) / (T! / (T_1! ... T_l!))`.
Kod idzie więc warstwami od `T` głosów w dół; stanów jest `C(T + l, l)`.

Jak to wygląda w kodzie:

- `G` ma kształt `(l + 1, *prior.shape)`: `G[c]` to `g_c`, a `G[l]` to `h`. Stan jest
  indeksowany głosami pierwszych `l - 1` klas; głosy ostatniej klasy wynikają z numeru
  warstwy `s` (`s - suma`).
- Warstwa `T` jest liczona w logarytmach (`gammaln`), żeby silnie się nie przepełniły.
- Krok rekurencji to dodanie przesuniętych kopii `G`: głos dla klasy `i` przesuwa indeks
  `i` o jeden, a głos dla ostatniej klasy zostawia indeksy bez zmian (stąd `nowe = G.copy()`).
- Po każdej warstwie `G` jest dzielone przez `max(h)`. Liczy się tylko iloraz `g_c / h`,
  a skalowanie chroni przed nadmiarem i niedomiarem.
- Stan jest „zły”, jeśli `P* < pi`. Wyjątek: gdy przewaga lidera jest większa niż liczba
  pozostałych głosów (`pewne`), wynik jest przesądzony niezależnie od a priori.
- **`P*` nie musi być monotoniczne** w liczbie głosów lidera. Próg jest więc ustawiany za
  ostatnim złym stanem (`np.maximum.at(prog, ..., top + 1)`), dzięki czemu każde
  przerwanie spełnia `P* >= pi`.
- Stany niemożliwe według a priori (`h = 0`) są złe, czyli nie przerywają głosowania.

Remis na końcu głosowania jest liczony tak, jak rozstrzyga go `full_voting` (`argmax`,
niższy indeks klasy). W IB pruning remis to przegrana, dlatego przy jednostajnym a priori
tablica HYPER nie jest identyczna z tablicą IB w komórkach bliskich progu.

### 5.4. Odczyt i ograniczenia

- `_lider(votes)` zwraca dla każdej próbki indeks klasy prowadzącej (remis: niższy
  indeks), jej głosy i głosy pozostałych klas w kolejności klas. Te trzy rzeczy są
  indeksem do `prog`.
- `MAX_CELLS = 2 000 000` ogranicza rozmiar a priori `(T+1)^(l-1)`. Dla 501 drzew
  przechodzą 2 i 3 klasy; przy większej liczbie `fit` zgłasza `ValueError`, a benchmark
  pomija metodę dla tego zbioru.
- `Hyper(n_prior=...)`: ile losowych próbek uczących użyć do głosów out-of-bag
  (`None` = wszystkie).
- `expected_trees(n_sim, seed)`: oczekiwana liczba odpytanych drzew, oszacowana metodą
  Monte Carlo bez danych testowych. Losuje skład urny z `prior`, ustawia głosy w losowej
  kolejności i liczy je do pierwszego stopu.

---

## 6. GLEE (`src/methods/glee.py`)

Basilico, Munson, Kolda, Dixon, Kegelmeyer, *COMET: A Recipe for Learning and Using Large
Ensembles on Massive Data* (ICDM 2011, arXiv:1103.2068), rozdz. II-C. Opis metody:
[instrukcja.md, rozdz. 2b](instrukcja.md).

### 6.1. Reguła

Głos drzewa to zmienna Bernoulliego, więc odsetek głosów klasy prowadzącej `p^` po `n`
głosach ma w przybliżeniu rozkład normalny. Głosowanie przerywamy, gdy dolna granica
jednostronnego przedziału ufności `1 - alfa` leży powyżej 0.5:

```
p^ - rho * delta > 0.5
delta = z_alfa * sqrt(p^ * (1 - p^) / n)
rho   = sqrt((m - n) / (m - 1))   jeśli n > 0.05 * m,   inaczej 1
```

`m` to liczba drzew. `rho` to poprawka na populację skończoną (FPC): drzewa są losowane
bez zwracania, więc niepewność maleje, gdy zostaje ich mało.

W artykule `alfa` to największa dopuszczalna częstość niezgodności z pełnym głosowaniem.
W kodzie podaje się `pi = 1 - alfa`, tak jak w IB i HYPER. `alfa` jest zaokrąglane do 12
miejsc, żeby błąd arytmetyki zmiennoprzecinkowej (`1 - 0.99 = 0.010000000000000009`) nie
wpływał na wybór `min_glosow` ani na kwantyl.

Dla wielu klas (przypis 1 w artykule) klasa prowadząca jest traktowana jak klasa 1,
a druga w kolejności jak klasa 0: `p^ = t_1 / (t_1 + t_2)`, `n = t_1 + t_2`.

### 6.2. `tablica_glee(T, alfa, min_votes, fpc, one_sided)`

```
prog[t_2] = najmniejsze t_1, przy którym p^ - rho * delta > 0.5
            (T + 1, jeśli przy tym t_2 nie da się przerwać)
```

Przy ustalonym `t_2` dolna granica przedziału rośnie z `t_1`, więc próg wystarcza do
decyzji. Tablica jest jednowymiarowa, ma `T // 2 + 1` elementów (druga klasa nie może
mieć więcej niż połowę głosów) i jest liczona w `fit()` w czasie `O(T^2)`. Nie zależy od
liczby klas ani od danych.

- `min_votes`: przybliżenie normalne jest słabe dla małych `n`, więc stop jest dozwolony
  dopiero po tylu głosach (lidera i drugiej klasy łącznie). `None` oznacza wartość
  z artykułu (`min_glosow`): 15, 30 i 45 dla `alfa` = 1e-2, 1e-3 i 1e-4; dla pośrednich
  `alfa` brana jest wartość ostrzejsza.
- `fpc=False`: bez poprawki na populację skończoną (warianty G1, G2 z artykułu).
- `one_sided=False`: przedział dwustronny, `z_{alfa/2}` (warianty G2, G2-FPC).
- Ustawienia domyślne to G1-FPC, czyli reguła, którą artykuł nazywa GLEE.
- Dla `alfa = 0` (`pi = 1`) kwantyl jest nieskończony i w obliczeniach pojawia się `nan`.
  Porównanie z `nan` daje fałsz, więc taka metoda nigdy nie przerywa.

Odczyt: `czy_stop_glee` używa `np.partition`, żeby wyciągnąć dwie największe liczby
głosów bez pełnego sortowania.

Z COMET zaimplementowana jest tylko ta reguła. Uczenie (MapReduce, IVoting) dotyczy
budowy zespołu, a nie inferencji na gotowym lesie.

---

## 7. Margines (`src/methods/margin.py`)

Barbareschi, Emmanuele, *A Margin Based Early-Stopping Approach for Random Forest
Classifiers* (DSN-W 2025). Opis metody: [instrukcja.md, rozdz. 2c](instrukcja.md).

### 7.1. Metoda

Margines próbki `x` (rozdz. II-B artykułu):

```
M_x = (v_c - v_c*) / N
```

`v_c` to głosy klasy prawdziwej, `v_c*` głosy najczęściej wybieranej innej klasy, `N`
liczba głosujących drzew. `M_x > 0` oznacza poprawną klasyfikację.

Faza offline (rozdz. III-A), na zbiorze kalibracyjnym z etykietami:

1. ranking drzew od najbardziej do najmniej istotnego,
2. średnia `mu` i odchylenie standardowe `sigma` marginesu próbek poprawnie
   sklasyfikowanych, a z nich `N* = N * (mu + k * sigma)`. Większe `k` to więcej drzew,
   mniejsze to krótsza inferencja.

Faza po wdrożeniu (rozdz. III-B): każdą próbkę ocenia `N*` pierwszych drzew rankingu,
wynik to głosowanie większościowe tych drzew. Liczba drzew jest stała dla wszystkich
próbek, dlatego klasa dziedziczy po `FixedVoting`, a nie po `SequentialPruning`.

### 7.2. Zbiór kalibracyjny: `predykcje_kalibracyjne`

W artykule zbiór kalibracyjny to połowa zbioru testowego. Tutaj są to próbki uczące
oceniane out-of-bag (rozdz. 8), więc cały zbiór testowy zostaje do pomiaru, tak jak dla
pozostałych metod. Skutek: margines jest dzielony przez liczbę drzew, które na próbkę
głosowały (ok. 37% lasu), a nie przez `N`.

Funkcja zwraca `(pred, y)`:

- `pred[i, n]` to indeks klasy, na którą drzewo `i` głosuje dla próbki `n`, albo `-1`,
  jeśli drzewo widziało tę próbkę przy uczeniu (nie głosuje). Typ `int16`, bo macierz ma
  `T × n_calib` elementów.
- `y[n]` to indeks klasy prawdziwej.
- Zostają tylko próbki z co najmniej dwoma głosami: margines po usunięciu jednego drzewa
  musi być określony.
- `n_calib` ogranicza liczbę próbek (`None` = wszystkie).

### 7.3. Ranking drzew

`ranking_drzew(pred, y, l, ranking)` zwraca indeksy drzew od najważniejszego. Remisy
rozstrzyga kolejność lasu (sortowanie stabilne).

- `"mean_d_m"`: malejąco według `spadek_marginesu`. To kryterium MeanD-M (Yang i in.,
  2012) użyte w eksperymentach artykułu: o ile średnio spada margines próbek, gdy to
  jedno drzewo usuniemy z zespołu. Średnia jest po próbkach, na które drzewo głosuje;
  drzewo bez głosów dostaje `-inf`.
- `"accuracy"`: malejąco według trafności pojedynczego drzewa (`trafnosc_drzew`).
- `"none"`: kolejność lasu.

**Jak `spadek_marginesu` unika pętli po parach (drzewo, próbka).** Skutek usunięcia
drzewa zależy tylko od klasy, na którą głosowało. Wystarczy więc policzyć
`spadek[n, c]` = spadek marginesu próbki `n` po odjęciu jednego głosu od klasy `c`
(`l` przebiegów po próbkach), a potem dla każdego drzewa uśrednić `spadek[n, pred[i, n]]`
po jego próbkach. Dla par, w których nikt nie głosował na `c`, wartość jest bez sensu,
ale nigdy nie jest czytana.

Ranking jest liczony raz, względem pełnego lasu. Wariant iteracyjny (ponowna ocena po
każdym usunięciu) nie jest zaimplementowany.

### 7.4. Liczba drzew

- `statystyki_marginesu(votes, y)`: „poprawnie sklasyfikowana” znaczy, że `argmax` głosów
  (tak jak w `full_voting`) wskazuje klasę prawdziwą.
- `liczba_drzew(T, mu, sigma, k)`: artykuł nie podaje zaokrąglenia; kod zaokrągla do
  najbliższej liczby całkowitej i ogranicza do `[1, T]`.
- `set_k(k)` zmienia `k` i `n_star` bez ponownej kalibracji (artykuł: strojenie `k` po
  wdrożeniu). Nazwa metody w wynikach zostaje taka, jak przy utworzeniu.

Po `fit`: `order` (ranking), `mu`, `sigma`, `n_star`, `voters` (pierwsze `n_star` drzew
rankingu).

---

## 8. Predykcje out-of-bag (`src/methods/oob.py`)

Każdą próbkę uczącą oceniają tylko drzewa, które jej nie widziały przy uczeniu. Daje to
nieobciążone głosy lasu bez osobnego zbioru walidacyjnego. Korzystają z tego `Hyper`
(głosy) i `MarginEarlyStop` (predykcje każdego drzewa osobno).

- `probka_uczaca(rf, X_train, n_max, seed)` losuje co najwyżej `n_max` wierszy i przy
  okazji sprawdza dwie rzeczy: że las był uczony z `bootstrap=True` oraz że `X_train` ma
  tyle wierszy, na ilu uczono las. Druga kontrola łapie pomyłkowe podanie innego zbioru.
  `seed` jest domyślnie 0 w obu metodach, więc przy tym samym `n` `Hyper`
  i `MarginEarlyStop` dostają te same wiersze.
- `predykcje_oob(rf, X_train, wiersze)` jest generatorem po drzewach. Dla każdego drzewa
  zwraca `oob` (pozycje **w `wiersze`**, nie w `X_train`, których drzewo nie widziało)
  i przewidziane dla nich indeksy klas.
- `glosy_oob` sumuje te predykcje do macierzy głosów `(n, l)`.

Scikit-learn nie przechowuje indeksów bootstrapu, tylko odtwarza je z ziarna każdego
drzewa. Kod używa prywatnego `rf._get_estimators_indices()` (generator, nie trzyma
wszystkich indeksów naraz), a gdy go nie ma, publicznego `rf.estimators_samples_`.
Liczba wierszy uczących jest czytana z prywatnego `rf._n_samples`. To miejsca, które mogą
wymagać poprawki po zmianie wersji scikit-learn.

---

## 9. Benchmark (`src/benchmarking/`)

### 9.1. `methods_rf`: metody jednego lasu

Po `run(X, mode)` wyniki są w macierzach, w których wiersz `i` odpowiada metodzie
`names[i]`:

```
Y_pred[mode]   (n_metod, n_próbek)   predykcje
N_used[mode]   (n_metod, n_próbek)   liczba odpytanych drzew
times[mode]    (n_metod,)            czas predykcji w s (mediana z n_repeats)
```

- `add_method` od razu woła `fit(rf, X_train, y_train)`. Wyjątek z `fit` (np. brak
  tablicy IB dla tej liczby klas) oznacza, że metoda nie pasuje do lasu; `testing.py`
  łapie go i pomija metodę dla tego zbioru.
- W trybie `online` każda próbka idzie osobnym wywołaniem `predict_one(x[None, :])`,
  a mierzony czas obejmuje całą pętlę.

### 9.2. `train_forest`

Las jest uczony na wszystkich rdzeniach (`n_jobs=-1`), a potem przestawiany na
`n_jobs=None`. Dzięki temu `rf.predict` działa w jednym wątku i porównanie czasu
z metodami, które odpytują drzewa po kolei, jest uczciwe.

### 9.3. `benchmark`: jeden las, jeden zbiór testowy

- Tryb `batch` używa całego `X_test`, tryb `online` pierwszych `n_online` próbek.
- Kolumny porównania z metodą odniesienia (`agree_rf`, `speedup_vs_rf`, `agree_full`,
  `speedup_vs_full`) są w wynikach tylko wtedy, gdy ta metoda jest w zbiorze, np. bez
  `rf_sklearn` przy `sklearn_soft_voting = false`.
- `benchmark` trzyma własne kopie tego, czego potrzebują wyniki: `names`, `T`, `classes`,
  `pi_levels` (nazwa metody → `pi`). Są odświeżane na początku `run()`, bo metody mogły
  dojść po utworzeniu benchmarku.
- `release()` zwalnia las, dane uczące i `X_test`. `results()`, `summary()` i `save()`
  działają dalej, `run()` zgłasza błąd. Ma to znaczenie, gdy wiele benchmarków jest
  trzymanych naraz w `benchmark_suite`: bez tego każdy trzymałby swój las i dane do końca
  przebiegu.
- `results(mode)` liczy metryki raz na scenariusz i zwraca kopię zapamiętanej tabeli.
  Ponowne `run()` unieważnia zapamiętaną tabelę.

### 9.4. `benchmark_suite`: wiele zbiorów i powtórzeń

- Każdy benchmark jest identyfikowany wartościami kluczy z metadanych (domyślnie
  `("dataset", "seed")`). **Ostatni klucz to powtórzenie**, po którym uśrednia
  `aggregate()`.
- `modes` zwraca tylko scenariusze, które mają wyniki we wszystkich benchmarkach.
- Z `out_dir` w konstruktorze każde `add()` zapisuje `.npz` przebiegu i pisze od nowa
  tabele `.csv`. Tabele idą przez plik tymczasowy i podmianę, więc przerwanie w trakcie
  zapisu nie zostawia uciętego pliku. Błąd zapisu tabel (np. plik otwarty w Excelu) nie
  przerywa benchmarku: wyniki przebiegu są już w `.npz`, a tabele zostaną zapisane przy
  następnym `add()` albo w `save()`.
- Nazwa pliku przebiegu składa się z kluczy: wartość `dataset` bez przedrostka, pozostałe
  jako `<klucz><wartość>`, np. `binary_small_seed1.npz`.
- Nie ma wznawiania: ponowne uruchomienie liczy wszystko od początku.

Formaty plików wynikowych: [instrukcja.md, rozdz. 6.3](instrukcja.md).

---

## 10. Rejestr metod i konfiguracja

`src/methods/registry.py` zamienia bloki `[[methods]]` z `config.toml` na obiekty:

```python
{"type": "ib_pruning", "pi": 0.98}      # -> IBPruning(pi=0.98)
{"type": "margin", "k": 0.0}            # -> MarginEarlyStop(k=0.0)
```

- `type` wybiera klasę z `METHOD_TYPES`, `enabled = false` wyłącza blok, a wszystkie
  pozostałe klucze są argumentami konstruktora. Zły argument daje `ValueError`
  z komunikatem konstruktora.
- `rf_sklearn` nie jest w `METHOD_TYPES`: włącza się przez
  `benchmark.sklearn_soft_voting` i wtedy trafia na początek listy metod.
- `build_methods` zwraca **nowe, niedopasowane** instancje. `testing.py` woła je dla
  każdego lasu, bo `fit` dobiera metodę do konkretnego lasu.
- Nazwy metod muszą być różne; dwa bloki o tych samych parametrach wymagają różnych
  `name`.

`src/config.py` czyta `config.toml` przy imporcie i wystawia wartości jako stałe modułu.
Brak klucza w pliku daje `KeyError` przy imporcie. Brak `max_depth` w `[forest]` jest
dozwolony i oznacza drzewa bez limitu głębokości.
