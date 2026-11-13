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
poziom ufności, `t = (t_1, …, t_l)` głosy zebrane do tej pory, `prog` tablica progów,
a w regresji `delta` tolerancja i `t` liczba odpytanych drzew.

Rozdziały 1–10 opisują klasyfikację. Regresja (las `RandomForestRegressor`) jest
w rozdz. 11, metody własne projektu w rozdz. 12.

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
| `Order` | `"fixed"`, `"shuffled"` albo `"random"` (rozdz. 3) |
| `Multiclass` | `"full"` albo `"pair"`: tablica dla wszystkich klas albo dla pary lider – druga klasa |
| `Task` | `"classification"` albo `"regression"` |
| `Forest` | `RandomForestClassifier` albo `RandomForestRegressor` |

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
- `task` (`"classification"`, w klasach z rozdz. 11 `"regression"`) mówi, do jakiego lasu
  metoda pasuje. `fit` metody klasyfikacyjnej na lesie regresyjnym (i odwrotnie) zgłasza
  `ValueError`. `delta` to tolerancja metod regresyjnych, `None` dla pozostałych.
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

`FastRegTree` to to samo dla drzewa regresyjnego: `leaf_value[liść]` jest wyjściem drzewa
(`tree_.value[:, 0, 0]`), a `predict(X)` zwraca wartości `float64`. `fast_reg_trees(rf)`
korzysta z tego samego słownika co `fast_trees(rf)`.

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

### 3.5. Kolejność `shuffled`

Jedna losowa permutacja lasu, wylosowana w `fit()` z `seed` i wspólna dla wszystkich
próbek: `self.trees` staje się nową listą w tej kolejności (lista z `fast_trees(rf)`,
dzielona z innymi metodami, zostaje bez zmian). Predykcja idzie potem tymi samymi pętlami
co `fixed`, więc nie ma kosztów z rozdz. 3.2, a wynik nie zależy od liczby wcześniejszych
predykcji. To kolejność z dokumentu projektu („losowa permutacja zespołu przy ustalonym
ziarnie”). Kalibracja na danych out-of-bag (HYPER, metody własne) nie zależy od kolejności.

Kolejność wybiera parametr `order` (z opcjonalnym `seed`) w IB, HYPER, GLEE, metodach
własnych i metodach regresyjnych. Domyślne nazwy dostają przyrostek `_shuf` albo `_rand`
(`ORDER_SUFFIX`).

### 3.6. Reguła dla pary lider – druga klasa

Dokument projektu rozszerza metody na wiele klas przez porównanie klasy prowadzącej
z drugą w kolejności na **tablicach binarnych**: warunek sprawdza parę `(t_1, t_2)`
dwóch największych liczb głosów, a tablica zakłada, że pozostałe `T - t_1 - t_2` głosów
trafi do tej pary (nie mniej, niż zostało naprawdę). `czy_stop_para(votes, prog)` w
`base.py` czyta `prog[t_2]` = najmniejsze `t_1`, przy którym wolno przerwać. To
przybliżenie: nie uwzględnia, że wygrać może jeszcze klasa dalsza niż druga.

- GLEE działa tak od początku (rozdz. 6.1),
- IB pruning i HYPER (oraz metoda własna) mają parametr `multiclass`: `"full"` (domyślnie)
  to dotychczasowa tablica dla wszystkich klas naraz, `"pair"` to reguła pary (rozdz. 4.4
  i 5.5). Domyślne nazwy dostają przyrostek `_pair`.

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

### 4.4. `multiclass = "pair"`

`IBPruning(pi=0.98, multiclass="pair")` przy każdej liczbie klas wczytuje tablicę binarną
`T{T}_L2_pi98.npz` i stosuje regułę pary z rozdz. 3.6. Tablica binarna ma już potrzebny
format (`prog[t_2]` = najmniejsze `t_1`), więc nie trzeba liczyć nic nowego, a metoda
działa dla zbiorów, dla których tablica `l`-klasowa jest niewykonalna (4 i więcej klas
przy 501 drzewach). Dla dwóch klas `"pair"` i `"full"` dają to samo.

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
  kolejności i liczy je do pierwszego stopu. Tylko dla `multiclass = "full"`.
- `Hyper._prior(prior)` i `Hyper._prog(prog)` to miejsca, w których podklasa zmienia
  a priori i tablicę (metoda własna, rozdz. 12). W `Hyper` nic nie robią.

### 5.5. `multiclass = "pair"`: tablice binarne dla każdej klasy prowadzącej

Reguła pary z rozdz. 3.6 wymaga binarnego a priori. Artykuł go nie definiuje dla wielu
klas, więc poniższe jest założeniem projektu.

`prior_par(votes, T, smooth)` zwraca tablicę `(l, T + 1)`: `prior[c, T_c]` to rozkład
liczby głosów klasy `c` w urnie złożonej tylko z niej i jej rywala (`T_c + T_rywala = T`).

1. Dla każdej próbki out-of-bag brane są dwie klasy o największej liczbie głosów,
   a ich głosy są skalowane do `T` tą samą metodą największych reszt co w rozdz. 5.2
   (`sklad_urny`).
2. Próbka trafia do histogramu **obu** klas pary: lidera z jego `T_c` i drugiej klasy z jej
   `T_c`. Przy więcej niż dwóch klasach druga klasa bez żadnego głosu jest wybrana
   przypadkiem (remis zer), więc wtedy do histogramu trafia tylko lider.
3. Każdy wiersz jest wygładzany oknem `smooth` i normalizowany. Klasa, która nigdy nie
   była w pierwszej dwójce, dostaje rozkład jednostajny (czyli regułę IB).

`tablica_par(prior, pi)` liczy dla każdej klasy tablicę binarną przez `tablica_hyper`
i zwraca `prog[c, t_2]` = najmniejsza liczba głosów lidera `c` przy `t_2` głosach drugiej
klasy. Dwa szczegóły:

- klasa `c` jest podawana do `tablica_hyper` jako **druga** klasa urny binarnej
  (`p[::-1]`, potem `[1]`), więc remis w pełnym zespole liczy się jako jej przegrana.
  Zgadza się to z definicją z dokumentu („ściśle więcej głosów”) i jest ostrożne przy
  parzystym `T`,
- próg jest podnoszony do co najmniej `t_2 + 1`: przy remisie w głosach lider nie jest
  wyznaczony, więc stop wymaga ścisłej przewagi.

Odczyt (`czy_stop_hyper_para`): lider to `argmax` głosów, `t_2` to druga największa liczba
głosów. Dla dwóch klas i nieparzystego `T` wynik jest identyczny z `multiclass = "full"`.
Nie ma tu limitu `MAX_CELLS`: koszt to `l` tablic binarnych.

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
- Dla lasu regresyjnego `predykcje_oob` zwraca wyjścia drzew zamiast indeksów klas,
  a `wariancje_oob` liczy z nich dla każdej próbki uczącej wariancję (z `n - 1`
  w mianowniku) wyjść drzew, które jej nie widziały. Próbki z mniej niż dwoma takimi
  drzewami są pomijane. Średnia i suma kwadratów są aktualizowane drzewo po drzewie, więc
  macierz `T × n` nie powstaje. Korzysta z tego `OwnRegression` (rozdz. 12.2).

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

### 9.3a. Regresja w benchmarku

`methods_rf` rozpoznaje zadanie po typie lasu (`task`). Dla `RandomForestRegressor`
`classes` jest pustą tablicą, `l = 0`, a `Y_pred` ma typ `float64`. Metodą odniesienia
dla kolumn `*_full` jest `full_averaging` (`benchmark.REF_FULL_REG`) zamiast `full_voting`.

Kolumny jakości w regresji: `mae`, `rmse`, `r2`. Względem metod odniesienia:

- `agree_full`, `agree_rf`: odsetek próbek, dla których `|y - y_odniesienia| <= delta`
  tej metody. To dopełnienie „odsetka przekroczeń tolerancji” z dokumentu, więc tak jak
  w klasyfikacji porównuje się je z `pi`. Metody bez tolerancji (`rf_sklearn`,
  `full_averaging`) muszą dać ten sam wynik z dokładnością arytmetyki (`np.isclose`),
- `mad_*`, `maxdiff_*`: średnia i największa różnica bezwzględna.

`benchmark.deltas` (nazwa metody → `delta` po `fit`) trafia też do metadanych przebiegu
jako `deltas`. Jest potrzebne, bo `delta_std` daje w każdym przebiegu inną tolerancję.

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
- `build_methods(specs, sklearn_soft_voting, task)` tworzy tylko bloki, których klasa ma
  `task` równe podanemu (`benchmark.task` z configu). Jeden `config.toml` może więc
  zawierać metody klasyfikacyjne i regresyjne naraz. Dla regresji `rf_sklearn` to
  `SklearnRFReg`.
- `build_methods` zwraca **nowe, niedopasowane** instancje. `testing.py` woła je dla
  każdego lasu, bo `fit` dobiera metodę do konkretnego lasu.
- Nazwy metod muszą być różne; dwa bloki o tych samych parametrach wymagają różnych
  `name`.

`src/config.py` czyta `config.toml` przy imporcie i wystawia wartości jako stałe modułu.
Brak klucza w pliku daje `KeyError` przy imporcie. Wyjątki: brak `max_depth` w `[forest]`
oznacza drzewa bez limitu głębokości, brak `task` w `[benchmark]` oznacza klasyfikację,
a brak `drop_columns` pustą listę.

---

## 11. Regresja (`src/methods/regression.py`)

Warianty regresyjne z dokumentu projektu (algorytmy 2 i 5). Las to
`RandomForestRegressor`, predykcją pełnego zespołu jest średnia wyjść drzew `y_T`,
a predykcja po `t` drzewach `y_t` jest zgodna z pełną, gdy `|y_t - y_T| <= delta`.

### 11.1. Klasy

- `RegressionMethod`: baza. `fit` wymaga lasu regresyjnego i ustawia `trees` na listę
  `FastRegTree`. Kontrakt predykcji jest ten sam co w rozdz. 1, tylko `y_pred` to liczby.
- `SklearnRFReg` (`rf_sklearn`): oryginalne `rf.predict`.
- `FullAveraging` (`full_averaging`): średnia wszystkich drzew, liczona drzewo po drzewie.
  Odpowiednik `full_voting`: różnica czasu względem metod z wczesnym stopem pokazuje zysk
  z przerywania, a nie różnice implementacji.
- `SequentialRegression`: wspólne pętle wczesnego stopu. Podklasa podaje tylko tablicę
  mnożników `_mnoznik()`.
- `IBRegression` (`ib_reg`), `GleeRegression` (`glee_reg`) tutaj, `OwnRegression`
  (`own_reg`) w `own.py` (rozdz. 12.2).

### 11.2. Warunek stopu jako próg sumy kwadratów

Wszystkie trzy metody mają warunek postaci

```
c[t] * sigma_t <= delta
sigma_t^2 = (nu0 * sigma0_2 + S_t) / (nu0 + t - 1)
```

gdzie `S_t` to suma kwadratów odchyleń `t` dotychczasowych wyjść od ich średniej. Dla
`nu0 = 0` (IB, GLEE) `sigma_t` to zwykłe odchylenie standardowe `s_t` z `t - 1`
w mianowniku; `nu0 > 0` to a priori metody własnej. Warunek da się przekształcić tak, żeby
po lewej zostało samo `S_t`:

```
S_t <= prog_S[t] = (delta / c[t])^2 * (nu0 + t - 1) - nu0 * sigma0_2
```

`prog_sumy_kwadratow` liczy tę tablicę raz w `fit()`. W pętli zostaje aktualizacja
średniej i `S_t` (algorytm Welforda, jak w algorytmie 2 dokumentu) oraz jedno porównanie,
bez pierwiastka i dzielenia. Konwencje tablicy:

- `prog_S[t] = -1`: po `t` drzewach nie wolno przerwać (`t < t_min` albo mnożnik
  nieskończony). `S_t >= 0`, więc warunek nigdy nie zachodzi. Ważne przy identycznych
  wyjściach pierwszych drzew: `S_t = 0` nie może zatrzymać próbki przed `t_min`,
- `prog_S[T] = inf`: po wszystkich drzewach zawsze koniec, więc każda próbka dostaje
  predykcję i `n_used <= T`,
- ujemny próg przy `nu0 > 0` oznacza, że samo a priori wyklucza stop po tylu drzewach.

Pętle `predict_batch` (kolejność `fixed` i `shuffled`), `_predict_batch_random`
i `predict_one` mają tę samą budowę co w `SequentialPruning` (rozdz. 3), ze stanem
`(srednia, S)` zamiast macierzy głosów.

### 11.3. Mnożniki

- `mnoznik_studenta(T, pi, t_min, nu0)`: `c[t] = q * sqrt((T - t) / (T * t))`, gdzie `q`
  to kwantyl rzędu `(1 + pi) / 2` rozkładu t-Studenta o `nu0 + t - 1` stopniach swobody
  (`scipy.special.stdtrit`). Dla `nu0 = 0` to wzór (2) dokumentu: przedział predykcyjny
  dla średniej pozostałych `T - t` wyjść, przeliczony na różnicę `y_T - y_t`. Wymaga
  `t >= 2`.
- `mnoznik_normalny(T, pi, t_min, fpc)`: `c[t] = z * rho[t] / sqrt(t)` z kwantylem
  rozkładu normalnego i poprawką na populację skończoną `rho[t]` (`poprawka_fpc`, ta sama
  co w klasyfikacyjnym GLEE). To wzór (6) dokumentu.

Oba mnożniki zależą tylko od `T`, `pi` i parametrów metody, nie od `delta` ani od danych.

### 11.4. Parametry

- `delta` albo `delta_std` (dokładnie jedno). `delta` to tolerancja w jednostkach zmiennej
  celu, tak jak w dokumencie. `delta_std` podaje ją w odchyleniach standardowych `y_train`
  i jest przeliczane na `delta` w `fit()`; istnieje po to, żeby jeden blok configu miał
  sens na zbiorach o różnej skali celu. Po `fit` zawsze obowiązuje `self.delta`.
- `t_min`: najmniejsza liczba drzew przed stopem. `IBRegression`: domyślnie 2 (minimum
  potrzebne do `s_t`). `GleeRegression`: domyślnie `None`, czyli wartość z klasyfikacyjnego
  GLEE (`min_glosow`: 15, 30 albo 45). Powód: kwantyl normalny nie rośnie przy małym `t`
  tak jak kwantyl t-Studenta, więc z `t_min = 2` metoda staje po 2–3 drzewach, gdy ich
  wyjścia są przypadkiem bliskie (w próbnym przebiegu `agree_full` 0,80–0,94 przy
  `pi = 0,98`). Dokument wymaga tylko `t_min >= 2`; wartość domyślna jest decyzją kodu.
- `fpc` (`GleeRegression`): poprawka na populację skończoną.
- `order`, `seed`: jak w rozdz. 3.
- Domyślne nazwy: `IBREG_pi98_d5` (z `delta = 5`), `GLEEREG_pi98_ds0.05` (z `delta_std`).
- `pi = 1` daje nieskończony kwantyl, czyli metodę, która nigdy nie przerywa.

Założenie obu wariantów: wyjścia drzew dla ustalonej próbki są w przybliżeniu niezależne
i normalne. `pi` nie jest gwarancją dla danych, więc `agree_full` trzeba mierzyć.

---

## 12. Metody własne (`src/methods/own.py`)

### 12.1. `Own`: HYPER z a priori zmieszanym z jednostajnym

Klasa dziedziczy po `Hyper` i zmienia dwa miejsca (`_prior`, `_prog`); model urny,
rekurencja i odczyt tablicy zostają bez zmian.

**A priori** (wzór (9) dokumentu):

```
P_lam(T) = lam * P_OOB(T) + (1 - lam) * P_jednostajny(T)
```

`prior_jednostajny(T, l, para)` daje rozkład jednostajny na wszystkich składach urny
o sumie `T` (dla dwóch klas `1 / (T + 1)`), a w trybie pary jednostajny w każdym wierszu.
`lam = 1` to HYPER (tablica identyczna), `lam = 0` to IB pruning: tablica jest taka sama
jak z pliku we wszystkich komórkach, do których głosowanie może dojść (pozostałe HYPER
zostawia z wartością 0, IB z `T + 1`; żadnej nie da się odczytać). Uwaga z rozdz. 5.3
o remisach przy parzystym `T` dotyczy też `lam = 0`.

**Wariant rozszerzony** (`glee=True`): stop, gdy pozwala reguła HYPER **lub** GLEE.
`dolacz_glee(prog, prog_glee, para)` łączy tablice w `fit()`: obie mówią „stop, gdy głosy
lidera >= próg”, więc alternatywa to mniejszy z dwóch progów w każdej komórce. Tablica
GLEE jest indeksowana głosami drugiej klasy (`t_2`), czyli największą liczbą głosów poza
liderem; w formacie `tablica_hyper` jest to maksimum indeksów komórki. Predykcja nie
zmienia się i nie ma dodatkowego kosztu. `pi_glee` (domyślnie `pi`) to poziom ufności
reguły GLEE, `min_votes` jak w `Glee`.

Domyślne nazwy: `OWN_lam0.5_pi98`, z przyrostkami `_glee`, `_pair`, `_shuf`/`_rand`.

### 12.2. `OwnRegression`: IB z a priori wariancji wyjść

Regresyjny IB szacuje rozrzut wyjść tylko z `t` odpytanych drzew. Metoda własna dokłada
wiedzę z danych uczących (wzory (10)–(11) dokumentu):

- `sigma0_2`: mediana po próbkach uczących wariancji wyjść drzew out-of-bag
  (`wariancje_oob`, rozdz. 8), liczona w `fit(rf, X_train)`,
- `nu0`: siła a priori w pseudoobserwacjach (skalowany odwrotny rozkład chi-kwadrat).

Rozrzut w warunku stopu to wariancja ważona z rozdz. 11.2, a mnożnik to kwantyl
t-Studenta o `nu0 + t - 1` stopniach swobody. Cała metoda to więc `SequentialRegression`
z ustawionymi `nu0`, `sigma0_2` i `_mnoznik()`; pętle są wspólne.

- `nu0 = 0` to dokładnie `IBRegression` (i wtedy `fit` nie potrzebuje `X_train`).
- `t_min` jest domyślnie 1: przy `nu0 > 0` rozrzut nie spada do zera, więc stop jest
  możliwy już po pierwszym drzewie, jeśli `c[1] * sigma0 <= delta`.
- `n_prior`: ile losowych próbek uczących użyć do `sigma0_2` (`None` = wszystkie).
- Tablica mnożników nie zależy od danych; po każdym treningu lasu liczona jest od nowa
  tylko liczba `sigma0_2`.
