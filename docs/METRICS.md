# Metriche e benchmark di paper_analyzer

Documento vivo: ogni volta che si aggiunge una metrica va aggiunta anche qui
(definizione, formula, intervallo, interpretazione, limiti, posizione nel
codice). Quando l'insieme di metriche è completo verrà reso in PDF.

**Stato:** ✅ implementata · 🚧 in corso · 📋 pianificata

## 0. Convenzioni

- **Record**: una riga del JSONL prodotto da `export-dataset`. Contiene il testo
  del paper visto dal modello (messaggio `user`), la risposta grezza (messaggio
  `assistant`) e il `provider`.
- **Risposta**: il JSON con i campi `problem_addressed`, `methodology`,
  `experiments`, `results`, `limitations`, `future_work` (stringhe),
  `main_contributions`, `datasets_used`, `visual_analysis_findings` (liste),
  `relevant_elements` (stringa o lista).
- **Campi del paper** (`PAPER_FIELDS`): tutti i campi tranne
  `visual_analysis_findings`, che riporta le formule PySR (approssimazioni
  dell'IA) e non il contenuto del paper.
- **Testo sorgente**: il testo del paper senza il blocco `ADVANCED PLOT ANALYSIS`
  (`split_source`).
- Un punteggio `None` significa "niente da verificare", non zero né uno: non
  entra nelle medie.

### Quale domanda misura ogni livello

| Domanda | Livello | Riferimento |
|---|---|---|
| La risposta è ben formata e non degenere? | L1 | §1 |
| Quello che afferma è nel paper? (precision) | L2 | §2 |
| Ha riportato quello che conta? (recall) | L4 rubrica | §3 |
| Le formule riportate sono equivalenti a quelle estratte? | equivalenza | §4 |
| Le differenze tra modelli sono rumore? | statistica | §5 |
| Ogni affermazione è supportata dal testo? | Groundedness a strati + audit | §6–§7 ✅ |

---

## 1. Livello 1 — Filtro strutturale ✅

`src/evaluation/metrics.py::level1`

Una risposta passa solo se **nessuna** delle condizioni seguenti scatta
(`passed = reasons == ∅`). Un fallimento significa "scartare".

| Controllo | Definizione | Soglia |
|---|---|---|
| JSON valido | `json.loads` riesce e produce un oggetto | — |
| Chiavi mancanti | `EXPECTED_KEYS \ chiavi(risposta) ≠ ∅` | nessuna |
| Tipi errati | stringa dove serve lista o viceversa | nessuno |
| Campi vuoti | vedi sotto | ≤ 0.5 |
| Ripetizione di 4-grammi | vedi sotto | ≤ 0.20 |
| Compressione | vedi sotto | ≥ 0.25 |
| Eco dello schema | campo simile a un esempio del prompt | similarità < 0.85 |
| Campi riempitivo | elemento senza lettere né cifre (". . .") | nessuno |

Una lista al posto di una stringa (`list_for_string`) non fa fallire: è lo
stesso contenuto, corretto da `normalize_answer`.

**Frazione di campi vuoti**

$$\text{empty\_ratio} = \frac{|\{k \in P : \text{vuoto}(r_k)\}|}{|P|}$$

con $P$ i campi del paper presenti nella risposta. Un valore è vuoto se è in
`{"", "not specified", "n/a", "none"}` oppure è un *segnaposto*: tutte le sue
parole stanno in un vocabolario di riempitivi e almeno una è una negazione
("None explicitly mentioned"). Un nome vero ("No Language Left Behind") contiene
sempre una parola di contenuto e non viene scambiato per segnaposto.

**Ripetizione di n-grammi** (n = 4, sulle parole minuscole)

$$\text{rep}_4 = 1 - \frac{|\text{n-grammi distinti}|}{|\text{n-grammi totali}|}$$

0 = nessuna ripetizione, vicino a 1 = testo in loop. Per testi con meno di
$5n$ parole vale 0.

**Rapporto di compressione**

$$\text{comp} = \frac{|\text{zlib}(t)|}{|t|}\quad(\text{in byte UTF-8})$$

Un testo che si ripete si comprime molto (valore basso). Sotto 500 byte vale 1.

**Eco dello schema**: per ogni elemento $x$ dei campi attesi e ogni esempio $e$ del
`SYSTEM_PROMPT`, si calcola

$$\text{sim}(x,e) = \frac{2\,M}{|x|+|e|}$$

(`difflib.SequenceMatcher.ratio`, $M$ = caratteri in blocchi comuni). Fallisce se
$\text{sim} \ge 0.85$.

*Limiti.* Le soglie sono una prima stima con margine: sui primi 16 record reali
i testi normali stanno a rep₄ 0.00–0.05 e compressione 0.39–0.51, un loop
sintetico a 0.98 e 0.03; non è ancora stato salvato un vero output degenere per
ricalibrarle.

---

## 2. Livello 2 — Grounding (precision) ✅

`src/evaluation/metrics.py::level2`

Misura se ciò che la risposta afferma **sul paper** si trova nel paper. Non dice
nulla su ciò che manca (vedi §3). Punteggio $\in [0,1]$, media dei componenti
verificabili:

$$L2 = \frac{1}{|C|}\sum_{c \in C} s_c, \qquad
C = \{\text{numeri, riferimenti, dataset, separazione visiva}\}\ \text{con}\ s_c \ne \text{None}$$

**Eccezione:** se una formula PySR è trapelata nei campi del paper, $L2 = 0$
qualunque sia il resto (è l'unico errore che non vogliamo mai nel training set).

### 2.1 Numeri — `number_grounding`

$$s_\text{num} = 1 - \frac{|\{n \in N_r : n \notin N_p\}|}{|N_r|}$$

$N_r$ = numeri citati nei campi del paper della risposta, $N_p$ = numeri del
testo sorgente. I numeri sono normalizzati (`1,000` = `1000`, `0.50` = `0.5`).
Le cifre singole sono escluse: compaiono ovunque (numeri di sezione, elenchi) e
"verrebbero trovate" senza provare nulla. `None` se $N_r = \emptyset$.

### 2.2 Riferimenti — `reference_grounding`

Tabelle, figure, algoritmi ed equazioni citati devono esistere nel paper
(anche in numeri romani, es. "Table II"; le equazioni come "(3)").

$$s_\text{ref} = 1 - \frac{|\text{riferimenti non trovati}|}{|\text{riferimenti citati}|}$$

### 2.3 Dataset — `dataset_grounding`

Ogni elemento di `datasets_used` è "trovato" se vale una di: (a) il nome intero
compare nel paper; (b) un acronimo tra parentesi compare; (c) almeno il 50 %
delle parole significative (≥ 4 lettere) compare.

$$s_\text{data} = 1 - \frac{|\text{dataset non trovati}|}{|\text{dataset non vuoti}|}$$

### 2.4 Separazione visiva — `visual_separation`

Regola del prompt verificata meccanicamente: le formule PySR vanno **solo** in
`visual_analysis_findings`, presentate come approssimazioni dell'IA, mai nei
campi del paper.

- **Copertura**: una formula è riportata se (i) le sue costanti con ≥ 5 decimali
  significativi compaiono nei findings (impronta), oppure, senza costanti, per
  confronto di stringa, oppure (ii) è numericamente equivalente a una formula
  trovata nel testo (§4).
- **Framing**: un finding è inquadrato se contiene una di: "approximation",
  "not derived", "not stated", "ai-generated", "curve-fit".
- **Leakage**: una costante-impronta compare nei campi del paper.

$$s_\text{vis} =
\begin{cases}
0 & \text{se c'è leakage} \\
\text{None} & \text{se non ci sono formule PySR} \\
\dfrac{\text{coperte}}{\text{formule}} \cdot \left(1 - \dfrac{\text{non inquadrati}}{\text{findings}}\right) & \text{altrimenti}
\end{cases}$$

Senza sezione ADVANCED ma con findings: $s_\text{vis}=0$ (sono inventati).
Se non ci sono formule il punteggio è `None` e non 1: darlo avrebbe premiato i
modelli di visione che non estraggono nulla (LLaVA-NeXT, confronto 2026-10-08).

### 2.5 Copertura (diagnostica, non entra in L2) — `coverage`

L2 misura solo la precision: una risposta scarna che cita poco ottiene un voto
alto. Per questo si riportano anche figure con dati passate al modello, formule
disponibili/riportate e campi del paper rimasti vuoti.

*Limiti L2.* È un controllo di **presenza**, non di significato: un numero può
essere nel paper ma attribuito alla cosa sbagliata. Questo buco è quello che la
Groundedness NLI (§6) deve chiudere.

---

## 3. Livello 4 — Completezza con rubrica (recall) ✅

`src/evaluation/rubric.py`, dati in `gold/rubrics.jsonl`

Per ogni paper del gold set (10 paper) la rubrica elenca 8–12 **fatti atomici
pesati** che una buona risposta deve contenere (stile checkpoint di PRL-Bench,
arXiv 2604.15411). Bozza scritta da Claude (`annotator: claude-draft`,
`reviewed: false`) da rivedere a mano.

$$\text{Recall}_\text{rubrica} = \frac{\sum_i w_i\, f_i}{\sum_i w_i}, \qquad
\text{Strict} = \mathbb{1}[\forall i:\ f_i = 1]$$

$f_i\in\{0,1\}$ = fatto $i$ trovato; somme sui fatti *verificabili*. Un fatto di
tipo `text` senza keyword non è verificabile e non viene conteggiato.

| Tipo | Il fatto è trovato se… |
|---|---|
| `name` | un alias compare come parola intera (case-insensitive) |
| `number` | un numero della risposta soddisfa la regola sotto |
| `text` | ogni gruppo di keyword ha almeno un'alternativa presente (sottostringa, quindi radici come "turbulen" funzionano) |

**Regola numerica** (dal benchmark PHYSICS-Verified): con valore di riferimento
$v$ e valore della risposta $a$, vale se

$$\frac{|a-v|}{|v|} \le \tau\ (\tau=0.01)\quad\text{oppure}\quad \operatorname{round}(a, d_v) = v$$

con $d_v$ i decimali di $v$ (5.4 nella rubrica e 5.43 nella risposta coincidono).
Se $v=0$ si usa $|a|\le\tau$. Con unità `%`, 0.932 e 93.2 sono lo stesso valore.

I testi sono normalizzati NFKC (ligature "ﬂux", meno e trattino Unicode).
`check_rubric` verifica inoltre che ogni fatto della rubrica sia nel testo del
paper, per scovare refusi nell'annotazione.

*Limiti.* Le keyword verificano la presenza di termini, non il significato (una
frase negata contiene le stesse keyword). Le rubriche sono una bozza non ancora
revisionata. Dieci paper sono pochi: serve il bootstrap (§5).

---

## 4. Equivalenza di formule ✅

`src/evaluation/formulas.py`

Stabilisce se una formula riportata dal modello equivale a quella PySR anche se
arrotondata, scritta in LaTeX o con termini riordinati. Entrambe sono parsate con
sympy (LaTeX → espressione in $x_0$) e valutate su una griglia fissa di 50
punti, 40 log-spaziati in $[0.05, 300]$ e 10 negativi.

$$\text{equivalenti} \iff
\frac{\left|\{x_j \in G:\ |f(x_j)-g(x_j)| \le \tau\,\max(|g(x_j)|,10^{-9})\}\right|}{|G_\text{valido}|} \ge 0.9
\ \wedge\ |G_\text{valido}| \ge 10$$

con $\tau = 0.01$ (stessa tolleranza di PHYSICS-Verified) e $G_\text{valido}$ i
punti in cui entrambe sono reali (parte immaginaria < $10^{-9}$) e finite.

*Limiti.* È un test numerico, non una dimostrazione simbolica: due funzioni
diverse che coincidono sulla griglia risultano equivalenti. Funziona solo per
formule in una variabile.

---

## 5. Statistica: intervalli di confidenza ✅

`src/evaluation/stats.py`

Con 3–30 paper ogni differenza tra modelli ha bisogno di una barra d'errore.

- **Unità di ricampionamento: il paper** (come in PHYSICS-Verified). Più run
  dello stesso provider sullo stesso paper (seed diversi) si mediano prima, così
  un paper non pesa di più perché è stato eseguito più volte.
- **IC bootstrap percentile al 95 %** della media sui paper, con $B=10000$
  ricampionamenti con reinserimento, seed 0:

$$\bar{x}^{*}_b = \frac{1}{n}\sum_{i=1}^{n} x^{*}_{b,i},\qquad
\text{IC}_{95\%} = \left[\bar{x}^{*}_{(0.025\,B)},\ \bar{x}^{*}_{(0.975\,B)}\right]$$

- **Rumore tra run**: media sui paper della deviazione standard tra i run dello
  stesso provider.
- **Confronto a coppie** tra due provider A e B sui paper in comune: bootstrap
  delle differenze $d_i = \bar{x}_{A,i}-\bar{x}_{B,i}$. Se l'IC non contiene 0
  vince uno dei due, altrimenti "non separabili". Sotto 10 paper in comune
  l'esito è marcato *indicativo* (con 3 paper i ricampionamenti distinti sono 10).

---

## 6. Scelta e calibrazione del giudice di groundedness ✅

*Questa sezione descrive i giudici e la loro calibrazione su un gold; l'uso sugli output veri è in §7.*

**Obiettivo.** Colmare il limite di L2: verificare che ogni affermazione estratta
sia **supportata** dal testo del paper, non solo che i suoi numeri vi compaiano.
Schema ispirato a SciFact (claim → frasi di evidenza → etichetta), con un
modello NLI locale al posto del RoBERTa di VeriSci.

Schema SciFact adattato:
1. **Scomposizione** della risposta in claim atomici.
2. **Recupero** delle frasi del paper più vicine al claim.
3. **Giudizio locale**: etichetta ∈ {SUPPORTED, CONTRADICTED, NOT ENOUGH INFO} (NLI) o SUPPORTED / non SUPPORTED (MiniCheck).
4. **Aggregazione** (versione semplice; quella usata è in §7, che tratta anche il dubbio):

$$\text{Groundedness} = \frac{|\text{claim SUPPORTED}|}{|\text{claim}|},\qquad
\text{Contradiction rate} = \frac{|\text{claim CONTRADICTED}|}{|\text{claim}|}$$

5. **Calibrazione del giudice** su un gold di claim etichettati
   (accuracy, macro-F1, κ di Cohen), vedi §6.1.

Il $F_1$ stile SciFact tra evidenze predette $\hat{E}$ e di riferimento $E$ è

$$P=\frac{|\hat{E}\cap E|}{|\hat{E}|},\quad R=\frac{|\hat{E}\cap E|}{|E|},\quad F_1=\frac{2PR}{P+R}$$

ma richiederebbe evidenze annotate a mano per ogni claim, che non abbiamo (il gold ne ha una citazione per claim, non un insieme completo): non è calcolato.

**Codice.** Giudici in `src/evaluation/judges.py` (MiniCheck e NLI con recupero
BM25), metriche di accordo in `src/evaluation/calibration.py`, esecuzione su
Leonardo con `scripts/nli/`. Soglia dell'NLI fissata a 0.5, non regolata sul
gold. Un giudice binario (MiniCheck) distingue solo SUPPORTED / non SUPPORTED,
quindi il tasso di contraddizioni lo può dare solo un NLI a 3 classi.

### 6.1 Gold dei claim per calibrare il giudice ✅ (dati) · 🚧 (confronto dei giudici)

`gold/claims.jsonl`, verifica con `scripts/gold/check_claims.py`.

Serve a misurare quanto è affidabile il giudice NLI *prima* di usarlo: si fa
girare il giudice sui claim, se ne confrontano le etichette con quelle del gold.

| Campo | Significato |
|---|---|
| `claim` | affermazione atomica sul paper |
| `label` | `SUPPORTED`, `CONTRADICTED` o `NOT_ENOUGH_INFO` |
| `evidence` | citazione **verbatim** dal testo che il modello ha visto (per SUPPORTED e CONTRADICTED) |
| `absent_terms` | per NEI: termini che non devono comparire nel paper |
| `origin` | `model` (preso dall'output di Qwen3.8), `perturbed` (variante falsa di un fatto del paper: numero, negazione o scambio), `written` (scritto da zero a partire dal testo) |
| `annotator`, `reviewed` | `claude`, `false` finché non rivisto da una persona |

**Composizione (10 paper del gold set, 114 claim):** 62 SUPPORTED, 37
CONTRADICTED, 15 NOT_ENOUGH_INFO; 44 `model`, 36 `perturbed`, 34 `written`.

**Come sono stati etichettati.** Solo claim di cui l'etichetta era
verificabile sul testo: ogni SUPPORTED/CONTRADICTED ha la frase del paper che la
prova, e lo script controlla che la citazione esista davvero. Per i NEI la
verifica è l'assenza dei termini chiave nell'intero testo (controllo
necessario, non sufficiente: un NEI senza `absent_terms` si basa solo sulla
lettura). I claim CONTRADICTED sono in maggioranza perturbazioni costruite, per
avere negativi sicuri: i veri errori dell'LLM sono rari e il caso più
interessante trovato (p0-07, "ED conferma chiaramente l'esponente", mentre il
paper dice di no) è incluso.

**Metriche di calibrazione del giudice**, sulle etichette $y_i$ (gold) e
$\hat{y}_i$ (giudice), con $n$ claim:

$$\text{Accuracy} = \frac{1}{n}\sum_i \mathbb{1}[\hat{y}_i = y_i],\qquad
\kappa = \frac{p_o - p_e}{1 - p_e},\quad
p_o=\text{Accuracy},\ p_e=\sum_{c}\frac{n_c^{\text{gold}}\, n_c^{\text{giudice}}}{n^2}$$

$$\text{macro-}F_1 = \frac{1}{3}\sum_{c}\frac{2P_cR_c}{P_c+R_c}$$

*Limiti.* (i) Etichettatore unico (Claude) e non ancora rivisto da un umano: se
il giudice e l'annotatore sbagliano allo stesso modo, il κ è ottimistico.
(ii) I claim perturbati (cambio di un numero, negazione) mantengono quasi
tutte le parole dell'evidenza: per l'NLI sono facili (0.94 di accuracy), per
MiniCheck sono i più difficili (0.56–0.69), quindi la difficoltà dipende dal
giudice e non va assunta. (iii) Con 114 claim l'IC dell'accuracy è
largo (fino a ±9 punti al 95 %, ±7 per un'accuracy intorno a 0.85). (iv) Il gold copre solo 10 paper e non misura il
tasso reale di allucinazione del modello, solo l'affidabilità del giudice.

### 6.2 Risultati preliminari dei giudici (gold di 114 claim, 2026-10-21)

Compito binario "SUPPORTED vs non SUPPORTED", media e [IC 95 % bootstrap]. Due
job su Leonardo (A100). `minicheck-deberta` è fallito in entrambi per
un'incompatibilità di `transformers` 4.49 con DeBERTa-v2 (`device_map='auto'`):
la correzione in `judges.py` va ancora verificata, quindi manca nella tabella.

| Giudice | Accuracy | Macro-F1 | κ | Falsi accettati | SUPPORTED persi |
|---|---|---|---|---|---|
| **Bespoke-MiniCheck-7B** | **0.939 [0.895, 0.982]** | 0.937 | 0.875 | 7/52 | 0/62 |
| MiniCheck-RoBERTa | 0.807 [0.728, 0.877] | 0.806 | 0.611 | 11/52 | 11/62 |
| DeBERTa-NLI + BM25 | 0.789 [0.711, 0.860] | 0.788 | 0.588 | 2/52 | 22/62 |
| MiniCheck-Flan-T5 | 0.711 [0.623, 0.798] | 0.707 | 0.414 | 18/52 | 15/62 |
| Bespoke AND DeBERTa-NLI (esplorativo) | 0.807 | — | — | 0/52 | 22/62 |

"Falsi accettati" = claim CONTRADICTED o NOT_ENOUGH_INFO giudicati SUPPORTED:
è l'errore grave per la Groundedness, perché lascia passare un'allucinazione.

Cosa mostrano i dati:
- **Bespoke-MiniCheck-7B è il giudice migliore, in modo significativo.** Il
  confronto a coppie su accuracy dà +0.132 [+0.053, +0.219] su RoBERTa, +0.149
  [+0.061, +0.246] sull'NLI e +0.228 [+0.140, +0.325] su Flan-T5: gli
  intervalli escludono lo zero. Non perde nessun claim vero (62/62).
- **Ma non è un certificatore.** Accetta 7 claim su 52 falsi o non verificabili
  (13 %): 6 contraddizioni e 1 NEI, tutti con probabilità di supporto tra 0.53
  e 0.80 (mediana 0.89 sui claim veri, 0.11 sulle contraddizioni). Sono per lo
  più perturbazioni sottili: un numero cambiato (p4-10), un'inversione di
  ruolo (p8-08, p2-11), un'affermazione di processo (p9-08). Un'accuracy del
  94 % su 114 claim ha un IC di ±5 punti.
- **Gli altri giudici sbagliano in modo opposto.** L'NLI è prudente (precisione
  su SUPPORTED 0.95) ma perde 22 claim veri, 17 dei quali classificati NEI: la
  frase giusta non è tra quelle recuperate o il claim richiede più frasi.
  MiniCheck-RoBERTa e Flan-T5 accettano troppe contraddizioni (11 e 16 su 37).
- **Punto cieco comune** dei tre giudici provati nel primo run: quattro claim
  sbagliati da tutti, tre su formule o valori simbolici (p6-09: "μc ≈ λ/2"
  invece di "2λ"; p2-02: C_T = cos²ωt). Per numeri e formule servono i
  controlli deterministici (§2.1, §4), non l'NLI.
- La soglia dell'NLI (0.5) non è stata regolata sul gold; la combinazione AND è
  stata scelta guardando i risultati sul gold stesso, quindi il suo numero è
  esplorativo e va riconfermato su claim nuovi.

### 6.3 Calibrazione delle fasce di Bespoke-MiniCheck-7B ✅

Sul gold, la probabilità di supporto $p$ di Bespoke separa bene i claim veri dai
falsi. Dividendo $p$ in tre fasce (bordi 0.5 e 0.85):

| Fascia | SUPPORTED | CONTRADICTED | NOT_ENOUGH_INFO | Lettura |
|---|---|---|---|---|
| $p \ge 0.85$ | 45 | 0 | 0 | 45/45 veri |
| $0.5 \le p < 0.85$ | 17 | 6 | 1 | 17/24 veri (71 %) |
| $p < 0.5$ | 0 | 31 | 14 | 0/45 veri |

Nessun claim vero cade sotto 0.5 e nessun falso sopra 0.85. Gli errori di
Bespoke sono tutti nella fascia centrale. Limite importante: il bordo 0.85 è
stato scelto **dopo aver visto questa tabella** (con bordi 0.8–0.95 il risultato
è simile: nessun falso tra i verificati), quindi le percentuali sono ottimistiche
e vanno riconfermate su claim nuovi. Con 45 claim senza errori, il limite
superiore al 95 % del tasso di errore tra i VERIFIED è circa 3/45 ≈ 7 % (regola
del tre).

**Attenzione: questa calibrazione non vale per i claim reali.** I claim del gold
sono brevi, atomici e in buona parte costruiti da me. Sulle frasi lunghe e
composte prodotte dall'LLM le fasce non separano il vero dal falso: vedi l'audit
in §7.2, dove l'82 % dei claim della fascia $p<0.5$ risulta in realtà
pienamente supportato (contro lo 0 % del gold).

---

## 7. Groundedness a strati (valutazione di un'estrazione) ✅

`src/evaluation/groundedness.py`, `scripts/groundedness/`

Risponde a: *quanto di ciò che l'LLM ha scritto sul paper è vero secondo il
paper?* Non esiste un verificatore con certezza: si combinano controlli
indipendenti, ciascuno con la propria incertezza dichiarata. Il risultato più
importante di questa sezione è negativo: **il giudice automatico da solo non
basta a contare gli errori sugli output reali** (§7.2), per questo c'è uno
strato finale di audit.

**Strato 1 — giudice.** Ogni campo testuale dell'estrazione
(`problem_addressed`, `methodology`, `experiments`, `results`, `limitations`,
`future_work`, `main_contributions`) è scomposto in claim: una frase per claim,
una voce per elemento delle liste. Restano fuori `datasets_used`,
`relevant_elements` e `visual_analysis_findings`, già coperti dai controlli
deterministici di §2.3, §2.2 e §2.4. Bespoke-MiniCheck-7B assegna a ogni claim
la probabilità $p$ che il testo del paper lo supporti.

**Strato 2 — controllo deterministico dei numeri.** Ogni numero con almeno due
cifre citato dal claim deve comparire nel paper (stessa regola di §2.1). Serve
perché i giudici sono ciechi alle perturbazioni numeriche (§6.2). Garantisce
solo che il numero non sia inventato, non che sia attribuito alla cosa giusta
(un "62 %" presente in un'altra sezione passa).

**Strato 3 — dubbio.** Un claim con $p$ non decisivo non viene forzato a
vero/falso:

$$\text{stato}(c)=\begin{cases}
\text{UNSUPPORTED} & p<0.5\\
\text{NUMBER\_MISMATCH} & p\ge 0.5 \text{ e un numero del claim non è nel paper}\\
\text{DOUBTFUL} & 0.5\le p<0.85\\
\text{VERIFIED} & p\ge 0.85 \text{ e numeri trovati}
\end{cases}$$

$$\text{verified rate}=\frac{|\text{VERIFIED}|}{|\text{claim}|},\qquad
\text{flagged rate}=\frac{|\text{UNSUPPORTED}|+|\text{NUMBER\_MISMATCH}|}{|\text{claim}|}$$

Questi due tassi sono **segnali del giudice, non tassi di errore**: si calcolano
per risposta e si mediano sui paper con bootstrap come in §5 (run diversi dello
stesso paper sono prima mediati).

**Strato 4 — audit stratificato.** Si estrae un campione di claim per ciascuna
fascia e se ne verifica a mano la verità sul testo del paper, con tre esiti:
`TRUE` (tutto supportato), `PARTIAL` (in parte supportato, ma con un dettaglio
non nel paper, un'attribuzione sbagliata o un'esagerazione), `FALSE`
(contraddetto o invertito). Con $N_h$ claim nella fascia $h$ e $k_h$ esiti
di tipo $o$ su $n_h$ verificati, la quota di tutti i claim con esito $o$ è

$$\hat{q}_o=\frac{1}{N}\sum_h N_h\,\frac{k_h}{n_h}$$

L'intervallo è il 95 % credibile ottenuto campionando ogni $p_h$ da una
Beta$(k_h+\tfrac12,\ n_h-k_h+\tfrac12)$ (priore di Jeffreys): una fascia con 0
eventi su 20 verificati conserva così la sua incertezza, cosa che un bootstrap
semplice non farebbe. Codice: `stratified_estimate`.

### 7.1 Esecuzione su Qwen3.8 (2026-10-21)

1757 claim estratti da 12 run sui 10 paper del gold set (Qwen3.8-27B in tre
configurazioni, `think`, `ctx32k`, `ctx128k`, tre seed ciascuna, più il baseline
Qwen2.5-14B). Segnale del giudice e metriche già esistenti, media sui paper
[IC 95 % bootstrap sui paper]:

| Provider | VERIFIED | DOUBTFUL | flagged | L2 | recall rubrica | claim/risposta |
|---|---|---|---|---|---|---|
| Qwen2.5-14B | 0.456 [0.33, 0.58] | 0.400 | 0.143 [0.11, 0.19] | 0.843 | 0.519 [0.44, 0.59] | 11.2 |
| Qwen3.8 ctx32k | 0.479 [0.37, 0.58] | 0.376 | 0.145 [0.07, 0.25] | 0.705 | 0.876 [0.82, 0.92] | 15.3 |
| Qwen3.8 ctx128k | 0.486 [0.38, 0.59] | 0.421 | 0.093 [0.05, 0.14] | 0.687 | 0.871 [0.82, 0.92] | 15.8 |
| **Qwen3.8 think** | 0.399 [0.32, 0.49] | 0.532 | 0.069 [0.05, 0.09] | **0.987** [0.97, 1.00] | **0.952** [0.92, 0.98] | 22.4 |

Nessun claim è risultato NUMBER_MISMATCH: i numeri che il modello cita sono nel
paper (coerente con L2 numeri = 1.0). Gli intervalli di VERIFIED e flagged si
sovrappongono quasi del tutto tra i provider; l'unica differenza netta è la
completezza (recall 0.52 → 0.95) e L2 della configurazione `think`.

### 7.2 Audit: il giudice non basta (100 claim, 2026-10-21)

Campione casuale stratificato di 100 claim sui 1757 (20 VERIFIED, 40 DOUBTFUL,
40 UNSUPPORTED, da tutti i provider), verificato **senza vedere la fascia**.
Dati e motivazioni in `gold/audit_qwen38.jsonl`.

| Fascia del giudice | Claim | Verificati | TRUE | PARTIAL | FALSE |
|---|---|---|---|---|---|
| VERIFIED ($p\ge0.85$) | 785 | 20 | 19 | 1 | 0 |
| DOUBTFUL | 802 | 40 | 35 | 5 | 0 |
| UNSUPPORTED ($p<0.5$) | 170 | 40 | 33 | 6 | 1 |

Estrapolazione a tutti i 1757 claim:

| Esito | Quota | IC 95 % |
|---|---|---|
| Pienamente supportati | 0.904 | [0.808, 0.945] |
| Parzialmente supportati | 0.094 | [0.052, 0.186] |
| Falsi o invertiti | 0.002 | [0.002, 0.063] |

Lettura:
- **Il giudice discrimina poco sui claim reali.** La quota di claim pienamente
  veri è 95 % tra i VERIFIED, 88 % tra i DOUBTFUL, 82 % tra gli UNSUPPORTED:
  differenze non significative con questi campioni. I claim a $p<0.5$ che
  ho letto erano per lo più veri ma lunghi e composti (es. una frase con una
  tabella di p-value riportata testualmente dal paper ha $p=0.44$). Questo
  smentisce la calibrazione del gold (§6.3).
- **Gli errori veri sono rari ma esistono.** Su 100 claim, 1 falso (paper STAR, campo `limitations`: "energie
  più alte … punto critico a μB più alto": è invertito, il punto critico a μB
  maggiore corrisponde a energie *più basse*) e 12 parziali: cause inventate o
  distorte, attribuzioni sbagliate (bootstrap usato per l'incertezza e non per
  la significatività), un'inferenza del modello presentata come dichiarazione
  del paper, una formula senza il quadrato, un risultato analitico attribuito a
  un caso trattato numericamente. Il giudice ne ha intercettati pochi: il
  claim PARTIAL con $p=0.89$ è stato dato per VERIFIED.
- **Con un solo falso su 100 il tasso di falsi non è stimabile con precisione**:
  il valore puntuale è 0.2 %, ma l'estremo superiore dell'intervallo è circa 6 %.
- **Confronto tra provider: solo indicativo.** Il campione è stratificato per
  fascia, non per provider, e il segnale del giudice non è usabile per
  confrontarli. Guardando comunque i claim verificati per provider: Qwen3.8
  (tre configurazioni) 69 `TRUE` su 76, 7 `PARTIAL`, 0 `FALSE`; Qwen2.5-14B
  18 `TRUE` su 24, 5 `PARTIAL`, 1 `FALSE` (l'unico falso). La differenza
  (91 % contro 75 %) non è significativa (test esatto di Fisher bilaterale,
  $p=0.076$). Per un confronto serve un audit per provider.

**Limiti.**
- Audit e gold sono opera della stessa annotatrice (Claude): non c'è
  indipendenza. Per ogni claim ho visto le 3 frasi del paper più vicine
  (recupero BM25) e, nei casi dubbi, cercato nel testo; per i claim composti
  posso aver verificato meno a fondo le parti secondarie, quindi i `TRUE`
  potrebbero essere un po' ottimistici. Una verifica umana dei 13 claim non
  `TRUE` e di un sottocampione dei `TRUE` è raccomandata.
- Il claim è una frase, non un fatto atomico: una frase con due affermazioni,
  una vera e una dubbia, penalizza il giudice e il risultato è `PARTIAL`.
- `problem_addressed` e `future_work` contengono obiettivi e proposte, non
  fatti asseriti dal paper.
- VERIFIED o `TRUE` significano "il paper contiene l'informazione", non "il
  paper ha ragione".
- Gli strati non si accorgono di errori su formule e simboli (§6.2): per
  quelli vale §4, e solo quando c'è una formula PySR confrontabile.

**Uso.** (1) `scripts/groundedness/extract_claims.py` (PC), (2)
`scripts/groundedness/run_groundedness.sbatch` (Leonardo, 1 GPU, circa 31 minuti
per 1757 claim), (3) `scripts/groundedness/report.py --audit
gold/audit_qwen38.jsonl` (PC): tabella, stima da audit, `claim_status.jsonl` e
`flagged_claims.md` con i claim da rivedere e le frasi del paper più vicine.

### 7.3 Scomposizione atomica e audit per provider ✅ (2026-10-21)

**Perché.** Il giudice ha fallito sulle frasi lunghe e composte (§7.2) e
l'audit pooled non permette di confrontare i provider. Due modifiche, in un
unico job (`scripts/groundedness/run_atomic.sbatch`):

1. **Claim atomici.** Qwen3.8-27B (thinking spento, JSON vincolato, temperatura 0)
   spezza ogni frase in claim atomici e autosufficienti, usando come contesto
   il resto dello stesso campo solo per risolvere pronomi e "gli autori". Poi
   Bespoke giudica gli atomi invece delle frasi. Codice:
   `src/evaluation/atomic.py`, `scripts/groundedness/decompose_claims.py`.
2. **Audit per provider.** Dai claim atomici si campiona un numero fisso per
   ogni coppia provider × fascia (`scripts/groundedness/sample_audit.py`, seme
   fisso) e li si verifica a mano. Il tasso di ciascun provider è

$$\hat{q}_o^{(p)}=\frac{1}{N^{(p)}}\sum_h N^{(p)}_h\,\frac{k^{(p)}_h}{n^{(p)}_h}$$

   (come in §7, ma con $N_h^{(p)}$ i claim del provider $p$ nella fascia $h$) e
   la differenza tra due provider $A$ e $B$ si ottiene campionando i due
   $q^{(A)}, q^{(B)}$ dalle rispettive posteriori Beta indipendenti;
   si riportano la differenza, l'intervallo al 95 % e $P(q^{(A)}>q^{(B)})$.
   Codice: `stratified_difference`.

**Il decompositore è anch'esso un LLM e Qwen3.8 è uno dei provider valutati:**
potrebbe aggiungere o correggere contenuto senza dirlo. Due controlli
deterministici (`check_atoms`) lo limitano: un numero (almeno due cifre)
dell'atomo deve stare nella frase di partenza, e almeno il 70 % delle parole
piene dell'atomo (prime 5 lettere) deve comparire nel testo dello stesso campo.
Se un atomo di una frase non li supera, o la richiesta fallisce, la frase resta
intera: si riporta la quota di frasi in questo caso. Il prompt vieta di
correggere, ammorbidire o aggiungere, ma l'autocorrezione di errori non è
rilevabile da questi controlli.

**Cosa si confronta.** Se i claim atomici rendono il giudice più informativo,
la quota di claim veri deve differire di più tra fascia VERIFIED e UNSUPPORTED
rispetto a §7.2 (95 % contro 82 %).

#### Risultati (job 61642604, Leonardo)

**Scomposizione.** 1009 frasi (think 672 + Qwen2.5-14B 337) → 2823 claim
atomici, 2,8 per frase. Solo 17 frasi (1,7 %) sono tornate intere, tutte per
`words_added`; nessuna per `number_added` né `no_valid_atoms`. Giudizio di
Bespoke sui 2823 atomi: 2913 s.

| Provider | Atomi | VERIFIED | DOUBTFUL | UNSUPPORTED | NUMBER_MISMATCH |
|---|---|---|---|---|---|
| Qwen2.5-14B | 692 | 399 | 221 | 72 | 0 |
| Qwen3.8-27B [think] | 2131 | 930 | 1047 | 154 | 0 |

Quota media su 10 paper (IC 95 % bootstrap): VERIFIED 0,57 [0,46, 0,67] contro
0,44 [0,37, 0,50]; flagged 0,108 contro 0,069. Sono segnali del giudice, non
tassi di errore.

**Audit per provider.** 84 atomi (14 per provider × fascia, seme 42), etichettati
alla cieca da un solo annotatore (Claude, `gold/audit_atomic.jsonl`,
`reviewed: false`): 63 TRUE, 19 PARTIAL, 2 FALSE.

| Fascia del giudice | Atomi veri / verificati | Quota vera | Frasi intere (§7.2) |
|---|---|---|---|
| VERIFIED | 26 / 28 | 0,93 | 0,95 |
| DOUBTFUL | 21 / 28 | 0,75 | 0,88 |
| UNSUPPORTED | 16 / 28 | 0,57 | 0,82 |

La separazione tra le fasce migliora: tra VERIFIED e UNSUPPORTED passa da 13 a
36 punti. Un atomo UNSUPPORTED è sbagliato nel 43 % dei casi contro il 18 % delle
frasi, quindi la fascia è un segnale utile per decidere dove guardare. Resta
un triage, non un conteggio degli errori: più di metà degli UNSUPPORTED è
comunque vera o quasi, e il campione per cella è di soli 14 atomi.

**Stime per provider** (estrapolate su tutti i claim del provider, IC 95 %
credibile):

| Provider | Pienamente supportati | Parziali | Falsi |
|---|---|---|---|
| Qwen3.8-27B [think] | 0,883 [0,729, 0,947] | 0,117 [0,052, 0,269] | 0,000 [0,002, 0,111] |
| Qwen2.5-14B | 0,778 [0,625, 0,861] | 0,207 [0,125, 0,358] | 0,015 [0,010, 0,126] |

Differenza think − Qwen2.5 nei pienamente supportati: +0,105
[−0,068, +0,263], $P(\text{think}>\text{Qwen2.5})=0{,}89$. L'intervallo include
lo zero: la direzione è favorevole a Qwen3.8 e coerente con §7.2, ma con questo
campione **non è dimostrata**. Servirebbero più atomi per cella.

**Lettura dei PARTIAL e dei FALSE.** Sono quasi tutti la stessa famiglia:
generalizzazioni delle sezioni *experiments* e *future_work* ("various
translation datasets" quando c'è un solo dataset WMT, "future work" ricavato da
una frase che non lo propone), più due inversioni ("i modelli senza punto
critico mostrano trend verso il punto critico", un limite inventato oltre
62,4 GeV). I numeri non sono mai la causa: NUMBER_MISMATCH è 0 su 2823.

**Limiti.**
- Un solo annotatore, lo stesso che ha etichettato il gold (§6.1) e l'audit di
  §7.2: errori sistematici dell'annotatore non emergono. Revisione umana su
  `gold/audit_atomic.jsonl` ancora da fare.
- Circolarità: Qwen3.8 scompone le frasi e al tempo stesso è il provider
  valutato. I controlli deterministici non vedono un'autocorrezione. L'effetto
  non è misurato.
- Il passaggio ai claim atomici cambia anche la popolazione (2,8 atomi per
  frase), quindi le quote VERIFIED/DOUBTFUL di §7.1 e di §7.3 non sono
  confrontabili direttamente.
- Alcune etichette su *future_work* e *experiments* dipendono da quanto si
  accetti un'inferenza implicita come "supportata".

#### Revisione indipendente di 40 atomi (2026-10-22)

Un revisore umano **non esperto del dominio** ha riletto 40 dei 84 atomi
(`gold/audit_atomic_review.jsonl`): tutti i 21 che Claude aveva segnato PARTIAL
o FALSE più 19 TRUE a caso, in ordine mescolato, senza le etichette di Claude e
con soltanto le 3 frasi del paper più vicine (le stesse del giudice). Poteva
rispondere `?` se le frasi non bastavano. Il campione non è casuale: la quota
di non-TRUE è sovrarappresentata, quindi l'accordo va letto per classe.

Accordo sulle risposte con etichetta (31 su 40; 9 risposte `?`):

| | Claude TRUE | Claude PARTIAL | Claude FALSE |
|---|---|---|---|
| Revisore TRUE | 15 | 3 | 0 |
| Revisore PARTIAL | 2 | 7 | 1 |
| Revisore FALSE | 0 | 2 | 1 |

$$\text{accordo}=\frac{23}{31}=0{,}74,\qquad \kappa=0{,}53$$

Se si distingue solo TRUE da non-TRUE (la distinzione che conta per «il paper
supporta il claim»): accordo 26/31 = 0,84, $\kappa=0{,}67$.

**Le 9 risposte `?`.** 7 sono atomi che Claude aveva segnato PARTIAL: le frasi
mostrate non contengono ciò che il claim afferma, quindi il revisore non può
confermarlo, che è coerente con «non pienamente supportato». Le altre 2 sono
atomi che Claude aveva segnato TRUE usando il paper intero (non solo le frasi
recuperate). Contando `?` come «non confermato TRUE», l'accordo binario su 40
atomi è 33/40 = 0,83.

**I 8 disaccordi (su 31) non hanno una direzione sola.** In 3 casi il revisore è
più indulgente (es. «comprehensive analysis» e il caso dei 4 metodi MFT/LFWT/
VMC/ED, dove Claude ha usato il fatto che ED viene da un lavoro precedente,
visibile solo nel paper intero); in 2 più severo (formula e «non generico»); 3
differenze tra PARTIAL e FALSE (la differenza è di grado). Mai un atomo TRUE per
il revisore e FALSE per Claude.

**Lettura.** Le etichette di Claude reggono a una lettura indipendente sul
discrimine TRUE / non-TRUE, ma l'accordo sulle tre classi è solo moderato
($\kappa=0{,}53$) e la distinzione PARTIAL / FALSE è soggettiva. Il tasso di FALSE
(0 – 1,5 % per provider in questa sezione) va quindi letto come «pochissimi
atomi sicuramente falsi», non come una cifra precisa. I valori di accordo
vanno riportati in tesi insieme al limite: revisore non esperto, solo frasi
recuperate, 31 etichette valide. Le stime per provider sono state calcolate con
le etichette di Claude e non sono state ricalcolate con quelle del revisore.

### 7.4 Stabilità, controlli del giudice e SciFact ✅

Quattro misure aggiunte il 2026-10-22. Nessuna misura la correttezza di un
singolo claim: servono a sapere quanto fidarsi del confronto tra provider e del
giudice.

**Stabilità tra run** ✅ (`src/evaluation/consistency.py`,
`scripts/groundedness/seed_stability.py`, senza GPU). Per un provider e un paper
con $k$ run, ogni claim del run $i$ si confronta con quelli del run $j$ con il
coefficiente di Dice sugli insiemi di radici delle parole piene (prime 5 lettere
delle parole di 4+ lettere) e dei numeri:

$$\mathrm{dice}(A,B)=\frac{2\,|A\cap B|}{|A|+|B|},\qquad
\text{agreement}_{i\to j}=\frac{\#\{a\in i:\ \max_{b\in j}\mathrm{dice}(a,b)\ge 0{,}5\}}{|i|}$$

L'accordo di un paper è la media sulle coppie ordinate $(i,j)$, $i\neq j$; si
riporta la media sui paper con IC bootstrap (§5). L'accordo sui numeri è
l'indice di Jaccard dei numeri (2+ cifre) citati da due run, mediato sulle
coppie. Un accordo basso significa che un singolo run è una base debole per una
conclusione; un accordo alto non dice che il contenuto sia giusto.

| Provider | Run/paper | Accordo sui claim | Accordo sui numeri |
|---|---|---|---|
| Qwen2.5-14B | 3,0 | 0,54 [0,43, 0,64] | 0,38 [0,29, 0,48] |
| Qwen3.8 ctx32k | 2,1 | 0,73 [0,62, 0,82] | 0,76 [0,60, 0,93] |
| Qwen3.8 ctx128k | 2,7 | 0,75 [0,66, 0,83] | 0,60 [0,36, 0,85] |
| Qwen3.8 think | 3,0 | 0,58 [0,52, 0,65] | 0,58 [0,45, 0,71] |

Lettura: la modalità thinking è meno stabile di quella senza thinking (0,58
contro 0,73), cosa attesa con temperatura 1,0 e risposte più lunghe (22 claim
per risposta contro 15): più claim dà più occasioni di dire cose diverse.
Qwen2.5-14B è il meno stabile. La stabilità non è correttezza: Qwen3.8 think
resta il migliore per recall e L2.

**Costo e affidabilità** ✅ (stesso script). Risposte rifiutate (JSON non
valido o troncato) e lunghezza media:

| Provider | Risposte | Rifiutate | Caratteri/risposta | Claim/risposta |
|---|---|---|---|---|
| Qwen2.5-14B | 30 | 2 (6,2 %) | 2867 | 11,2 |
| Qwen3.8 ctx32k | 21 | 9 (30,0 %) | 7338 | 15,3 |
| Qwen3.8 ctx128k | 27 | 3 (10,0 %) | 8328 | 15,8 |
| Qwen3.8 think | 30 | 0 (0 %) | 11204 | 22,4 |

**Controlli del giudice su claim reali** ✅ (`src/evaluation/controls.py`,
`make_controls.py`, `controls_report.py`, job `run_controls.sbatch`). I gold di
§6.1 sono claim brevi e sintetici (§7.2 ha mostrato che non si trasferiscono).
Qui il giudice vede claim veri dei provider, modificati in modo che la risposta
giusta sia nota:
- *controllo negativo*: un claim del paper A giudicato contro un paper non
  correlato B (mai dello stesso gruppo tematico). Risposta giusta: non
  supportato. Tasso di falso supporto $=\#\{p\ge t\}/n$, con $t=0{,}5$ e $0{,}85$.
- *claim perturbati*, partendo da claim che il giudice aveva già dato VERIFIED:
  `number` (un numero sostituito con uno assente dal paper), `flip` (una parola
  sostituita dal suo contrario: increase/decrease, higher/lower, ...), `negate`
  (copula negata, mai se il claim è già negativo). Tasso di rilevazione
  $=\#\{p<t\}/n$.
Intervalli di Wilson al 95 %. Il livello numerico deterministico (§2.1) rileva i
claim `number` per costruzione. Limiti: i `flip` non sono sempre sensati e un
paper non correlato può supportare un claim molto generico; i tassi misurano la
sensibilità del giudice, non il tasso di errore dei provider.

**Risultati** (job 61804913; 771 claim di controllo; Wilson 95 %):

| Controllo | Provider | n | Esito |
|---|---|---|---|
| Negativo, falso supporto $p\ge0{,}5$ | Qwen2.5-14B | 114 | 0,000 [0,000, 0,033] |
| | Qwen3.8 think | 286 | 0,014 [0,005, 0,035] |
| Negativo, falso supporto $p\ge0{,}85$ | Qwen2.5 / Qwen3.8 | 114 / 286 | 0,000 / 0,000 [0,000, 0,013] |
| `number` rilevato ($p<0{,}5$) | Qwen2.5 / Qwen3.8 | 12 / 80 | 1,00 / 0,925 [0,846, 0,965] |
| `flip` rilevato ($p<0{,}5$) | Qwen2.5 / Qwen3.8 | 37 / 92 | 0,838 / 0,707 [0,607, 0,790] |
| `negate` rilevato ($p<0{,}5$) | Qwen2.5 / Qwen3.8 | 54 / 96 | 1,00 / 0,927 [0,857, 0,964] |

Se si conta come «rilevato» anche solo un declassamento sotto 0,85, `flip` sale a
0,97 / 0,94 e `number` e `negate` a 1,00. Il $p$ medio scende da circa 0,90 a
0,08–0,35.

Lettura. (1) Il giudice non dà quasi mai supporto a un claim giudicato contro un
paper non correlato (al più 1,4 %): non "approva" a caso. (2) Sui claim reali
rileva quasi tutte le negazioni e i numeri cambiati (93 %), meno le inversioni
di una parola (71 % sotto 0,5): la fascia DOUBTFUL raccoglie soprattutto
queste. (3) Questo corregge in parte §6.2, dove i giudici risultavano ciechi
ai numeri: lì i numeri cambiati erano piccole variazioni su claim brevi, qui
un cambio di un fattore 1,5–2,5 dentro un claim lungo; la sensibilità dipende
dalla dimensione della perturbazione, quindi lo strato numerico deterministico
(§2.1, §7) resta necessario per variazioni piccole. (4) Perturbazioni su
claim già VERIFIED: i tassi non valgono per il caso in cui il giudice
sbaglia già in partenza.

**SciFact** ✅ (`scripts/scifact/`, `run_scifact.py`). Il giudice sul dev set di
SciFact (Wadden et al., 2020) in modalità *oracle*: la premessa è l'abstract del
documento citato dal claim (`cited_doc_ids`), quindi si misura la verifica e non
il recupero. Etichette: SUPPORTED se esiste evidenza SUPPORT, CONTRADICTED se
CONTRADICT, altrimenti NOT_ENOUGH_INFO; il giudice è binario (supportato o no),
quindi si riportano accuratezza, accuratezza bilanciata, macro-F1, kappa (§6.1) e
l'accuratezza per classe. Serve a collocare il giudice su un benchmark noto;
non è confrontabile con il punteggio pubblico di SciFact, che include il
recupero e l'identificazione delle frasi di evidenza.

**Risultati** (Bespoke-MiniCheck-7B, 300 claim del dev set: 124 SUPPORTED, 64
CONTRADICTED, 112 NOT_ENOUGH_INFO):

| Metrica (binaria) | Valore |
|---|---|
| Accuratezza | 0,877 [0,837, 0,913] |
| Accuratezza bilanciata | 0,857 |
| Macro-F1 | 0,867 |
| Kappa di Cohen | 0,737 |
| Precisione / richiamo su SUPPORTED | 0,948 / 0,742 |

Accuratezza per classe di gold: SUPPORTED 0,742 (92/124), CONTRADICTED 0,953
(61/64), NOT_ENOUGH_INFO 0,982 (110/112). Il giudice è conservativo: quando
dice «supportato» ha ragione il 95 % delle volte (solo 5 falsi positivi su 97),
ma rifiuta un claim supportato su quattro (32 su 124). Per un valutatore di
allucinazioni è il lato giusto dell'errore, perché sottostima la quota di
claim supportati invece di sovrastimarla; coincide con la quota alta di
DOUBTFUL osservata sui claim dei provider.


---

## Registro delle modifiche

| Data | Modifica |
|---|---|
| 2026-10-21 | Prima stesura: L1, L2, rubrica, equivalenza formule, bootstrap; §6 pianificata |
| 2026-10-21 | §6.1: gold di 114 claim (`gold/claims.jsonl`) e script di verifica |
| 2026-10-21 | §6.2: risultati dei giudici su Leonardo (Bespoke-MiniCheck-7B 0.939); fix DeBERTa-v2 in `judges.py` |
| 2026-10-21 | §6.3 calibrazione delle fasce di Bespoke; §7 groundedness a strati e scripts/groundedness |
| 2026-10-21 | §7.1-7.2: valutazione di Qwen3.8 (1757 claim) e audit di 100 claim; la calibrazione di §6.3 non vale sui claim reali; rimossi precisione stimata dal gold e F1 indicativo |
| 2026-10-21 | §7.3: scomposizione atomica (`atomic.py`, `decompose_claims.py`, `run_atomic.sbatch`) e audit per provider (`sample_audit.py`, `stratified_difference`); risultati inseriti |
| 2026-10-21 | §7.3: risultati del job atomico (2823 atomi) e audit di 84 atomi per provider (`gold/audit_atomic.jsonl`); separazione tra fasce 93/75/57 % contro 95/88/82 %; differenza tra provider non significativa |
| 2026-10-22 | §7.3: revisione di 40 atomi da un revisore non esperto (`gold/audit_atomic_review.jsonl`): accordo 0,74 (κ 0,53) sulle 3 classi, 0,84 (κ 0,67) TRUE contro non-TRUE |
| 2026-10-22 | §7.4: stabilità tra run e costo/affidabilità (risultati, Qwen); controlli del giudice su claim reali e SciFact (script pronti, risultati da inserire); preparati gli script per Gemma 4 31B (`scripts/gemma4/`) |
| 2026-10-22 | §7.4: risultati dei controlli (negativo: falso supporto ≤1,4 %; rilevazione di number/negate 93 %, flip 71 %) e di SciFact dev oracle (accuratezza 0,877, kappa 0,74; richiamo SUPPORTED 0,74) |
