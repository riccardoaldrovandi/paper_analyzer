"""Writes gold/figure_checks.jsonl: verifiable questions on the main figures of the 10 gold papers.

The rubrics (gold/rubrics.jsonl) check facts that are almost all in the text,
so they can't tell whether a model READ a figure: "v -> 0 ... laminar" contains
the rubric keywords even when the figure shows a plateau at 0.6 (Flux tube,
Mistral Small 4 single, matrix 2026-10-27). These checks ask what only a correct
reading of the figure answers: number of curves, values at the start, peaks and
asymptotes, which curve is higher or decays faster, crossing points.

Drafted by Claude from the figure images and captions, to be reviewed (as the
rubrics). One JSON line per check:
  paper       gold PDF name without extension (the LaTeX record of the paper matches too)
  figure      the paper's figure number;  panel: "a"/"b"/... or null
  id, question, answer (human-readable), kind: count | value | trend | relation | caption
  accept      regexes (case-insensitive, DOTALL); the check passes if ANY matches...
  reject      ...and NONE of these matches (a wrong reading stated next to a right one fails)
Whether a check is answerable from the text alone is computed by the scorer
(in_text: an accept pattern matches the paper text), not written here.

    python scripts/matrix/build_figure_checks.py
"""
import json
import os

FLUX = "1482468_Flux tube in turbulent flow and quantum chromodyna"
SYK = "1473712_Sachdev–Ye–Kitaev model as Liouville quantum mecha"
OTOC = "1519859_Out-of-time-order correlators in quantum mechanics"
SU4 = "1751397_SU(4) Heisenberg model on the honeycomb lattice wi"
NETP = "1850675_Nonmonotonic Energy Dependence of Net-Proton Numbe"
TWIST = "3056128_Simulating twistronics without a twist"
QAH = "3057253_Quantum anomalous Hall phase in synthetic bilayers"
GQA = "5ae6fb6b_GQA Training Generalized Multi-Query Transformer M"
AKT = "779080_The anti-$k_t$ jet clustering algorithm"
TXL = "c4744a7c_Transformer-XL Attentive Language Models beyond a"

# Small regex vocabulary shared by the checks.
APPROX = r"(?:≈|~|∼|about|around|near|approximately|roughly|close to|of|at|=|\\approx|\\sim)"
TO_ZERO = r"(?:(?:decay\w*|go(?:es)?|approach\w*|tend\w*|fall\w*|drop\w*|vanish\w*)\s+(?:to|toward|towards)|approach\w*)\s+(?:zero|0)\b"
PHI = r"(?:φ|ϕ|phi|\\phi|\\varphi|curve 2)"
V = r"(?:\bv\(x\)|\bv\b|velocity|curve 1)"

CHECKS = [
    # ---------------------------------------------------------------- Flux tube, Fig. 1
    (FLUX, 1, None, "fl1", "count", "Quante curve ci sono?", "2: v(x) (curva 1) e φ(x) (curva 2)",
     [r"\b(?:two|2)\s+(?:curves|profiles|lines|functions)", r"(?=.*curve 1)(?=.*curve 2)"], [r"\b(?:three|3)\s+(?:curves|profiles|lines)"]),
    (FLUX, 1, None, "fl2", "value", "Valore di v(x) all'origine?", "≈6",
     [rf"(?:start\w*|begin\w*|maximum|max|peak\w*|x\s*=\s*0|origin|v\(0\)|core)[^.;]{{0,60}}{APPROX}\s*6(?:\.0)?\b"], []),
    (FLUX, 1, None, "fl3", "value", "A cosa tende v(x) per x grande?", "plateau non nullo ≈0.6 (v∞ ≠ 0)",
     [r"(?:plateau|levels? off|saturat\w*|finite|non-?zero|nonvanishing)[^.;]{0,80}(?:v|velocity|curve 1|v_?\{?\\?infty|v∞)",
      r"(?:v|velocity|curve 1|v_?\{?\\?infty\}?|v∞)[^.;]{0,80}(?:plateau|levels? off|saturat\w*|finite|non-?zero|≠\s*0|\\neq\s*0|0\.[4-8])"],
     [rf"\bboth\b[^.;]{{0,80}}{TO_ZERO}", rf"{V}(?:(?!{PHI}|\bwhile\b|\bwhereas\b|,)[^.;]){{0,60}}{TO_ZERO}", r"v_?\{?\\?infty\}?\s*(?:→|->|\\to|approaches)\s*(?:0|zero)", r"v∞\s*(?:→|->)\s*0", rf"{V}(?:(?!{PHI}|\bwhile\b|,)[^.;]){{0,80}}\bto (?:zero|0) at large"]),
    (FLUX, 1, None, "fl4", "relation", "Quale curva va a zero prima?", "φ(x): ≈0 già a x ≈ 2-3, prima di v",
     [rf"{PHI}[^.]{{0,140}}(?:faster|more rapidly|more quickly|earlier|sooner|by x\s*{APPROX}\s*[23])"],
     [rf"{PHI}(?:(?!{V}|\bwhile\b|,)[^.;]){{0,100}}more slowly"]),
    # ---------------------------------------------------------------- SYK, Fig. 1 (left: linear, right: log-log)
    (SYK, 1, "left", "sy1", "caption", "Cosa si confronta nel pannello sinistro?", "risultato numerico (rosso) e analitico (tratteggiato)",
     [r"numeric\w*[^.]{0,120}analytic|analytic[^.]{0,120}numeric\w*"], []),
    (SYK, 1, "left", "sy2", "value", "Dove sta il massimo di G(ε)?", "≈2.7 a ε/J ≈ 0.05",
     [r"(?:peak|maximum|max)\w*[^.;]{0,80}(?:2\.[5-9]|0\.0[3-7])", r"(?:2\.[5-9]|0\.0[3-7])[^.;]{0,60}(?:peak|maximum)"], []),
    (SYK, 1, "left", "sy3", "trend", "Come si comporta G(ε) per ε → 0?", "va a zero (soppressione ∝ ε^1/2)",
     [r"(?:ε|epsilon|\\epsilon|\\varepsilon|energ\w*)[^.;]{0,60}(?:→|->|\\to|small|low)[^.;]{0,60}(?:zero|0\b|vanish|suppress)",
      r"(?:vanish\w*|suppress\w*|goes to zero|rises? from zero|starts? (?:at|from) zero)", r"\^\{?1/2\}?|√|sqrt"], []),
    (SYK, 1, "right", "sy4", "trend", "Quali leggi di scala mostra il pannello destro?", "ε^1/2 a piccola energia, ε^-1/2 a grande",
     [r"(?:-|−)\s*1/2|\^\{?-1/2\}?|−½|-½"], []),
    # ---------------------------------------------------------------- OTOC, Fig. 1(a)
    (OTOC, 1, "a", "ot1", "trend", "Come evolve l'OTOC della particella in una scatola?", "periodico: torna al valore iniziale 1",
     [r"period\w*", r"return\w*\s+to\s+(?:its\s+)?initial|comes? back|back to 1"], [r"saturat\w*[^.]{0,40}box|box[^.]{0,60}saturat"]),
    (OTOC, 1, "a", "ot2", "value", "Qual è il periodo?", "t ≈ 0.32 (≈1/π)",
     [r"0\.3[0-3]\d*", r"1\s*/\s*(?:π|\\pi)", r"\\pi\^\{-1\}"], []),
    (OTOC, 1, "a", "ot3", "relation", "Come dipende dalla temperatura?", "T più alta → OTOC più grande",
     [r"(?:higher|larger|increas\w*)\s+(?:T|temperature)[^.]{0,80}(?:higher|larger|grow\w*|increas\w*|bigger)",
      r"(?:grow\w*|increas\w*|rise\w*|scale\w*)\s+with\s+(?:the\s+)?(?:T|temperature)"], []),
    (OTOC, 1, "a", "ot4", "value", "Massimo dell'OTOC per T = 200?", "≈35-40 (scala log)",
     [r"\b(?:3[0-9]|40)(?:\.\d+)?\b[^.;]{0,60}(?:T\s*=\s*200|peak|maximum)", r"(?:T\s*=\s*200|peak|maximum)[^.;]{0,80}\b(?:3[0-9]|40)\b"], []),
    # ---------------------------------------------------------------- SU(4), Fig. 6(a)
    (SU4, 6, "a", "su1", "caption", "Quali stati si confrontano?", "FM AFOxz, QSOL e NCD",
     [r"(?=.*QSOL)(?=.*NCD)(?=.*(?:FM\s*AFO|AFOxz|AFO_?\{?xz))"], []),
    (SU4, 6, "a", "su2", "value", "A quale η si incrociano QSOL e FM AFOxz?", "η ≈ 0.175",
     [r"0\.1[78]\d*"], []),
    (SU4, 6, "a", "su3", "value", "A quale η l'NCD incrocia FM AFOxz?", "η ≈ 0.12",
     [r"0\.1[12]\d*"], []),
    (SU4, 6, "a", "su4", "trend", "Come varia l'energia con η?", "diminuisce (più negativa) per tutti gli stati",
     [r"(?:decreas\w*|more negative|lower\w*|drop\w*|fall\w*)[^.]{0,80}(?:η|eta|\\eta|Hund)", r"(?:η|eta|\\eta|Hund)[^.]{0,80}(?:decreas\w*|more negative|drop\w*|fall\w*)"], []),
    # ---------------------------------------------------------------- Net-proton, Fig. 1 and 2
    (NETP, 1, None, "np1", "trend", "Come si spostano le distribuzioni con l'energia?", "verso valori più bassi all'aumentare di √sNN",
     [r"(?:shift|move)\w*[^.]{0,40}(?:to(?:ward)?s?\s+)?(?:lower|smaller|left|zero)[^.]{0,80}(?:higher|increasing|larger)[^.]{0,30}(?:energ|√s|sqrt|collision)",
      r"(?:increasing|higher|larger)[^.]{0,30}(?:energ|√s|sqrt)[^.]{0,100}(?:shift|mov)\w*[^.]{0,30}(?:lower|smaller|left|zero)",
      r"(?:peak|mean|centr)\w*[^.]{0,60}(?:decreas|lower|shift\w* (?:to|toward)s? (?:lower|smaller|zero))[^.]{0,60}(?:energ|√s|sqrt)"], []),
    (NETP, 1, None, "np2", "value", "Dove sta il picco a 7.7 GeV?", "ΔNp ≈ 27",
     [r"7\.7[^.]{0,120}\b2[5-9]\b", r"\b2[5-9]\b[^.]{0,120}7\.7"], []),
    (NETP, 1, None, "np3", "count", "Quante energie?", "9 (7.7-200 GeV)",
     [r"\b(?:nine|9)\b[^.]{0,40}(?:energ|√s|values|collision)"], []),
    (NETP, 2, None, "np4", "trend", "Come varia C1 con l'energia?", "diminuisce da ≈40 (7.7 GeV) a ≈5 (200 GeV)",
     [r"C_?\{?1\}?[^.]{0,140}(?:decreas|drop|fall|declin|lower)", r"(?:mean|C1)[^.]{0,60}(?:decreas|drop|fall|declin)"], []),
    (NETP, 2, None, "np5", "value", "Quanto valgono i cumulanti periferici (70-80%)?", "vicini a zero",
     [r"(?:70.?80|peripheral)[^.]{0,140}(?:near|close to|consistent with|≈|~|around)\s*(?:zero|0)\b", r"(?:70.?80|peripheral)[^.]{0,140}(?:small|negligible|flat)"], []),
    (NETP, 2, None, "np6", "relation", "Dove è più grande l'incertezza di C4?", "a 7.7 GeV",
     [r"7\.7[^.]{0,140}(?:large|big|larger|largest)[^.]{0,40}(?:uncertaint|error)", r"(?:large|largest|big)[^.]{0,40}(?:uncertaint|error)[^.]{0,100}7\.7"], []),
    # ---------------------------------------------------------------- Twistronics, Fig. 2
    (TWIST, 2, None, "tw1", "caption", "Quali valori di Ω0α/h nei due pannelli?", "2t (a) e 20t (b)",
     [r"(?:2\s*t|2t)\b[^.]{0,80}20\s*t", r"20\s*t[^.]{0,80}\b2\s*t\b"], []),
    (TWIST, 2, "b", "tw2", "value", "A quale energia stanno le bande quasi piatte e il picco del DOS in (b)?", "E/t ≈ −80",
     [r"(?:-|−)\s*80\b"], []),
    (TWIST, 2, None, "tw3", "trend", "Dove si incontrano le bande (coni di Dirac)?", "nel punto Γ",
     [r"(?:Dirac|cone|cross\w*|touch\w*|degenera\w*|meet\w*)[^.]{0,80}(?:Γ|Gamma|\\Gamma)", r"(?:Γ|Gamma|\\Gamma)[^.]{0,80}(?:Dirac|cone|cross\w*|touch\w*|degenera\w*)"], []),
    (TWIST, 2, None, "tw4", "trend", "Cosa mostra il DOS al centro?", "un picco stretto (bande piatte)",
     [r"(?:sharp|narrow|strong|pronounced)[^.]{0,30}(?:peak|maximum)[^.]{0,60}(?:DOS|density)", r"(?:DOS|density of states)[^.]{0,80}(?:sharp|narrow|strong|pronounced)\s*(?:peak|maximum)"], []),
    # ---------------------------------------------------------------- QAH, Fig. 2(a)
    (QAH, 2, "a", "qa1", "value", "Intorno a quale energia sono le bande?", "E/t ≈ −160 (−Ω0(1−α))",
     [r"(?:-|−)\s*16[0-2]\b"], []),
    (QAH, 2, "a", "qa2", "caption", "C'è un gap tra bande piatte e coni di Dirac per λ = 0?", "no, nessun gap",
     [r"no\s+(?:hard\s+)?gap|gapless|without (?:a )?gap|touch\w*"], [r"(?:opens?|open\w*)\s+(?:a\s+)?gap[^.]{0,40}λ\s*=\s*0"]),
    (QAH, 2, "a", "qa3", "count", "Quante bande?", "6",
     [r"\b(?:six|6)\b[^.]{0,30}bands?"], []),
    # ---------------------------------------------------------------- GQA, Fig. 3, 5, 6
    (GQA, 3, None, "gq1", "count", "Quali modelli sono nel grafico?", "MHA-XXL, GQA-XXL, MQA-XXL, MHA-Large",
     [r"(?=.*MHA-?XXL)(?=.*GQA-?XXL)(?=.*MQA-?XXL)(?=.*MHA-?Large)"], []),
    (GQA, 3, None, "gq2", "value", "Tempo per campione di MHA-XXL?", "≈1.5 ms, contro ≈0.3 di GQA/MQA",
     [r"1\.5\d*\s*ms", r"MHA-?XXL[^.;]{0,80}1\.5", r"1\.5[^.;]{0,60}MHA-?XXL"], []),
    (GQA, 3, None, "gq3", "value", "Prestazione di GQA-XXL?", "≈47.1, vicina a MHA-XXL (≈47.2)",
     [r"\b47\.[0-2]\d*"], []),
    (GQA, 5, None, "gq4", "value", "Prestazione di MQA senza uptraining (α = 0)?", "≈54",
     [r"MQA[^.;]{0,100}\b5[34](?:\.\d+)?\b", r"\b5[34](?:\.\d+)?\b[^.;]{0,60}MQA"], []),
    (GQA, 5, None, "gq5", "relation", "Come si confrontano GQA e MQA al variare di α?", "GQA sempre sopra MQA, entrambe salgono",
     [r"GQA[^.]{0,100}(?:above|higher|better|outperform|starts? high)", r"MQA[^.]{0,100}(?:below|lower|worse)"], []),
    (GQA, 6, None, "gq6", "trend", "Come cresce il tempo con il numero di gruppi?", "piatto fino a 8, cresce rapidamente a 32-64",
     [r"(?:32|64)[^.]{0,100}(?:increas|grow|sharp|steep|rise|jump)", r"(?:increas|grow|sharp|steep|rise|jump)\w*[^.]{0,100}(?:32|64)"], []),
    (GQA, 6, None, "gq7", "value", "Tempo di MHA (e di GQA a 64 gruppi)?", "≈2.5 s",
     [r"\b2\.[4-6]\d*\s*s\b", r"(?:MHA|64)[^.;]{0,80}\b2\.[4-6]"], []),
    # ---------------------------------------------------------------- anti-kt, Fig. 2(a) and 4
    (AKT, 2, "a", "ak1", "trend", "Com'è la distribuzione delle aree dell'anti-kt?", "un picco stretto ad area = πR² (1)",
     [r"anti[- ]?k_?\{?t\}?[^.]{0,140}(?:peak\w*|spike|sharp\w*|concentrated|narrow|delta)", r"(?:peak\w*|spike|sharp\w*|narrow)[^.]{0,80}anti[- ]?k_?\{?t\}?"], []),
    (AKT, 2, "a", "ak2", "caption", "Come è riscalata la curva anti-kt?", "× 0.1",
     [r"0\.1\b", r"scaled"], []),
    (AKT, 2, "a", "ak3", "relation", "Quale algoritmo ha la distribuzione più larga?", "kt",
     [r"(?<!anti-)(?<!anti )\bk_?\{?t\}?\b[^.]{0,100}(?:broad\w*|wide\w*|spread)", r"(?:broad\w*|widest|spread)[^.]{0,60}(?<!anti-)(?<!anti )\bk_?\{?t\}?\b"], []),
    (AKT, 4, None, "ak4", "relation", "Come si confronta il back-reaction dell'anti-kt?", "molto più stretto (|Δpt| ≲ 4 GeV contro ±20)",
     [r"anti[- ]?k_?\{?t\}?[^.]{0,160}(?:narrow\w*|small\w*|suppress\w*|less|reduced|much lower|minimal)"], []),
    (AKT, 4, None, "ak5", "caption", "Che scala ha l'asse y?", "logaritmica",
     [r"\blog\w*"], []),
    # ---------------------------------------------------------------- Transformer-XL, Fig. 2 (diagram)
    (TXL, 2, None, "tx1", "caption", "Con che lunghezza di segmento è illustrato il modello?", "4",
     [r"segment\s+length\s+(?:of\s+)?4\b", r"\b(?:4|four)[- ]token segments?"], []),
]


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gold", "figure_checks.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for paper, fig, panel, cid, kind, q, a, accept, reject in CHECKS:
            f.write(json.dumps({"paper": paper, "figure": fig, "panel": panel, "id": cid, "kind": kind,
                                "question": q, "answer": a, "accept": accept, "reject": reject,
                                "annotator": "claude-draft", "reviewed": False}, ensure_ascii=False) + "\n")
    print(f"[+] {len(CHECKS)} checks on {len({(c[0], c[1]) for c in CHECKS})} figures -> {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
