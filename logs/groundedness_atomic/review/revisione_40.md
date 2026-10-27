# Revisione di 40 claim

Per ogni claim leggi le frasi del paper e scrivi dopo `Etichetta:` una sola di queste:

- **TRUE**: il paper dice questo, anche con parole diverse.
- **PARTIAL**: il paper dice qualcosa di simile, ma il claim aggiunge, generalizza o esagera un dettaglio (un numero, "vari" al posto di "uno", un motivo non dato dal paper).
- **FALSE**: il paper dice il contrario, o il claim non c'entra con quanto scritto.
- **?**: le frasi mostrate non bastano per decidere.

Non serve capire la fisica: confronta il claim con le frasi. Il campo (es. `results`) dice che tipo di affermazione è: `future_work` = lavoro futuro proposto, `limitations` = limiti dichiarati, `experiments` = cosa è stato fatto, `results` = cosa è stato trovato.

Rispondi senza cercare altro: se serve il paper intero, scrivi `?`.

## Claim 01
**Paper:** Quantum anomalous Hall phase in synthetic bilayers via twistless twistronics
**Campo:** main_contributions

**Claim:** The proposed cold-atom experimental implementation uses superlattice or holographic imprinting.

- Here the NN tunnelling is standard and con- stant, t(r) = t. The staggered chemical potential is given by⃗ μ = μ(ˆx + ˆy), ˆn = 2d(ˆx + ˆy), and can be relatively easily re- alized using super-lattice or holographic potential imprinting methods. The challenging part here is relate to the phases as- sociated with the next-nearest neighbor complex tunnelings, set to φL(⃗r) -φR(r) = π, where φR(⃗r) = (2r.1y + 1)π/2.
- However these two effect compete and the system can be gapless for differernt values of λ (see Fig 10(b)). A. Basic experimental scheme We proposed in Ref. [91] to use a subset of four states out of the large nuclear spin manifold I = 9/2 of 87Sr, or 173Yb (I = 5/2).
- Recently, the Hamburg group combined the studies of the Haldane model with the use of machine learning methods [111–113]. Super-lattice and holographic potential imprinting methods These are nowadays standard methods, developed already many years ago and described in textbooks such as [64]. They have a plethora applications ranging from de- signing traps of special shape, through creation of dimerized lattices, to imprinting random potentials.

**Etichetta:** TRUE 

---

## Claim 02
**Paper:** GQA: Training Generalized Multi-Query Transformer Models (informatica, modelli T5)
**Campo:** experiments

**Claim:** Models were evaluated on various question-answering datasets after undergoing uptraining procedures.

- For α = 0.05, training took approxi- mately 600 TPUv3 chip-days. Data We evaluate on summarization datasets CNN/Daily Mail (Nallapati et al., 2016), arXiv and PubMed (Cohan et al., 2018), MediaSum (Zhu et al., 2021), and Multi-News (Fabbri et al., 2019); 1https://github.com/google/flaxformer Model Tinfer Average CNN arXiv PubMed MediaSum MultiNews WMT TriviaQA s R1 R1 R1 R1 R1 BLEU F1 MHA-Large 0.37 46.0 42.9 44.6 46.2 35.5 46.6 27.7 78.2 MHA-XXL 1.51 47.2 43.8 45.6 47.5 36.4 46.9 28.4 81.9 MQA-XXL 0.24 46.6 43.0 45.0 46.9 36.1 46.5 28.5 81.3 GQA-8-XXL 0.28 47.1 43.5 45.4 47.7 36.3 47.2 28.4 8
- We eval- 0 0.5 1 1.5 46 46.5 47 MHA-Large MHA-XXL MQA-XXL GQA-XXL Time per sample (ms) Performance Figure 3: Uptrained MQA yields a favorable tradeoff compared to MHA with higher quality and faster speed than MHA-Large, and GQA achieves even better performance with similar speed gains and comparable quality to MHA-XXL. Average perfor- mance on all tasks as a function of average inference time per sample for T5-Large and T5-XXL with multi- head attention, and 5% uptrained T5-XXL with MQA and GQA-8 attention. uate performance on a representive subsample of tasks: CNN/Daily Mail, (short-form summ
- We apply MQA and GQA to decoder self-attention and cross-attention, but not encoder self-attention. Uptraining Uptrained models are initialized from public T5.1.1 checkpoints. The key and value heads are mean-pooled to the appropriate MQA or GQA structure, and then pre-trained for a further α proportion of original pre-training steps with the original pre-training setup and dataset from (Raffel et al., 2020).

**Etichetta:** PARTIAL

---

## Claim 03
**Paper:** The anti-k_t jet clustering algorithm (fisica delle particelle)
**Campo:** future_work

**Claim:** Investigating applicability in diverse collider environments with varying levels of background activity.

- They are however the identifying characteristics of the collinear unsafe plain “iterative cone” algorithm, for which the anti-kt algorithm provides a natural, fast, infrared and collinear safe replacement. 1 Introduction and definition Jet clustering algorithms are among the main tools for analysing data from hadronic collisions. Their widespread use at the Tevatron and the prospect of unprecedented final-state complexity at the upcoming Large Hadron Collider (LHC) have stimulated considerable debate concerning the merits of different kinds of jet algorithm. Part of the discussion has centred 
- In this respect we recall that they do deviate from πR2 for configurations with several neighbouring hard particles. A summary of these and other properties of the jet area is given in table 1, together with a comparison to other IRC safe algorithms as studied in [4]. Fig. 2 illustrates the distribution of areas in dijet events at the LHC (generated with Pythia 6.4, with a transverse momentum cut of 1 TeV on the 2 →2 scattering, considering those among the two hardest jets in each event that have |y| < 2), compared with those for other algorithms, and one sees a near δ-function at an area of π
- The key feature above is that the soft particles do not modify the shape of the jet, while hard particles do. I.e. the jet boundary in this algorithm is resilient with respect to soft radiation, but flexible with respect to hard radiation.3 The behaviours of different jet algorithms are illustrated in fig. 1. We have taken a parton-level event together with ∼104 random soft ‘ghost’ particles (as in [4]) and then clustered them with 4 different jet algorithms.

**Etichetta:** ?

---

## Claim 04
**Paper:** The anti-k_t jet clustering algorithm (fisica delle particelle)
**Campo:** problem_addressed

**Claim:** To propose a new jet clustering algorithm ('anti-kt') that maintains consistency and simplicity while addressing limitations of existing algorithms regarding soft radiation influence.

- The case of p = 0 is special and it corresponds to the inclusive Cambridge/Aachen algorithm. 1This is discussed in the appendix in detail for the iterative cone, and there we also introduce the terminology iterative cone with split–merge steps (IC-SM) and iterative cone with progressive removal (IC-PR), so as to distinguish the two broad classes of iterative cone algorithms. 2 Negative values of p might at first sight seem pathological. We shall see that they are not.2 The behaviour with respect to soft radiation will be similar for all p < 0, so here we will concentrate on p = -1, and refer t
- Rather surprisingly, taking p to be negative also yields an algorithm that is infrared and collinear safe and has sensible phenomenological behaviour. We have specifically studied p = -1, the “anti-kt” algorithm, and highlighted various simple theoretical properties, notably the resilience of its jet boundaries with respect to soft radiation. The other properties that we’ve discussed are essentially consequences of this feature.
- For each of the partonic jets, we have shown the region within which the random ghosts are clustered into that jet. For the kt and Cambridge/Aachen algorithms, that region depends somewhat on the specific set of ghosts and the jagged borders of the jets are a consequence of the randomness of the ghosts — the jet algorithm is adaptive in its response to soft particles, and that adaptiveness applies also to the ghosts which take part in the clustering. For SISCone one sees that single-particle jets are regular (though with a radius R/2 — as pointed out in [4]), while composite jets have more var

**Etichetta:**  TRUE

---

## Claim 05
**Paper:** Out-of-time-order correlators in quantum mechanics (billiardi, caos quantistico)
**Campo:** future_work

**Claim:** Further investigation is needed into higher-energy regimes to identify potential exponential growth patterns.

- At a very early time t ≲0.01, one may be able to argue that there is an exponential region. However, to find the exponential growth eλt, we need much longer time than 1/λ. (Otherwise, we cannot distinguish the exponential and linear functions.) Moreover, a similarly- looking behavior can be found even for the circle billiard.
- To answer these questions, we need further study of OTOCs of classically chaotic systems. 4This similarity was discussed and further generalized in [32–38], for example to a generalized fluctuation-dissipation theorem [37]. – 19 – The OTOC for the Sachdev-Ye-Kitaev (SYK) model grows exponentially [11, 12]. What was essential for the exponential growth? There are two significant difference between the SYK and our examples.
- The random coupling is known to enhances the decoherence [42]. Adding to that, it is known that the wave function is localized in space when the system has random potential (Anderson localization). The localization of the wave function would be regarded as emergence of particle nature and thus the exponential growth of the OTOC might be expected since the presence of the wave nature in our model prevents us to observe the exponential growth.

**Etichetta:** PARTIAL

---

## Claim 06
**Paper:** The anti-k_t jet clustering algorithm (fisica delle particelle)
**Campo:** future_work

**Claim:** Investigate multi-scale processes with large pileup.

- One should also bear in mind that top reconstruction, near threshold, with a moderate jet radius and no pileup is a relatively simple application for inclusive jet algorithms. One expects that greater differences between algorithms will be seen in other contexts, an extreme example being multi-scale processes with large pileup. 3 Conclusions There starts to be a certain choice of infrared and collinear safe inclusive jet algorithms for hadron colliders. As we have seen, some of these (kt and Cambridge/Aachen) belong to a more general class of sequential recombination algorithms, parametrised b
- We note that the difference between various choices of R can be substantially greater than the differences between the various algorithms at a given R. One should also bear in mind that top reconstruction, near threshold, with a moderate jet radius and no pileup is a relatively simple application for inclusive jet algorithms. One expects that greater differences between algorithms will be seen in other contexts, an extreme example being multi-scale processes with large pileup. 3 Conclusions There starts to be a certain choice of infrared and collinear safe inclusive jet algorithms for hadron c
- They are however the identifying characteristics of the collinear unsafe plain “iterative cone” algorithm, for which the anti-kt algorithm provides a natural, fast, infrared and collinear safe replacement. 1 Introduction and definition Jet clustering algorithms are among the main tools for analysing data from hadronic collisions. Their widespread use at the Tevatron and the prospect of unprecedented final-state complexity at the upcoming Large Hadron Collider (LHC) have stimulated considerable debate concerning the merits of different kinds of jet algorithm. Part of the discussion has centred 

**Etichetta:** TRUE

---

## Claim 07
**Paper:** GQA: Training Generalized Multi-Query Transformer Models (informatica, modelli T5)
**Campo:** experiments

**Claim:** Models were evaluated on various translation datasets after uptraining with different proportions of original pre-training steps.

- Uptraining Uptrained models are initialized from public T5.1.1 checkpoints. The key and value heads are mean-pooled to the appropriate MQA or GQA structure, and then pre-trained for a further α proportion of original pre-training steps with the original pre-training setup and dataset from (Raffel et al., 2020). For α = 0.05, training took approxi- mately 600 TPUv3 chip-days.
- Multi-query atten- tion reduces this overhead at the cost of decreased model capacity and quality. We propose to convert multi-head attention models to multi-query models with a small fraction of original pre-training com- pute. Moreover, we introduce grouped-query atten- tion, an interpolation of multi-query and multi-head attention that achieves quality close to multi-head at comparable speed to multi-query attention.
- However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) intro- duce grouped-query attention (GQA), a gener- alization of multi-query attention which uses an intermediate (more than one, less than num- ber of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA. 1 Introduc

**Etichetta:** ?

---

## Claim 08
**Paper:** Flux tube in turbulent flow and quantum chromodynamics
**Campo:** results

**Claim:** In the turbulent-flow model, the transverse fluctuation function of the cylindrically symmetric tube-like solution decays toward zero at infinity.

- There exists a turbulent flow moving in z direction. In the flow exists a core (tube) with the velocity profile given in Fig. 1, curve 1; the z-component of the velocity at the infinity is v∞̸ = 0, ρ →∞; the profile of statistical fluctuations describing by the function φ is given by curve 2 in Fig. 1; as φ →0 then at the infinity the flow become laminar one. It means that we have to choose the solution in such a way that v∞becomes small enough that flow becomes laminar one.
- If the viscosity μ = 0 then we have the ideal fluid. We want to find cylindrically symmetric solution of the Navier - Stokes equations but not in a pipe. For the ideal (ν = 0) and laminar fluids the equation is vρ dvz dρ = ν d2vz dρ2 (7) but vρ = 0 for the flow along axes z.
- arXiv:0907.3624v2 [physics.flu-dyn] 7 Aug 2009 Flux tube in turbulent flow and quantum chromodynamics Vladimir Dzhunushaliev Institut f ̈ur Physik, Universit ̈at Oldenburg, Postfach 2503, D-26111 Oldenburg, Germany; Department of Physics and Microelectronic Engineering, Kyrgyz-Russian Slavic University, Bishkek, Kievskaya Str. 44, 720021, Kyrgyz Republic and Institute of Physics of National Academy of Science Kyrgyz Republic, 265 a, Chui Street, Bishkek, 720071, Kyrgyz Republic∗ (Dated: October 24, 2018) Using some assumptions about the correlation function of velocity in a turbulent flow, a c

**Etichetta:** TRUE

---

## Claim 09
**Paper:** Quantum anomalous Hall phase in synthetic bilayers via twistless twistronics
**Campo:** main_contributions

**Claim:** Uniform next-to-nearest-neighbor hopping cannot reach the nonstandard topology.

- [91] and demonstrate that such sys- tem is very flexible and can be tweaked to exhibit topological band structure in various situations. In particular, to achieve non-trivial topology we engineer artificial complex next-to- nearest neighbor tunneling analogous to the ones appearing in the Haldane model [32]. Typically, the energy bands of our interest in this system form three groups and the Chern num- ber changes from trivial (0,0,0), to a topological phase with a trivial flat band (-1,0,1) and a topological phase with a non- tivial flat band (-1,1,0).
- In this paper we extend these ideas and demonstrate that our system exhibits topological band structures under appropriate conditions. To achieve non-trivial band topology we consider imanaginary next-to-nearest neighbor tunnelings that drive the system into a quantum anomalous Hall phase. In particular, we focus on three groups of bands, whose their Chern numbers triplet can be associated to a trivial insulator (0,0,0), a standard non-trivial (-1,0,1) and a non-standard non-trivial (-1,1,0).
- We also observe a gap clos- ing and reopening of the lower gap for increasing staggering potential with opposite sign. The latter is reminiscent of the interplay between the staggering potential and the imaginary next-to-nearest neighbor hopping in the Lieb lattice [92]. In fact, the system undergoes a topological phase transition.

**Etichetta:** ?

---

## Claim 10
**Paper:** Out-of-time-order correlators in quantum mechanics (billiardi, caos quantistico)
**Campo:** results

**Claim:** For a Gaussian wavepacket in a box, the 2-point OTOC matches the classical step-function prediction before the Ehrenfest time.

- Here, l(t) = 2k0t+x0 -(2N +1) is the difference between the center of the wavepacket and the right boundary x = 1. Before the – 16 – Ehrenfest time σt ≪1, the 2-point OTOC is approximated by a step function. Therefore, for a well localized wavepacket in a box, we have a quantum-classical correspondence: [x(t), p(0)] ∼i{x(t), p(0)}PB , (t ≪tE) .
- (B.55) – 28 – Therefore, we obtain cφ = 1 + 128 π 1/2 2π3σk2 0(1 + cos2 πx0) 3 sin3 πx0 + 1 σ3 sin πx0  t2e-σ2l2 2t2 ≃1 + 42.0σk2 0(1 + cos2 πx0) sin3 πx0 + 6.38 σ3 sin πx0  t2e-σ2l2 2t2 . (B.56) In addition to the classical prediction of the 4-point OTOC, cclassical φ = 1, we find the gaussian spike at the bounce. References [1] A. I. Larkin and Y. N. Ovchinnikov, JETP 28, 6 (1969): 1200-1205.
- Why do the OTOCs deviate from their classical value at a high temperature and t ≪tE? In section.5.2, we found the other time scale ts, at which the quantum- classical correspondence of the 4-point OTOC is violated, for a wavepacket in a box. We showed that ts is sufficiently smaller than the Ehrenfest time tE.

**Etichetta:** TRUE

---

## Claim 11
**Paper:** GQA: Training Generalized Multi-Query Transformer Models (informatica, modelli T5)
**Campo:** experiments

**Claim:** Experiments use T5.1.1 models based on T5 XXL.

- We apply MQA and GQA to decoder self-attention and cross-attention, but not encoder self-attention. Uptraining Uptrained models are initialized from public T5.1.1 checkpoints. The key and value heads are mean-pooled to the appropriate MQA or GQA structure, and then pre-trained for a further α proportion of original pre-training steps with the original pre-training setup and dataset from (Raffel et al., 2020).
- We note that GQA is not applied to the encoder self-attention layers; encoder representations are computed in parallel, and memory bandwidth is therefore generally not the primary bottleneck. 3 Experiments 3.1 Experimental setup Configurations All models are based on the T5.1.1 architecture (Raffel et al., 2020), im- plemented with JAX (Bradbury et al., 2018), Flax (Heek et al., 2020), and Flaxformer1. For our main experiments we consider T5 Large and XXL with multi-head attention, as well as up- trained versions of T5 XXL with multi-query and grouped-query attention. We use the Adafactor op- 
- Therefore, we expect GQA to present a particularly good trade-off for larger models. We note that GQA is not applied to the encoder self-attention layers; encoder representations are computed in parallel, and memory bandwidth is therefore generally not the primary bottleneck. 3 Experiments 3.1 Experimental setup Configurations All models are based on the T5.1.1 architecture (Raffel et al., 2020), im- plemented with JAX (Bradbury et al., 2018), Flax (Heek et al., 2020), and Flaxformer1. For our main experiments we consider T5 Large and XXL with multi-head attention, as well as up- trained versi

**Etichetta:** PARTIAL

---

## Claim 12
**Paper:** Out-of-time-order correlators in quantum mechanics (billiardi, caos quantistico)
**Campo:** future_work

**Claim:** Further investigation is needed into whether single-particle QM exhibits clear exponential growth in thermal OTOCs.

- How- ever, in our analyses of the quantum stadium billiards, we do not find the exponential growth of the OTOC. Is there single particle quantum mechanics which shows clear exponential growth in the thermal OTOC? Can the quantum Lyapunov exponent saturate the bound provided in Ref.[2]?
- In the inset of the right panel, we show the early time evolution of the OTOC. We find no clear exponential growth of the OTOC. Among our main results, we show two numerical results in Fig. 1, which shows typical behavior of the OTOCs.
- Fig.8 shows an early time evolution of thermal OTOCs for stadium and circle billiards. The OTOC for the stadium billiard does not show a clear exponential growth. At a very early time t ≲0.01, one may be able to argue that there is an exponential region.

**Etichetta:** TRUE

---

## Claim 13
**Paper:** Out-of-time-order correlators in quantum mechanics (billiardi, caos quantistico)
**Campo:** results

**Claim:** No clear exponential growth is found in the early-time thermal OTOC of the stadium billiard for T ≲ 400.

- In the inset of the right panel, we show the early time evolution of the OTOC. We find no clear exponential growth of the OTOC. Among our main results, we show two numerical results in Fig. 1, which shows typical behavior of the OTOCs.
- Fig.8 shows an early time evolution of thermal OTOCs for stadium and circle billiards. The OTOC for the stadium billiard does not show a clear exponential growth. At a very early time t ≲0.01, one may be able to argue that there is an exponential region.
- How- ever, in our analyses of the quantum stadium billiards, we do not find the exponential growth of the OTOC. Is there single particle quantum mechanics which shows clear exponential growth in the thermal OTOC? Can the quantum Lyapunov exponent saturate the bound provided in Ref.[2]?

**Etichetta:** PARTIAL

---

## Claim 14
**Paper:** Nonmonotonic energy dependence of net-proton number fluctuations (STAR, RHIC)
**Campo:** results

**Claim:** In the Monte Carlo test, 1143 of one million randomly varied data sets had the same derivative sign at all energies.

- Then for each new κσ2 versus √sNN set of points, a fourth order polynomial function is fitted and the derivative values are calculated at different √sNN (as discussed above). A total of 1143 sets were found to have the same derivative sign at all √sNN. The probability that at least one derivative at a given √sNN has a different sign is found to be 0.998857, which corresponds to 3.1σ.
- The over- all significance of the change in the sign of the slope for κσ2 versus √sNN, based on the fourth order polynomial func- tion fitting procedure from √sNN = 7.7 to 62.4 GeV, is 3.1σ. This significance is obtained by generating one million sets of points, where for each set, the measured κσ2 value at a given √sNN is randomly varied within the total Gaussian uncertain- ties (systematic and statistical uncertainties added in quadra- ture). Then for each new κσ2 versus √sNN set of points, a fourth order polynomial function is fitted and the derivative values are calculated at different √sN
- A cross-check using a different method based on unfolding [34] of the distributions for central Au+Au collisions at √sNN = 200 GeV has been found to give values consistent with the cumulants shown in Fig. 2. Further, the efficiency correction method used has been verified in a Monte Carlo calculation. Typical values for the efficiencies in the TPC (TOF-matching) for the momentum range stud- ied in 0-5% central Au+Au collisions at √sNN = 7.7 GeV are 83%(72%) and 81%(70%) for the protons and anti-protons, respectively.

**Etichetta:** TRUE

---

## Claim 15
**Paper:** Nonmonotonic energy dependence of net-proton number fluctuations (STAR, RHIC)
**Campo:** limitations

**Claim:** Higher collision energies might affect reconstruction efficiencies differently.

- The transverse momentum (pT) range for the mea- surements is from 0.4 to 2 GeV/c and the rapidity (y) range is -0.5 < y < 0.5. those shown in Fig 2 . The cumulants are also corrected for finite track reconstruction efficiencies of the TPC and TOF detectors. This is done by assuming a binomial response of the two detectors [35, 38].
- These require- ments include the distance of the proton and anti-proton tracks from the primary vertex position, track quality reflected by the number of TPC space points used in the track reconstruction, the particle identification criteria passing certain selection cri- teria, and the uncertainties in estimating the reconstruction ef- ficiencies. The systematic uncertainties at different collision energies are uncorrelated. The large values of C3 and C4 for central Au+Au collisions show that the distributions have non-Gaussian shapes, a possi- ble indication of enhanced fluctuations arising 
- Experimental tests of thermaliza- tion for these femto-scale expanding systems are non-trivial. However, the yields of produced hadrons and fluctuations of multiplicity distributions related to conserved quantities have been studied and shown to have characteristics of thermody- namic equilibrium for higher collision energies [10, 15–20]. Upon approaching a critical point, the correlation length di- verges and thus renders, to a large extent, microscopic details irrelevant.

**Etichetta:** ?

---

## Claim 16
**Paper:** Sachdev–Ye–Kitaev model as Liouville quantum mechanics (SYK, fisica teorica)
**Campo:** methodology

**Claim:** The proposed methodology focuses on soft reparameterization modes.

- To perform the crucial third step of the program we need to know the effective theory gov- erning the reparameterization modes. An effective action for the f-modes was recently proposed by Maldacena, Stanford, and Yang [4, 10]. In the next section we extend this construction by deriving a generalized form of the action and the corresponding integration measure, and by in- troducing a variable transform which will be instrumental to all further calculations.
- We will show that this program leads to the emergence of Liouville quantum mechanics in the picture and in the consequence to universal ∝τ-3/2 scaling of all long-time correlation functions. To perform the crucial third step of the program we need to know the effective theory gov- erning the reparameterization modes. An effective action for the f-modes was recently proposed by Maldacena, Stanford, and Yang [4, 10].
- Sachdev-Ye-Kitaev Model as Liouville Quantum Mechanics Dmitry Bagretsa, Alexander Altlanda, Alex Kamenevb aInstitut f ̈ur Theoretische Physik, Universit ̈at zu K ̈oln, Z ̈ulpicher Straße 77, 50937 K ̈oln, Germany bW. I. Fine Theoretical Physics Institute and School of Physics and Astronomy, University of Minnesota, Minneapolis, MN 55455, USA Abstract We show that the proper inclusion of soft reparameterization modes in the Sachdev-Ye-Kitaev model of N randomly interacting Majorana fermions reduces its long-time behavior to that of Liouville quantum mechanics. As a result, all zero temperature 

**Etichetta:** TRUE

---

## Claim 17
**Paper:** The anti-k_t jet clustering algorithm (fisica delle particelle)
**Campo:** limitations

**Claim:** The anti-kt algorithm may require additional computational resources for extremely large datasets.

- It too would have the property of soft-resilience, but would differ perturbatively from IC-PR at NLO. This, together with the issue of computational speed, leads us to prefer the anti-kt algorithm as a replacement. References [1] S. Catani, Y. L. Dokshitzer, M.
- This means that asymptotically the algorithm takes a time O  N 3/2 to cluster N particles (rather than N ln N for the kt algorithm). However for N ≲20000 the FastJet implementation [21] in any case uses other strategies, which are insensitive to this issue, and the anti-kt clustering is then as fast as kt clustering. 2.4 Example application: top reconstruction One may wonder whether the unusual soft-resilience of the anti-kt algorithm leads to poorer results in phenomenological applications. We have investigated various examples and found that in general this is not the case.
- However in the anti-kt algorithm these properties are obtained without having to pay the price of collinear unsafety. Therefore the anti-kt algorithm should be a good candidate as a replacement algorithm for IC-PR algorithms. One might worry that the resilience (or rigidity) with respect to soft radiation could worsen 9 the practical performance of the tool.

**Etichetta:** TRUE

---

## Claim 18
**Paper:** Simulating twistronics without a twist (atomi freddi, doppio strato sintetico)
**Campo:** experiments

**Claim:** Experimental feasibility is discussed with realistic strontium parameters.

- This results in a large light shift of the 3P1 ex- cited state of amplitude δ, leading to a detuning of the Raman beams ∆(x, y) = ∆0 + δ(1 + cos (2πx/lx) cos (2πy/ly)). Its effect is to modulate the Raman coupling amplitude Ω(x, y) ≃Ω0[(1 -α) -α cos (2πx/lx) cos (2πy/ly)], with α = δ/∆0 ∼0.2 for realistic experimental parameters, see Fig. 1(c) [46, 47]. We therefore name it “modulation laser”.
- In order to demonstrate that the band struc- tures obtained for Θ(4, 4) are not generic, we illustrate ex- emplary results for the configurations Θ(4, 7) and Θ(4, 8) in Fig. S5(a-b). The chosen parameters have been selected for experimental convenience. We again focus in the vicinity of E/t = -Ω0(1 -α).
- Although we focus in our illustrative example on a particular spatial modulation, general interlayer coupling patterns can be experimentally induced including quasi-perioidic or Moiré-like patterns. Our proposal can be realized with fermionic two-electron atoms, such as strontium or ytterbium, using available experimental techniques. arXiv:1912.12736v2 [cond-mat.quant-gas] 22 Oct 2020 2 Concept. We consider a two-dimensional Fermi gas with four internal states, indicated here generically as spin states {m, σ} = ±1/2.

**Etichetta:** TRUE

---

## Claim 19
**Paper:** Quantum anomalous Hall phase in synthetic bilayers via twistless twistronics
**Campo:** problem_addressed

**Claim:** Engineering interesting topological band structures in synthetic bilayers without physical twisting.

- The latter could be done by a suitable quench protocol [122, 123]. VII. CONCLUSIONS - OUTLOOK In the present paper we developed further the idea of “twistronics without a twist” and demonstrated that it can be used to engineer interesting topological band structures un- der various conditions. Focussing on a square lattice system with synthetic dimensions, we showed the appearance of an anomalous Hall phase in presence of artificial complex next- to-nearest neighbor interlayer tunneling.
- The latter feature facilitates the engineering of flat bands at larger "magic" angles, or more directly, for smaller unit cells than in conventional twisted materials. In this paper we extend these ideas and demonstrate that our system exhibits topological band structures under appropriate conditions. To achieve non-trivial band topology we consider imanaginary next-to-nearest neighbor tunnelings that drive the system into a quantum anomalous Hall phase.
- Quantum anomalous Hall phase in synthetic bilayers via twistless twistronics Tymoteusz Salamon,1 Ravindra W. Chhajlany,2 Alexandre Dauphin,1 Maciej Lewenstein,1, 3 and Debraj Rakshit1, 4 1ICFO - Institut de Ciencies Fotoniques, The Barcelona Institute of Science and Technology, Av.

**Etichetta:** TRUE

---

## Claim 20
**Paper:** Out-of-time-order correlators in quantum mechanics (billiardi, caos quantistico)
**Campo:** results

**Claim:** The harmonic-oscillator OTOC satisfies c_n(t) = C_T(t) = cos^2(omega t).

- In the case of the harmonic oscillator, on the other hand, the classical solutions, x(t) and p(t), are completely identical to Eq.(3.5). Therefore, classical statistics gives the same result as the quantum calculation: C(harmonic) cl (t) = cos2 ωt. 5.2 Out-of-time-order correlator for a wavepacket Why does quantum statistics not approach classical statistics? To answer the ques- tion, we consider a simpler setup: OTOC for a wavepacket in a 1D box.
- The OTOC of a particle in a box is periodic in time because of its commen- surable energy spectrum, while that of the stadium billiards saturates to a constant value. 3. The OTOCs grow with temperature except for the harmonic oscillator. The high temperature limit does not reproduce the classical value in general.
- In appendix.A, we check that our results do not depend on the truncation when we take the truncation cut-offsufficiently large. 3 Integrable examples 3.1 Harmonic oscillator For concreteness, we will show some explicit calculation for the OTOC in integrable systems. One of the simplest integrable examples is the 1D harmonic oscillator, H = p2 + ω2 4 x2 . (3.1) 2 In the definition of the thermal OTOC in Ref.

**Etichetta:** PARTIAL

---

## Claim 21
**Paper:** Simulating twistronics without a twist (atomi freddi, doppio strato sintetico)
**Campo:** limitations

**Claim:** Dirac features are not generic.

- Non-magic configurations As already mentioned before, the proposed scheme can be exploited to engineer a broad range of band structures by simply manipulating the periodicity of the spatial modula- tion of the Raman lasers in the square lattice under consid- eration, or by modifying the lattice structure in itself, along with other parameters, such as Raman coupling strength and magnetic flux. In order to demonstrate that the band struc- tures obtained for Θ(4, 4) are not generic, we illustrate ex- emplary results for the configurations Θ(4, 7) and Θ(4, 8) in Fig. S5(a-b). The chosen parameter
- To illustrate this concept, we focus on a synthetic square bilayer lattice and show that it leads to tunable quasi-flatbands and Dirac cone spectra under certain magic supercell periodicities. The appearance of these features are explained using a perturbative analysis. Our proposal can be implemented using available state-of-the-art experimental techniques, and opens the route towards the controlled study of strongly-correlated flat band accompanied by hybridization physics akin to magic angle bilayer graphene in cold atom quantum simulators.
- Black lines denote the tunneling between the Lieb lattice sites generated. They are responsible for the main features of the band structures centered at ±Ω0(1 -α) described in the main text: a flat band intersected by the Dirac cones. The effects of the grey sites are only taken into account in second order perturbation theory (see Fig.

**Etichetta:** PARTIAL

---

## Claim 22
**Paper:** Flux tube in turbulent flow and quantum chromodynamics
**Campo:** limitations

**Claim:** Perfect fluids are modeled by Euler equations.

- It is necessary to emphasize that between the turbulence and quantum chromodynamics there is a difference. Any field theory has Lagrangian and field equations are corresponding Euler equations for this Lagrangian. The Navier-Stokes equtions are not Euler equations in the consequence of the presence of viscocity. 8 VI.
- Any field theory has Lagrangian and field equations are corresponding Euler equations for this Lagrangian. The Navier-Stokes equtions are not Euler equations in the consequence of the presence of viscocity. 8 VI. ACKNOWLEDGMENTS I am grateful to the Research Group Linkage Program of the Alexander von Humboldt Foundation for financial support, to J.
- For example, the “electric” field operator, bEB z = ∂0 b AB z -∂z b AB 0 + gf BCDAC 0 AD z giving the expectation ⟨bEB z ⟩. The simple gauge field expectation values, ⟨Aμ(x)⟩, are obtained by average Eq. (38) over some quantum state |Q⟩ D Q ∂ν bFBμν Q E = 0.

**Etichetta:** PARTIAL

---

## Claim 23
**Paper:** GQA: Training Generalized Multi-Query Transformer Models (informatica, modelli T5)
**Campo:** experiments

**Claim:** Models were evaluated on various translation datasets after undergoing uptraining procedures.

- We apply MQA and GQA to decoder self-attention and cross-attention, but not encoder self-attention. Uptraining Uptrained models are initialized from public T5.1.1 checkpoints. The key and value heads are mean-pooled to the appropriate MQA or GQA structure, and then pre-trained for a further α proportion of original pre-training steps with the original pre-training setup and dataset from (Raffel et al., 2020).
- For α = 0.05, training took approxi- mately 600 TPUv3 chip-days. Data We evaluate on summarization datasets CNN/Daily Mail (Nallapati et al., 2016), arXiv and PubMed (Cohan et al., 2018), MediaSum (Zhu et al., 2021), and Multi-News (Fabbri et al., 2019); 1https://github.com/google/flaxformer Model Tinfer Average CNN arXiv PubMed MediaSum MultiNews WMT TriviaQA s R1 R1 R1 R1 R1 BLEU F1 MHA-Large 0.37 46.0 42.9 44.6 46.2 35.5 46.6 27.7 78.2 MHA-XXL 1.51 47.2 43.8 45.6 47.5 36.4 46.9 28.4 81.9 MQA-XXL 0.24 46.6 43.0 45.0 46.9 36.1 46.5 28.5 81.3 GQA-8-XXL 0.28 47.1 43.5 45.4 47.7 36.3 47.2 28.4 8
- First, we note that GQA already achieves reasonable performance af- ter conversion while MQA requires uptraining to 54.4 54.6 54.8 55 55.2 55.4 55.6 Mean First Random Figure 4: Performance comparison of different check- point conversion methods for T5-Large uptrained to MQA with proportion α = 0.05. ‘Mean’ mean-pools key and value heads, ‘First’ selects the first head and ‘Random’ initializes heads from scratch. be useful. Both MQA and GQA gain from 5% uptraining with diminishing returns from 10%. 0 0.02 0.04 0.06 0.08 0.1 54 55 56 57 Uptraining proportion α Performance MHA GQA MQA Figure 5: P

**Etichetta:** PARTIAL

---

## Claim 24
**Paper:** Out-of-time-order correlators in quantum mechanics (billiardi, caos quantistico)
**Campo:** results

**Claim:** The thermal maximum of the one-dimensional box OTOC is max C_T ~ 0.1672 x 2mTL^2 for mTL^2 >> 1.

- The thermal OTOC also tends to be large at high temperature. We have found that the maximum of the thermal OTOC increases linearly as a function of T: max CT ≃0.1672 × 2mTL2 (mTL2 ≫1) where the size of the box L and the mass of the particle m are restored. We have also checked that the time average of the OTOC is given by ̄CT ≃0.0836 × 2mTL2 (mTL2 ≫1) , (3.9) where ̄CT = limτ→∞ R τ 0 dtCT(t)/τ.
- This method is not useful for other cases since it is difficult (or impossible) to obtain explicit expressions of Heisenberg operators for a general Hamiltonian. 3.2 Particle in a box One of the other integrable examples is a particle in a 1D box. The Hamiltonian for the one-dimensional case is H = p2 + Vbox(x) , Vbox(x) = ( 0 0 < x < 1 ∞ else . (3.6) Eigenfunctions and eigenvalues are given by ψn = √ 2 sin πnx , En = π2n2 , (3.7) where n = 1, 2, · · ·.
- We have found that the maximum of the thermal OTOC increases linearly as a function of T: max CT ≃0.1672 × 2mTL2 (mTL2 ≫1) where the size of the box L and the mass of the particle m are restored. We have also checked that the time average of the OTOC is given by ̄CT ≃0.0836 × 2mTL2 (mTL2 ≫1) , (3.9) where ̄CT = limτ→∞ R τ 0 dtCT(t)/τ. For the particle in a 2D square box, V (x, y) = 0 (0 < x < 1, 0 < y < 1), ∞(else), we can obtain the same result for the thermal OTOC.

**Etichetta:** TRUE

---

## Claim 25
**Paper:** GQA: Training Generalized Multi-Query Transformer Models (informatica, modelli T5)
**Campo:** experiments

**Claim:** Main experiments compare T5 Large and T5 XXL multi-head attention models with 5%-uptrained T5 XXL MQA and GQA-8 models on Multi-News.

- We note that GQA is not applied to the encoder self-attention layers; encoder representations are computed in parallel, and memory bandwidth is therefore generally not the primary bottleneck. 3 Experiments 3.1 Experimental setup Configurations All models are based on the T5.1.1 architecture (Raffel et al., 2020), im- plemented with JAX (Bradbury et al., 2018), Flax (Heek et al., 2020), and Flaxformer1. For our main experiments we consider T5 Large and XXL with multi-head attention, as well as up- trained versions of T5 XXL with multi-query and grouped-query attention. We use the Adafactor op- 
- We eval- 0 0.5 1 1.5 46 46.5 47 MHA-Large MHA-XXL MQA-XXL GQA-XXL Time per sample (ms) Performance Figure 3: Uptrained MQA yields a favorable tradeoff compared to MHA with higher quality and faster speed than MHA-Large, and GQA achieves even better performance with similar speed gains and comparable quality to MHA-XXL. Average perfor- mance on all tasks as a function of average inference time per sample for T5-Large and T5-XXL with multi- head attention, and 5% uptrained T5-XXL with MQA and GQA-8 attention. uate performance on a representive subsample of tasks: CNN/Daily Mail, (short-form summ
- We use greedy decoding for inference. Timing We report time per sample per TPUv4 chip, as measured by xprof (Google, 2020). For timing experiments we use 8 TPUs with the largest batch size that fits up to 32 per TPU, and paral- lelization optimized separately for each model. 3.2 Main results Figure 3 shows average performance over all datasets as a function of average inference time for MHA T5-Large and T5-XXL, and uptrained MQA and GQA-8 XXL models with uptraining proportion α = 0.05. We see that a larger up- trained MQA model provides a favorable trade- off relative to MHA models, with highe

**Etichetta:** ?

---

## Claim 26
**Paper:** SU(4) Heisenberg model on the honeycomb lattice with exchange-frustrated perturbations
**Campo:** future_work

**Claim:** Exploration of extended models accounting for more intricate geometrical structures is proposed.

- These models can be alternatively expressed in terms of pseudospins and pseudo-orbitals that mimic the eg operators [16–20]. Lastly, two-orbital degenerate KK models were proposed as relevant descrip- tions for correlated insulators observed in twistronic sys- tems [21, 22]. This proposal hangs upon the validity of Wannier orbitals to reproduce the twist-induced flat bands.
- (5) is too simplistic to de- scribe Ba3CuSb2O9 and a QSOL is likely not its ground state. For this material, a more realistic Hamiltonian on a decorated honeycomb lattice should be taken into account. In TBG, the orbital degrees of freedom originate from the two Dirac points in the original Brillouin zone of each graphene sheet, which should be centered on a honey- comb superlattice due to symmetry constraints [53–55].
- The same comment applies to the tiny-angle TBG under an electric field, which is possibly described by a KK model on an emergent kagome lattice [61]. Concerning the layered honeycomb material α-ZrCl3, it is expected that extended versions of the minimal model derived in Refs. [19, 20] would lead to exchange frustration simi- lar to the ones discussed in this paper.

**Etichetta:** TRUE

---

## Claim 27
**Paper:** GQA: Training Generalized Multi-Query Transformer Models (informatica, modelli T5)
**Campo:** main_contributions

**Claim:** The paper provides a recipe for uptraining existing multi-head checkpoints into MQA models.

- However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) intro- duce grouped-query attention (GQA), a gener- alization of multi-query attention which uses an intermediate (more than one, less than num- ber of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA. 1 Introduc
- We show that uptrained GQA achieves quality close to multi- head attention while being almost as fast as multi- query attention. 2 Method 2.1 Uptraining Generating a multi-query model from a multi-head model takes place in two steps: first, converting the checkpoint, and second, additional pre-training to allow the model to adapt to its new structure. Fig- ure 1 shows the process for converting a multi-head checkpoint into a multi-query checkpoint. The pro- jection matrices for key and value heads are mean pooled into single projection matrices, which we find works better than selecting a sing
- We apply MQA and GQA to decoder self-attention and cross-attention, but not encoder self-attention. Uptraining Uptrained models are initialized from public T5.1.1 checkpoints. The key and value heads are mean-pooled to the appropriate MQA or GQA structure, and then pre-trained for a further α proportion of original pre-training steps with the original pre-training setup and dataset from (Raffel et al., 2020).

**Etichetta:** TRUE

---

## Claim 28
**Paper:** Nonmonotonic energy dependence of net-proton number fluctuations (STAR, RHIC)
**Campo:** results

**Claim:** Models predicting no critical point showed trends suggesting sensitivity to physical phenomena indicative of a critical point.

- We report the first evidence of a non- monotonic variation in kurtosis times variance of the net-proton number (proxy for net-baryon number) dis- tribution as a function of √sNN with 3.1σ significance, for head-on (central) gold-on-gold (Au+Au) collisions measured using the STAR detector at RHIC. Data in non-central Au+Au collisions and models of heavy-ion collisions without a critical point show a monotonic variation as a function of √sNN. 3 One of the fundamental goals in physics is to understand the properties of matter when subjected to variations in tem- perature and pressure. Currently, 
- Moments HRG GCE HRG EV HRG CE UrQMD (r = 0.5 fm) Sσ < 0.001 < 0.001 0.0754 < 0.001 κσ2 0.00553 0.0145 0.0450 0.0221 without a critical point, as observed experimentally in periph- eral collisions. The deviation of the measured κσ2 from sev- eral baseline calculations with no critical point, and its non- monotonic dependence on √sNN, are qualitatively consistent with expectations from a QCD-based model which includes a critical point [9, 11]. Our measurements can also be compared to the baryon-number susceptibilities computed from QCD to understand various other features of the QCD phase struct
- However, the yields of produced hadrons and fluctuations of multiplicity distributions related to conserved quantities have been studied and shown to have characteristics of thermody- namic equilibrium for higher collision energies [10, 15–20]. Upon approaching a critical point, the correlation length di- verges and thus renders, to a large extent, microscopic details irrelevant. Hence observables like the moments of the con- served net-baryon number distribution, which are sensitive to the correlation length, are of interest when searching for a crit- ical point.

**Etichetta:** FALSE

---

## Claim 29
**Paper:** Flux tube in turbulent flow and quantum chromodynamics
**Campo:** future_work

**Claim:** Further exploration of the mappings identified could lead to new insights into both areas.

- Singleton, Mod. Phys. Lett. A 18, 955 (2003); hep-th/0210287. [17] W. Heisenberg, Introduction to the unified field theory of elementary particles., (Max - Planck - Institut f ̈ur Physik und Astrophysik, Interscience Publishers London, New York-Sydney, 1966). [18] V. Dzhunushaliev, Science Echoes, Vol.4, No:3, 82-112 (2008); hep-ph/0605070.
- In this approach (the dual superconductivity scenario) a condensate of the monopoles breaks spontaneously an internal U(1) gauge symmetry. According to the dual superconductor idea, the breaking of the dual symmetry gives rise naturally to the dual Meissner effect, which insures a formation of a tube, which in turn leads to the confinement the quarks into hadronic bound states. In Ref. [5] it is supposed that the SU(2) Yang-Mills theory in the low-temperature phase can be considered as a nematic liquid crystal.
- In order to understand the reason of confinement problem in quantum chromodynamics there are many attempts to compare quantum chromodynamics witn other branches of physics. In Ref. [3] it was suggested that the confinement of quarks into hadrons may happen due to a condensation of special gluonic configurations called Abelian monopoles. In this approach (the dual superconductivity scenario) a condensate of the monopoles breaks spontaneously an internal U(1) gauge symmetry.

**Etichetta:** ?

---

## Claim 30
**Paper:** Flux tube in turbulent flow and quantum chromodynamics
**Campo:** experiments

**Claim:** No explicit experiments conducted; rather, theoretical analysis and derivation of equations governing flux tube dynamics in both turbulent flows and QCD scenarios.

- In our opinion it is an analog of a flux tube filled with a gauge fields and stretched between quark and antiquark from quantum chromodynamics. Such kind of solution is characterized by having a strongly pronounced core (tube) through which flows the turbulent fluid. Such flux of the turbulent fluid may be inserted either into a flow of a laminar fluid v∞̸ = 0 (that is more probably) or in a rest fluid v∞= 0 (that is less probably).
- arXiv:0907.3624v2 [physics.flu-dyn] 7 Aug 2009 Flux tube in turbulent flow and quantum chromodynamics Vladimir Dzhunushaliev Institut f ̈ur Physik, Universit ̈at Oldenburg, Postfach 2503, D-26111 Oldenburg, Germany; Department of Physics and Microelectronic Engineering, Kyrgyz-Russian Slavic University, Bishkek, Kievskaya Str. 44, 720021, Kyrgyz Republic and Institute of Physics of National Academy of Science Kyrgyz Republic, 265 a, Chui Street, Bishkek, 720071, Kyrgyz Republic∗ (Dated: October 24, 2018) Using some assumptions about the correlation function of velocity in a turbulent flow, a c
- They are a mixture of a superconductor vortex and a superfluid vortex and are called a semi-superfluid vortices. The color magnetic flux tubes also exist (as BPS vortices) in supersymmetric QCD. In Ref’s [11]-[14] one can find reviews on this subject.

**Etichetta:** ?

---

## Claim 31
**Paper:** Nonmonotonic energy dependence of net-proton number fluctuations (STAR, RHIC)
**Campo:** results

**Claim:** Models predicting no critical point showed similar but opposite trends.

- We have observed a non-monotonic behavior as a function of √sNN, in net-proton κσ2 in central Au+Au collisions with a signifi- cance of 3.1σ relative to Skellam expectation. Other baselines without a critical point result in similar significance. In con- trast, monotonic behavior with √sNN is predicted for the sta- tistical hadron gas model, and for a nuclear transport model 6 5 10 20 50 100 200 0.0 0.2 0.4 0.6 0.8 1.0 (1) σ S Au+Au Collisions Au+Au Collisions Net-proton < 2.0 (GeV/c) T |y| < 0.5, 0.4 < p UrQMD 0%-5% HRG GCE HRG CE HRG EV (r=0.5fm) STAR 0%-5% 70%-80% Stat. uncertainty Syst. un
- Moments HRG GCE HRG EV HRG CE UrQMD (r = 0.5 fm) Sσ < 0.001 < 0.001 0.0754 < 0.001 κσ2 0.00553 0.0145 0.0450 0.0221 without a critical point, as observed experimentally in periph- eral collisions. The deviation of the measured κσ2 from sev- eral baseline calculations with no critical point, and its non- monotonic dependence on √sNN, are qualitatively consistent with expectations from a QCD-based model which includes a critical point [9, 11]. Our measurements can also be compared to the baryon-number susceptibilities computed from QCD to understand various other features of the QCD phase struct
- Corresponding κσ2 (Sσ) results for 0-5% Au+Au collisions from a transport-based UrQMD model [31] calculation, which incorporates conservation laws and most of the relevant physics apart from a phase transi- tion or a critical point, and which is calculated within the ex- perimental acceptance, show a monotonic decrease (increase) with decreasing collision energy (see Supplemental Mate- rial [34] for a quantitative comparison). An exercise with the UrQMD and HRG model with canonical ensemble as the non- critical baseline yielded a similar significance as reported in Fig. 3. Similar conclusions 

**Etichetta:** FALSE

---

## Claim 32
**Paper:** The anti-k_t jet clustering algorithm (fisica delle particelle)
**Campo:** experiments

**Claim:** The simulations compared the method's behavior to the kt method.

- Only the two hardest jets have been kept with a further requirement |y| ≤2. The area distribution obtained from anti-kt (scaled by 0.1) is compared to the other algorithms. (a) passive area at parton level; (b) active area at hadron level including the underlying event and pileup corresponding to high luminosity LHC running. 0 0.5 1 1.5 2 5 10 100 1000 〈AJA,R〉/πR2 pt (GeV) Herwig 6.5 LHC, gg R=1 SISCone(f=0.75) Cam/Aachen ktanti-kt 0.9 1 1.1 5 10 100 1000 Figure 3: Average jet area in dijet events at the LHC.
- In usual IRC safe jet algorithms (JA), the passive area aJA,R(∆12) is πR2 when ∆12 = 0, but changes when ∆12 is increased. In contrast, since the boundaries of anti-kt jets are unaffected by soft radiation, 4 a(1PJ) A(1PJ) σ(1PJ) Σ(1PJ) d D s S kt 1 0.81 0 0.28 0.56 0.52 0.45 0.41 Cam/Aachen 1 0.81 0 0.26 0.08 0.08 0.24 0.19 SISCone 1 1/4 0 0 -0.06 0.12 0.09 0.07 anti-kt 1 1 0 0 0 0 0 0 Table 1: A summary of main area results for the anti-kt algorithm compared to those for other IRC safe algorithms in [4]: the passive (a) and active (A) areas for 1-particle jets (1PJ), the magnitude of the pas
- A summary of these and other properties of the jet area is given in table 1, together with a comparison to other IRC safe algorithms as studied in [4]. Fig. 2 illustrates the distribution of areas in dijet events at the LHC (generated with Pythia 6.4, with a transverse momentum cut of 1 TeV on the 2 →2 scattering, considering those among the two hardest jets in each event that have |y| < 2), compared with those for other algorithms, and one sees a near δ-function at an area of πR2, to be compared to the broader distributions of other algorithms. Fig. 3 shows the mean jet area, together with a 

**Etichetta:** TRUE

---

## Claim 33
**Paper:** Nonmonotonic energy dependence of net-proton number fluctuations (STAR, RHIC)
**Campo:** results

**Claim:** Peripheral 70-80% collisions have small cumulants.

- E. Deviation in κσ2 values at various √sNN of 0-5% central collision data from models and 70-80% peripheral collisions: Figure 9 shows the deviation of κσ2 values for central 0- 5% Au+Au collisions from the corresponding values from UrQMD and HRG models. Also shown is the deviation from 70-80% peripheral Au+Au collisions. The uncertainties used TABLE II. Values of the parameters of fourth (fifth) order polynomial that describes the collision energy dependence of κσ2 (Sσ) at various √sNN along with their uncertainties.
- A different volume fluctuation correc- tion method [37] has been applied to the 0-5% central Au+Au collision data and the results were found to be consistent with 0 10 20 30 40 (1) C1 0% - 5% 70% - 80% 5 10 20 50 100 200 0 10 20 30 40 (3) C3 0 10 20 30 40 (2) C2 0 20 40 60 80 100 5 10 20 50 100 200 (4) C4 Stat. uncertainty Syst. uncertainty Net-proton Cumulants (GeV) NN s Collision Energy Au+Au Collisions at RHIC Au+Au Collisions at RHIC Net-proton < 2.0 (GeV/c) T |y| < 0.5, 0.4 < p FIG. 2. Cumulants (Cn) of the net-proton distributions for central (0-5%) and peripheral (70-80%) Au+Au collisio
- An alternate approach called the moment expansion method [8] was used for efficiency correction and found to be consistent with the unfolding method. C. Cumulants for 70-80% Au+Au collisions net-proton distribution Figure 7 shows a magnified version of the peripehral (70- 80%) Au+Au collisions data presented in Fig.2 of the paper. D. Rapidity dependence of C4/C2 for 0-5% central Au+Au collisions The cumulant ratio C4/C2 of net-proton multiplicity dis- tributions for 0-5% central Au+Au collisions at √sNN = 7.7, 11.5, 14.5, 19.6, 27, 39, 54.4, 62.4, and 200 GeV is shown in Fig. 8.

**Etichetta:** ?

---

## Claim 34
**Paper:** The anti-k_t jet clustering algorithm (fisica delle particelle)
**Campo:** main_contributions

**Claim:** The paper provides comprehensive analysis supporting the advantages of the anti-kt algorithm over traditional approaches.

- None of these properties hold for existing sequential recombination algorithms, nor for cone algorithms with split–merge steps, such as SISCone. They are however the identifying characteristics of the collinear unsafe plain “iterative cone” algorithm, for which the anti-kt algorithm provides a natural, fast, infrared and collinear safe replacement. 1 Introduction and definition Jet clustering algorithms are among the main tools for analysing data from hadronic collisions. Their widespread use at the Tevatron and the prospect of unprecedented final-state complexity at the upcoming Large Hadron 
- Their widespread use at the Tevatron and the prospect of unprecedented final-state complexity at the upcoming Large Hadron Collider (LHC) have stimulated considerable debate concerning the merits of different kinds of jet algorithm. Part of the discussion has centred on the relative advantages of sequential recombination (kt [1] and Cambridge/Aachen [2]) and cone (e.g. [3]) jet algorithms, with an issue of particular interest being that of the regularity of the boundaries of the resulting jets.
- It has distance measures dij = max(k2 ti, k2 tj)R2/∆2 ij and diB = k2 ti, and recombines the pair of objects with the largest dij unless a diB is larger in which case i is called a jet. We observe that these distance measures are just the reciprocals of those for anti-kt; together with the recombination of the most distant pair first, this causes the algorithm to produce identical jets to anti-kt. Acknowledgements One of us (GPS) thanks Bruce Knuteson for a stimulating discussion on the interest of parametrised clustering jet algorithms.

**Etichetta:** TRUE

---

## Claim 35
**Paper:** Nonmonotonic energy dependence of net-proton number fluctuations (STAR, RHIC)
**Campo:** limitations

**Claim:** Higher collision energies beyond √N₂N = 62.4 GeV could not be included due to lack of agreement with predictions for a critical point at higher baryon chemical potentials.

- The lines and shaded areas represent statistical and systematic uncertainties. larger χ2 values. It may also be noted that the possible critical point is predicted to exist at baryon chemical potential values much larger than those corresponding to √sNN = 200 GeV. The values of the derivatives of the polynomial functions for κσ2 and Sσ for 0-5% central Au+Au collisions at √sNN = 7.7, 11.5, 14.5, 19.6, 27, 39, 54.4 and 62.4 GeV are given in Table III.
- The C3/C2 = Sσ = (χB 3 /T)/(χB 2 /T 2) and C4/C2 = κσ2 = (χB 4 )/(χB 2 /T 2). Close to the critical point, QCD-based calculations predict the net- baryon number distributions to be non-Gaussian and suscep- tibilities to diverge, causing moments, especially higher-order quantities like κσ2, to have non-monotonic variation as a func- tion of √sNN [40, 41]. Figure 3 shows the central 0-5% Au+Au collision data for Sσ and κσ2 in the collision energy range of 7.7 – 62.4 GeV, fitted to a polynomial function of order five and four, respec- tively.
- Within 1.0 σ significance the Sσ allows for a non-monotonic energy depen- dence. This is consistent with a QCD based model expectation that the higher the order of the moments the more sensitive it is to physics processes such as a critical point [11]. Figure 4 shows the variation of Sσ (or C3/C2) and κσ2 (or C4/C2) as a function of √sNN for central and peripheral Au+Au collisions.

**Etichetta:** PARTIAL

---

## Claim 36
**Paper:** SU(4) Heisenberg model on the honeycomb lattice with exchange-frustrated perturbations
**Campo:** limitations

**Claim:** The authors state that the minimal model is too simplistic for Ba3CuSb2O9.

- Nevertheless, as stressed by Ref. [31], the microscopic model in Eq. (5) is too simplistic to de- scribe Ba3CuSb2O9 and a QSOL is likely not its ground state. For this material, a more realistic Hamiltonian on a decorated honeycomb lattice should be taken into account.
- The physical interpretation of τ y is that it distinguishes be- tween states with different orbital chiralities [25] and is, therefore, related to orbital-magnetic orders [24]. B. Kugel-Khomskii Model We now introduce the minimal model proposed for the Mott insulating phase of Ba3CuSb2O9 and TBG in Refs. [24, 31]. Despite the different nature of the orbitals in these systems, their in-plane symmetries enable one to assign the same Hubbard model in both cases.
- This result raises hopes that a QSOL state may be observed in honeycomb lattice materials with active spin and orbital degrees of freedom. Ba3CuSb2O9 contains Cu2+ ions with a 3d9 configura- tion. In a first approximation, one may assume that this hole has a fourfold degeneracy: a twofold spin degener- acy and a twofold orbital degeneracy of the eg orbitals.

**Etichetta:** TRUE

---

## Claim 37
**Paper:** Flux tube in turbulent flow and quantum chromodynamics
**Campo:** results

**Claim:** In the turbulent-flow model, a cylindrically symmetric tube-like solution is obtained.

- If the viscosity μ = 0 then we have the ideal fluid. We want to find cylindrically symmetric solution of the Navier - Stokes equations but not in a pipe. For the ideal (ν = 0) and laminar fluids the equation is vρ dvz dρ = ν d2vz dρ2 (7) but vρ = 0 for the flow along axes z.
- arXiv:0907.3624v2 [physics.flu-dyn] 7 Aug 2009 Flux tube in turbulent flow and quantum chromodynamics Vladimir Dzhunushaliev Institut f ̈ur Physik, Universit ̈at Oldenburg, Postfach 2503, D-26111 Oldenburg, Germany; Department of Physics and Microelectronic Engineering, Kyrgyz-Russian Slavic University, Bishkek, Kievskaya Str. 44, 720021, Kyrgyz Republic and Institute of Physics of National Academy of Science Kyrgyz Republic, 265 a, Chui Street, Bishkek, 720071, Kyrgyz Republic∗ (Dated: October 24, 2018) Using some assumptions about the correlation function of velocity in a turbulent flow, a c
- arXiv:0907.3624v2 [physics.flu-dyn] 7 Aug 2009 Flux tube in turbulent flow and quantum chromodynamics Vladimir Dzhunushaliev Institut f ̈ur Physik, Universit ̈at Oldenburg, Postfach 2503, D-26111 Oldenburg, Germany; Department of Physics and Microelectronic Engineering, Kyrgyz-Russian Slavic University, Bishkek, Kievskaya Str. 44, 720021, Kyrgyz Republic and Institute of Physics of National Academy of Science Kyrgyz Republic, 265 a, Chui Street, Bishkek, 720071, Kyrgyz Republic∗ (Dated: October 24, 2018) Using some assumptions about the correlation function of velocity in a turbulent flow, a c

**Etichetta:** ?

---

## Claim 38
**Paper:** SU(4) Heisenberg model on the honeycomb lattice with exchange-frustrated perturbations
**Campo:** results

**Claim:** Phase boundaries extended understanding via advanced simulations.

- Based on exact diagonalization on small clusters, Ref. [31] argued for a QSOL phase over an extended region in the pa- rameter space around the SU(4)-symmetric point. In the following, we use VMC methods to investigate the sta- bility of the QSOL in our model.
- [19, 20] would lead to exchange frustration simi- lar to the ones discussed in this paper. The phase dia- gram of a realistic model for this compound would then present extended regions of stability for the QSOL and the NCD phases, with possible connection with α-ZrCl3 magnetism. Acknowledgments We thank Rafael Fernandes and Johannes Knolle for interesting discussions and critical reading of our manuscript.
- This model contains an SU(4)- symmetric point at which a QSOL phase may be realized [36]. Using a combination of analytical and numerical techniques, we found that this QSOL covers an extended parameter regime in the phase diagram where we include the effects of Hund’s coupling and bond-dependent frus- trated exchange interactions. This result raises hopes that a QSOL state may be observed in honeycomb lattice materials with active spin and orbital degrees of freedom.

**Etichetta:** PARTIAL

---

## Claim 39
**Paper:** SU(4) Heisenberg model on the honeycomb lattice with exchange-frustrated perturbations
**Campo:** methodology

**Claim:** Combination of mean-field theory (MFT), linear flavor-wave theory (LFWT), variational Monte Carlo (VMC), and exact diagonalization (ED).

- We present a detailed analysis of a KK model derived independently in Refs. [24] and [31] us- ing mean-field theory (MFT), linear flavor-wave theory (LFWT) (complemented by a variational study consider- ing the Huse-Elser wavefunction [39, 40]) and VMC. Our main results are summarized in Fig. 1.
- Here we provide a unified discussion of these systems by investigating an extended model that includes the effects of Hund’s coupling and anisotropic, orbital-dependent exchange interactions. Using a combination of mean-field the- ory, linear flavor-wave theory, and variational Monte Carlo, we show that this model harbors a quantum spin-orbital liquid over a wide parameter regime around the SU(4)-symmetric point. For large Hund’s coupling, a ferromagnetic antiferro-orbital ordered state appears, while a valence-bond crystal combined with a vortex orbital state is stabilized by dominant orbital
- [36] using several numeri- cal and analytical techniques. The combination of ex- act diagonalization (ED) and Variational Monte Carlo (VMC) provided good evidence in favor of a π-flux QSOL with fermionic excitations similar to the ones obtained in large-N theories. The experimental motivation of [36] was the eg system Ba3CuSb2O9, in which Cu2+ ions were proposed to form layered honeycomb lattices [37].

**Etichetta:** TRUE

---

## Claim 40
**Paper:** Simulating twistronics without a twist (atomi freddi, doppio strato sintetico)
**Campo:** results

**Claim:** Changing the periodicity of the Raman coupling enables fine-tuning of flatness.

- More generally, for lx(y) = 4νx(y), where νx(y) are posi- tive integers, a similar argument shows that the system can be effectively described by super-Lieb lattices with a supercell of 2(νx + νy) -1 sites. Changing the periodicity of the Raman coupling in general leads to band structures without the above combination of Dirac spectra and flat bands [49]. An additional control parameter in our system is the artifi- cial magnetic flux γ.
- Finally, in the context of twistronics, it is amusing to note that the two disentangled lattices at γ = π are rotated with respect to each other by π/4. By tuning γ towards 0, their coupling becomes stronger. This leads to a change in the band structure with the formation of Dirac cones that are attracted to each other at the center of the Brillouin zone towards the quasi-flat band.
- Moreover, it is possible to derive the ratio, F, between the bandwidth of the quasi-flat bands within pertur- -82 -80 -78 E/t Γ (b) Γ M X M X’ -82 -80 -78 E/t Γ (a) Γ M X M X’ -82 -80 -78 E/t 1×10 5 1×10 8 (c) DOS(a.u.) FIG. S4. Effect of a phase shift in the spatially-dependent periodic Raman coupling. Panels (a-c) corresponds to Ω0α/h = 20t, with α = 0.2 and γ = 0.8.

**Etichetta:** FALSE

---

