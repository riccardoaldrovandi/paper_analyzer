"""
Uncertainty of the per-provider summary: with 3-30 papers per comparison,
every difference between two models needs an error bar.

- The unit of resampling is the paper, as in PHYSICS-Verified: the 95%
  confidence interval of a mean comes from a percentile bootstrap that draws
  papers with replacement.
- Several runs of the same provider on the same paper (export-dataset run
  again with a different LOCAL_SEED) are averaged per paper first, so a paper
  never weighs more because it was run more often. Their spread is reported
  as run-to-run noise: the mean over papers of the standard deviation across
  runs.
- Two providers are compared on the papers they have in common with a paired
  bootstrap of the per-paper differences: if the interval of the difference
  contains 0, the comparison does not separate them.
"""
import os
import random
import statistics
from collections import defaultdict
from itertools import combinations

N_BOOTSTRAP = 10000
SEED = 0
# Below this, the bootstrap has too few distinct resamples (10 with 3 papers)
# for its interval to mean much.
MIN_PAPERS_FOR_CI = 10
METRICS = (("L2", "l2"), ("recall", "recall"))


def _per_paper(results: list) -> dict:
    """{provider: {paper: {metric: [value per run]}}}, None values dropped."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        paper = os.path.basename(r["source_file"] or "")
        for _, key in METRICS:
            if r.get(key) is not None:
                out[r["provider"]][paper][key].append(r[key])
    return out


def bootstrap_ci(values: list, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple:
    """(mean, low, high) with a 95% percentile bootstrap of the mean."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return sum(values) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def _fmt(values: list, n_boot: int) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f} (1 paper)"
    mean, low, high = bootstrap_ci(values, n_boot)
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def print_statistics(results: list, n_boot: int = N_BOOTSTRAP) -> None:
    """results: dicts with provider, source_file, l2 and recall (either may be None)."""
    data = _per_paper(results)
    if not data:
        return
    print(f"\nPer-paper statistics (mean over papers, 95% bootstrap CI over papers, {n_boot} resamples):")
    print(f"{'provider':<40} {'papers':>6} {'runs':>5} {'L2 mean [95% CI]':>26} {'recall mean [95% CI]':>26} {'run noise L2/recall':>20}")
    paper_means = {}
    for provider in sorted(data, key=str):
        papers = data[provider]
        means = {key: {p: statistics.mean(v[key]) for p, v in papers.items() if v[key]} for _, key in METRICS}
        paper_means[provider] = means
        runs = [max(len(v[key]) for _, key in METRICS) for v in papers.values()]
        noise = []
        for _, key in METRICS:
            sds = [statistics.stdev(v[key]) for v in papers.values() if len(v[key]) > 1]
            noise.append(f"{statistics.mean(sds):.3f}" if sds else "-")
        runs_txt = f"{min(runs)}" if min(runs) == max(runs) else f"{min(runs)}-{max(runs)}"
        print(f"{str(provider):<40} {len(papers):>6} {runs_txt:>5} {_fmt(list(means['l2'].values()), n_boot):>26} "
              f"{_fmt(list(means['recall'].values()), n_boot):>26} {'/'.join(noise):>20}")

    pairs = []
    for a, b in combinations(sorted(paper_means, key=str), 2):
        for label, key in METRICS:
            common = sorted(set(paper_means[a][key]) & set(paper_means[b][key]))
            if len(common) >= 2:
                diffs = [paper_means[a][key][p] - paper_means[b][key][p] for p in common]
                pairs.append((a, b, label, len(common), bootstrap_ci(diffs, n_boot)))
    if pairs:
        print("\nPaired differences (A - B on the papers both have; the CI excludes 0 only if the gap is not noise):")
        for a, b, label, n, (mean, low, high) in pairs:
            verdict = "A > B" if low > 0 else "B > A" if high < 0 else "not separable"
            if n < MIN_PAPERS_FOR_CI:
                verdict += f" (only {n} papers: indicative)"
            print(f"  {label:<7} {str(a)} - {str(b)} ({n} papers): {mean:+.3f} [{low:+.3f}, {high:+.3f}] -> {verdict}")
