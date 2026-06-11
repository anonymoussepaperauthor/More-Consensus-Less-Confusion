"""
causal_tools.py w/ num-to-sym preprocessing.

This module is causal_tools.py with converting strictly
binary numeric X columns (0/1) into symbolic columns before tree building.
"""

from math import *
from collections import Counter, defaultdict
import sys
from tools.ezr import *

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
# Number of equal-probability bins used by `disc()` for Num columns.
# Single source of truth — `_bin_legend_for_col` reads this so the printed
# legend is always in sync with the actual discretization.
DISC_BINS = 10

# Effect-size threshold for `confounder()`. Z is declared a confounder of
# the X→Y link when conditioning on Z explains away (1 - CONFOUND_EPS) of
# the unconditional dependence, i.e. when
#     cond_dep(X, Y | Z) / dep(X, Y)  <  CONFOUND_EPS.
# Smaller = stricter (must explain away more of the link to count). 0.1
# means "Z accounts for at least 90% of the X-Y dependence".
CONFOUND_EPS = 0.1

# Maximum p-value at which `correlation()` calls a column "relevant".
# A permutation test (rather than a raw effect-size cut like CONFOUND_EPS)
# is used here because `dependency()` has a column-specific finite-sample
# bias — different cardinalities have different no-signal baselines, and
# the permutation distribution auto-calibrates per column.
RELEVANCE_ALPHA = 0.1

def disc(col, v, eps=1e-32):
  """Discretize a single value.
  Missing ("?") inputs return None uniformly for both Sym and Num — this
  is what lets downstream code rely on `is None` as the single missing
  marker, and makes per-row caches in `disc_cache` / `causalTreeLeaf`
  consistent across column types.
  Numeric values are placed in one of DISC_BINS bins using a logistic-CDF
  approximation of the normal CDF (1.702 ≈ probit slope).
  Symbolic values are returned as-is (already discrete)."""
  if v == "?":      return None
  if col.it is Sym: return v

  def z(x): return (x - col.mu) / (col.sd + eps)
  def logistic(x):
    zv = -1.702 * z(x)
    if zv >  500: return 0.0   # exp(500) overflows; logistic → 0
    if zv < -500: return 1.0   # exp(-500) ≈ 0;     logistic → 1
    return 1.0 / (1.0 + exp(zv))

  edges = [i / DISC_BINS for i in range(1, DISC_BINS)]
  p = logistic(v)
  return sum(1 for e in edges if p > e)

def disc2(col, v, eps=1e-32):
  """Discretize a list of values; LENGTH-PRESERVING.

  Missing ("?") entries become None at the same index they appeared at,
  so multiple discretized columns from the same dataset stay row-aligned
  and downstream `zip`s pair the right rows. Use `_drop_missing` to
  filter aligned columns to their common non-missing subset right before
  doing math that can't tolerate None (Counter, group-by, etc.)."""
  return [None if x == "?" else disc(col, x, eps) for x in v]

def _drop_missing(*cols):
  """Filter N row-aligned columns to the common subset of indices where
  every column is non-missing (None). Returns N parallel lists.
  Single source of truth for "drop misaligned rows" — used at the top of
  every function that consumes parallel discretized columns."""
  out = [[] for _ in cols]
  for row in zip(*cols):
    if any(v == "?" for v in row):
      continue
    for i, v in enumerate(row):
      out[i].append(v)
  return tuple(out)


# ---------------------------------------------------------------------------
# Information-theoretic measures
# ---------------------------------------------------------------------------
def impurity(x):
  """
  Measures uncertainty/spread of a distribution.
  Entropy = -Σ p(x) log₂ p(x)
  Gini    = 1 - Σ p(y)²
  Impurity = 0 means all members are same.
  Missing entries (None) are excluded from the count, so they neither
  add a phantom category nor inflate the denominator.
  """
  def entropy(counts, n):
    return -sum((c / n) * log(c / n, 2) for c in counts.values())
  def gini(counts, n):
    return 1 - sum( (c/n)**2 for c in counts.values() )

  x = [v for v in x if v is not None]
  n = len(x)
  if n == 0: return 0.0
  counts = Counter(x)
  return (gini if the.Impurity == "gini" else entropy)(counts, n)

def dependency(x, y):
  """
  Measures the dependency of the two columns
  Returning = impurity(Y) - Σ p(x) * impurity(Y|X=x)
  Returns 0 when X and Y are independent; higher = more dependence.
  Inputs are expected pre-discretized and row-aligned; rows where either
  side is missing (None) are dropped together so the group-by key never
  becomes None and the row count is honest.
  """
  x, y = _drop_missing(x, y)
  n = len(x)
  if n == 0: return 0.0
  x_groups = defaultdict(list)
  for a, b in zip(x, y):
      x_groups[a].append(b)

  return impurity(y) - sum( (len(vals) / n) * impurity(vals)
      for vals in x_groups.values() )

def correlation(x, y, n_permutations=1000, seed=42, alpha=RELEVANCE_ALPHA):
    """One-sided permutation test for `dependency(x, y) > 0`.

    Returns True iff the observed dependency is unlikely under H0 of
    independence (p-value <= alpha), where the null is built by shuffling
    Y. Permutation testing is used (rather than a raw effect-size
    threshold like in `confounder`) because `dependency` has a
    column-specific finite-sample bias that the permutation distribution
    auto-calibrates.

    Implementation notes:
      * Uses a *local* random.Random(seed) via sys.modules so the global
        polluted (we are called once per X column).
      * `+1` smoothing on numerator and denominator: `observed` itself is
        one valid configuration under H0, so the lower bound on p is
        `1/(n+1)`, not 0.
      * Sequential early-stop with asymmetric bands around `alpha` —
        stops "not significant" sooner than "significant" so we lean
        toward dropping likely-noise features (the tree's DELTA gate is
        a second guard against false positives slipping through).
    """
    rng = sys.modules['random'].Random(seed)
    # Align once at the top so the shuffled null and the observed value
    # are both computed on the same row subset.
    if len(x) < 2:
        return False
    observed = dependency(x, y)
    y_shuffled = list(y)

    count = 0
    lo, hi = alpha * 0.2, alpha * 1.8
    warmup = 100

    for i in range(n_permutations):
        rng.shuffle(y_shuffled)
        if dependency(x, y_shuffled) >= observed:
            count += 1
        # First check at i == warmup, then every 50 perms.
        if i >= warmup and (i - warmup) % 50 == 0:
            p = (count + 1) / (i + 2)
            if p < lo or p > hi:
                return p <= alpha

    return (count + 1) / (n_permutations + 1) <= alpha

def cond_impurity(x, y):
  """Kernel for cond_impurity: H(X|Y) on already-discretized, row-aligned
  inputs. Drops rows where either side is missing so the Y group-by key
  never becomes None and the row count is honest."""
  x, y = _drop_missing(x, y)
  n = len(y)
  if n == 0: return 0.0
  y_groups = defaultdict(list)
  for a, bi in zip(x, y):
      y_groups[a].append(bi)
  return sum( (len(vals) / n) * impurity(vals)
        for vals in y_groups.values() )

def cond_dependency(x, y, z):
  """
  Conditional dependency D(X,Y|Z) = Σ_z p(z) * D(X;Y|Z=z)
    where D is dependency
  Used to detect confounders: if D(X;Y|Z) ≈ 0,
  then Z explains away the X↔Y link.
  All three inputs are aligned to their common non-missing subset so
  Z-strata never include None as a category and `total_weight` reflects
  the actually-used rows (matching the denominator that `dependency`
  will later use when called with the same x/y).
  """
  x, y, z = _drop_missing(x, y, z)
  total_weight = len(x)
  if total_weight == 0: return 0.0

  z_groups = defaultdict(lambda: ([], []))
  for xi, yi, zi in zip(x, y, z):
      xs, ys = z_groups[zi]
      xs.append(xi)
      ys.append(yi)

  weighted_dep = 0.0
  for xs, ys in z_groups.values():
      if len(xs) > 1:
          weighted_dep += (len(xs) / total_weight) * dependency(xs, ys)
  return weighted_dep

def confounder(x, y, z, eps=CONFOUND_EPS):
    """Test whether Z confounds the X→Y link, via an effect-size ratio.

    Idea (matches the docstring of `cond_dependency`): if D(X;Y|Z) ≈ 0
    then Z screens off X from Y. Concretely we declare Z a confounder
    when the residual conditional dependence is a small fraction of the
    unconditional dependence:

        cond_dep(X, Y | Z) / dep(X, Y)  <  eps

    Edge case: if X and Y are essentially independent already
    (`dep(X, Y) ≈ 0`) there is no link for Z to explain away, so we
    return False rather than dividing by ~0.
    """
    if len(x) < 2:
        return False
    uncond = dependency(x, y)
    if uncond < 1e-9:
        return False
    cond = cond_dependency(x, y, z)
    return (cond / uncond) <= eps

# ---------------------------------------------------------------------------
# Tree Generation
# ---------------------------------------------------------------------------
def causalTree(data, Y=None, filter_confounders=True):
  """Prepare labeled data and pass it to the causal tree generator.
  Steps: compute d2h target → remove confounded columns → build tree.

  `filter_confounders`: when True (default), drop X columns that are
  confounded by another X (via `remove_confounder`). Set to False to skip
  that step and keep all X columns available as split candidates.

  Returns: (tree, data) where `data` is the *canonical* Data whose
  `cols.all` defines the discretization the tree was built against.
  Callers MUST pass this same `data` object to `causalTreeLeaf` /
  `causalTreeShow` so test-time `disc()` lookups use the same column
  objects (and same mu/sd, same Num/Sym choice) as training time.
  Note: `data` here is the same object the caller passed in — we mutate
  it in place (treat_binary_numeric_as_sym, drop confounded/constant x
  columns, attach `data.ys`)."""
  def update_data(data):
    """Attach mu/sd to Sym columns (for disc/disc2) and compute d2h summary."""
    for col in data.cols.x:
      if col.it is Sym:
        col.sd = div(col)
        col.mu = mid(col)
    ys = [Y(r) for r in data.rows]
    col = adds(ys)
    col.rows = ys
    data.ys = col
    return data

  def remove_confounder(data, disc_cache):
    """
    Identify and exclude columns whose association with Y is explained
    away by another column (confounder). When mutual confounding is detected,
    keep the stronger causal signal (lower H(Y|X)/H(Y) ratio).

    `disc_cache` carries the already-discretized columns (built once at
    the top of `causalTree`), so this function does no `disc2` work of
    its own — it just looks up `disc_cache[col.at]` and the d2h target
    at `disc_cache['ys']`.
    """
    y_disc = disc_cache['ys']

    # Causal strength: H(Y|X) / H(Y) — lower means stronger causal signal.
    # IMPORTANT: numerator and denominator are computed on the SAME per-
    # column non-missing row subset, so columns with more "?"s don't get
    # artificially favourable ratios from a denominator computed on noisier
    # rows that the numerator never sees. Equal-row-subset is what makes
    # cross-column comparison meaningful in `stronger()` below.
    
    strength = {}
    for col in data.cols.x:
      xs_full = disc_cache[col.at]
      ys_sub, xs_sub = _drop_missing(y_disc, xs_full)
      if len(ys_sub) < 2:
        strength[col.txt] = 1e32
        continue
      i_y_col = impurity(ys_sub)   
      if i_y_col < 1e-9:
        strength[col.txt] = 1e32
        continue
      cond = cond_impurity(xs_sub, ys_sub)
      strength[col.txt] = (cond + 1e-32) / i_y_col
    
    def stronger(a, b):
        "Return name of the causally stronger column, or a if equal."
        return a if strength[a] <= strength[b] else b

    # Step 2: Detect confounders among relevant column pairs.
    relevant = {col.txt: correlation(disc_cache[col.at], y_disc)
                for col in data.cols.x}

    # iterate only on unique pairs (i < j)
    cols = [c for c in data.cols.x if relevant[c.txt]]

    remove = set()
    for i, c1 in enumerate(cols):
        if c1.txt in remove: continue
        x1 = disc_cache[c1.at]

        for c2 in cols[i+1:]:
            if c2.txt in remove: continue
            x2 = disc_cache[c2.at]

            c2_confounds_c1 = confounder(x1, y_disc, x2)
            if not c2_confounds_c1: continue

            c1_confounds_c2 = confounder(x2, y_disc, x1)
            keep = stronger(c1.txt, c2.txt) if c1_confounds_c2 else c1.txt
            drop = c2.txt if keep == c1.txt else c1.txt
            # kind = "mutual" if c1_confounds_c2 else "one-way"
            # print(f"  {kind} confounder: {c1.txt} vs {c2.txt} — keeping {keep}, dropping {drop}")
            remove.add(drop)
            if drop == c1.txt: break

    # Drop confounded columns in place. We keep them in data.cols.all so
    # row indexing by `col.at` still works; they just won't appear as
    # candidate split columns. The `disc_cache` keeps its entries for
    # dropped columns too — harmless, since downstream lookups go through
    # `data.cols.x`.
    data.cols.x = [c for c in data.cols.x if c.txt not in remove]
    return data

  def drop_constant_x_columns(data):
    """Ignore X columns with <=1 unique non-missing value (numeric or symbolic)."""
    keep = []
    for col in data.cols.x:
      vals = [r[col.at] for r in data.rows if r[col.at] != "?"]
      if len(set(vals)) > 1:
        keep.append(col)
    data.cols.x = keep
    return data
  
  def treat_binary_numeric_as_sym(data):
    """Convert numeric X columns with values in {0,1} (or float equivalents) into Sym."""
    binary_markers = {0, 1, 0.0, 1.0}
    for col in list(data.cols.x):
      if col.it is not Num:
        continue
    
      vals = [r[col.at] for r in data.rows if r[col.at] != "?"]
      if not vals:
        continue

      uniq = set(vals)
      if len(uniq) <= 2 and uniq.issubset(binary_markers):
        new = Sym(col.at, col.txt)
        for v in vals:
          add(new, v)
        data.cols.all[col.at] = new
        data.cols.x = [new if c.at == col.at else c for c in data.cols.x]
    return data

  ## Step 1: prepare canonical columns (Num→Sym for binary; mu/sd; ys).
  data = treat_binary_numeric_as_sym(data)
  if Y is None:
    Y = (lambda row: disty(data, row))
  updated = update_data(data)

  ## Step 2: build the discretization cache ONCE, before any consumer.
  ## Order matters: must come after step 1 because `disc()` reads col.it
  ## (Num vs Sym) and col.mu/col.sd, both of which step 1 finalizes.
  ## Both `remove_confounder` and `causalTreeGenerate` consume this same
  ## object, so each row gets discretized exactly once per column.
  disc_cache = {
      col.at: [disc(col, row[col.at]) for row in updated.rows]
      for col in updated.cols.x
  }
  disc_cache['ys'] = [disc(updated.ys, Y(row)) for row in updated.rows]

  ## Step 3: remove confounders + constant columns. Confounder removal
  ## reads from disc_cache; column drops are reflected in `data.cols.x`
  ## but cache entries are left in place (lookups go through cols.x).
  cleaned = remove_confounder(updated, disc_cache) if filter_confounders else updated
  cleaned = drop_constant_x_columns(cleaned)

  ## Step 4: build the tree.
  irows = list(enumerate(cleaned.rows))
  tree  = causalTreeGenerate(cleaned, Y, irows=irows, disc_cache=disc_cache)

  return tree, cleaned

def causalTreeGenerate(data, Y, irows=None, how=None, disc_cache=None):
  """Recursively build a causal decision tree.
  Split criterion: H(Y|X)/H(Y) — lower means X explains more of Y's uncertainty.
  Termination: the.leaf controls min leaf size, which bounds depth."""
  DELTA = 0.02
  irows = irows or list(enumerate(data.rows))
  rows  = [r for _, r in irows]
  Y    = Y or (lambda row: disty(data, row))
  tree  = o(rows=rows, irows=irows, how=how, kids=[],
            mu=mid(adds(Y(r) for r in rows)))

  if len(irows) > the.leaf:
    if not data.cols.x:
      return tree

    spread, cuts = min(causalCuts(col, irows, disc_cache) for col in data.cols.x)
    if spread < -DELTA:
      for cut in cuts:
        op, at, y = cut
        sub_irows = [(i, r) for i, r in irows
                      if causalTreeSelects(disc_cache[at][i], op, y)]
        if the.leaf <= len(sub_irows) < len(irows):
          tree.kids += [causalTreeGenerate(data, Y, sub_irows, cut, disc_cache)]

  return tree

def causalCuts(col, irows, disc_cache=None):
  """Score a column using H(Y|X)/H(Y) ratio.
  Lower ratio = X explains more of Y = stronger causal signal.
  Returns (score, list_of_equality_cuts)."""
  xs_at, ys = disc_cache[col.at], disc_cache['ys']
  xs_full = [xs_at[i] for i, _ in irows]
  ys_full = [ys[i]    for i, _ in irows]
  x_vals, y_disc = _drop_missing(xs_full, ys_full)
  if len(x_vals) < 2:
    return big, []

  i_y = impurity(y_disc)

  if i_y < 1e-9 or len(x_vals) < 2:
    return big, []

  n = len(y_disc)
  best_score = big
  best_val   = None

  for v in set(x_vals):
    y_eq  = [y for x, y in zip(x_vals, y_disc) if x == v]
    y_neq = [y for x, y in zip(x_vals, y_disc) if x != v]

    if len(y_eq) < the.leaf or len(y_neq) < the.leaf:
      continue

    score = (len(y_eq)/n)  * impurity(y_eq) + (len(y_neq)/n) * impurity(y_neq)
    if score < best_score:
      best_score = score
      best_val   = v

  if best_val is None:
    return big, []

  ## normalizes against baseline impurity
  ## higher = better split, so we negate below
  causality_ratio = 1 - (best_score / i_y)  

  return -causality_ratio, [("==", col.at, best_val), ("!=", col.at, best_val)]

def causalTreeSelects(x: Atom, op: str, y: Atom) -> bool:
  """Apply a split given an already-discretized feature value `x`.
  This is the shared kernel used by both build-time recursion (with `x`
  read from `disc_cache[at][i]`) and test-time descent (with `x` read from
  the per-row cache in `causalTreeLeaf`).
  `x is None` means missing ("?") → pass through both branches, matching
  the behaviour of `causalTreeSelects` on a "?" cell."""
  if x is None:  return False
  if op == "<=": return x <= y
  if op == "==": return x == y
  if op == "!=": return x != y
  if op == ">":  return x >  y

def causalTreeLeaf(data, tree, row, _cache=None):
  """Find which leaf a row belongs to in a causal tree.
  On the first call we precompute `disc()` for each column actually used
  in any split below `tree`, so the same (row, column) pair is never
  discretized twice while traversing the path. The cache is threaded
  through recursion via the `_cache` arg (intended as private)."""
  if _cache is None:
    _cache = {at: disc(data.cols.all[at], row[at])
              for at in _used_split_ats(tree)}
  for kid in tree.kids:
    op, at, y = kid.how
    if causalTreeSelects(_cache[at], op, y):
      return causalTreeLeaf(data, tree=kid, row=row, _cache=_cache)
  return tree

def causalTreeNodes(tree, lvl=0):
  "Iterate over all causal tree nodes."
  yield lvl, tree
  for kid in sorted(tree.kids, key=lambda kid: kid.mu):
    yield from causalTreeNodes(kid, lvl + 1)

def causalTreeShow(data, tree, win=None):
  "Display causal tree structure with Y means (mirrors ezr treeShow)."
  win = win or (lambda v: int(100 * v))
  n = {s: 0 for s in data.cols.names}
  for lvl, node in causalTreeNodes(tree):
    if lvl == 0:
      continue
    op, at, y = node.how
    indent = "|  " * (lvl - 1)
    rule = f"if {data.cols.names[at]} {op} {y}"
    n[data.cols.names[at]] += 1
    leaf = ";" if not node.kids else ""
    print(f"n:{len(node.rows):4}   win:{win(node.mu):5}     ", end="")
    print(f"{indent}{rule}{leaf}")
  print("\nUsed: ", *sorted([k for k in n.keys() if n[k] > 0],
                           key=lambda k: -n[k]))

def _used_split_ats(tree):
  "Collect column indexes used in split rules."
  ats = set()
  for lvl, node in causalTreeNodes(tree):
    if lvl == 0:
      continue
    _, at, _ = node.how
    ats.add(at)
  return ats

def _fmt_symbol(v):
  if isinstance(v, float):
    return f"{v:.3f}".rstrip("0").rstrip(".")
  return str(v)

def _bin_legend_for_col(col, q=DISC_BINS):
  """Human-readable mapping from discretized bins to approximate raw values.
  `q` defaults to `DISC_BINS` so the legend always matches `disc()`."""
  if col.it is Sym:
    vals = sorted(col.has.keys(), key=lambda x: str(x))
    joined = ", ".join(_fmt_symbol(v) for v in vals[:12])
    if len(vals) > 12:
      joined += ", ..."
    return [f"symbolic values: {joined}"]

  if col.sd <= 0:
    return [f"all bins collapse near value {col.mu:.3f}"]

  # `q` bins ↔ `q+1` boundary probabilities. Compute the inverse-logistic
  # value once per boundary (the high edge of bin i is the low edge of
  # bin i+1, so doing it per-bin would do every interior boundary twice).
  probs = [i / q for i in range(q + 1)]   # [0, 1/q, 2/q, ..., 1]
  def inv(p):
    if p <= 0: return float("-inf")
    if p >= 1: return float("inf")
    return -log(1 / p - 1) / 1.702
  values = [col.mu + col.sd * inv(p) for p in probs]

  def fmt(x):
    if   x == float("-inf"): return "-inf"
    elif x == float("inf"):  return "inf"
    else:                    return f"{x:.3f}"

  return [f"bin {i}: p in ({probs[i]:.1f},{probs[i+1]:.1f}] "
          f"-> value in [{fmt(values[i])}, {fmt(values[i+1])}]"
          for i in range(q)]

def print_bin_legend(data, tree):
  "Print legend for columns that were actually used by the tree."
  used_ats = _used_split_ats(tree)
  if not used_ats:
    print("\nBin legend: no splits, so no bins to report.")
    return
  print("\nBin legend (used split columns):")
  for at in sorted(used_ats):
    col = data.cols.all[at]
    print(f"  {col.txt}:")
    for line in _bin_legend_for_col(col):
      print(f"    {line}")



# ## Demos ------------------------------------------------------------
def eg__demo():
  "The usual run (n2s causal tree)."
  the.Budget  = 50
  the.Check   = 10
  the.Impurity= "gini"
  the.acq     = "near"
  the.leaf    = 3
  data = Data(csv(sys.argv[1] if len(sys.argv) > 1 else the.file))
  # print("\nFile:\t",the.file)
  # print("Rows:\t",len(data.rows))
  # print("X:\t",len(data.cols.x))
  # print("Y:\t",len(data.cols.y),*[c.txt for c in data.cols.y])
  # print(" ")
  b4 = adds(disty(data,row) for row in data.rows)
  win = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
  
  half = len(data.rows) // 2
  for _ in range(3):
    data.rows = shuffle(data.rows)
    train   = clone(data, data.rows[:half])
    holdout = clone(data, data.rows[half:])
    labels = likely(train)
    tree, canon = causalTree(clone(train, labels), filter_confounders=True)
  # causalTreeShow(canon, tree, win)
  # print([c.txt for c in canon.cols.x])
  # # print_bin_legend(canon, tree)

  # top_holdout     = sorted(
  #     [(causalTreeLeaf(canon, tree, row).mu, row)
  #       for row in holdout.rows],
  #     key=lambda x: x[0]
  # )[:the.Check]
  # print("Best train:", win(min([disty(data, row) for row in labels])), 
  #       "hold-out:",   win(min([disty(data, row) for _, row in top_holdout])))

if __name__ == "__main__":
  eg__demo()
