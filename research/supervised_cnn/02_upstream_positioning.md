# Positioning against `impl/ml-supervised-1dcnn` and `impl/ml-transformer-faultformer`

Two branches on this repository's remote already implement supervised
bearing-fault models. **They were written first and they are substantial.**
Claiming novelty for this branch without saying so would be exactly the kind of
quiet inaccuracy this repository is organised against.

Audited commits: `impl/ml-supervised-1dcnn` at `33b3c19`,
`impl/ml-transformer-faultformer` at `b889645`. Both are cut from `main`
(the plans-only branch) and neither has been merged into `platform/core`, so
neither has access to `pdmcore` — which matters, because `pdmcore.eval.splits`
is the thing that would have caught most of what follows.

Everything below was produced by `prototype/supervised_cnn/audit_upstream.py`,
which **imports and runs their code** against the real CWRU files. Results in
`prototype/supervised_cnn/results/upstream_audit.json`. Where a finding is
static rather than executed, it says so and quotes the line.

## What they built

### `impl/ml-supervised-1dcnn` (~2 000 lines, PyTorch)

| file | what it is |
|---|---|
| `model.py` | `BearingFaultCNN1D`: four Conv1d+BN+ReLU+MaxPool+Dropout blocks with a wide first kernel (64 taps, stride 2), global average pool, FC 256→128→12. **1 350 410 parameters**, confirmed by import. |
| `dataset.py` | `CWRUBearingDataset`, `MFPTBearingDataset`, `DatasetFactory.combined_cwru_mfpt`, `segment_and_fft`, and `split_by_bearing` — with a docstring citing the data-leakage literature and calling the bearing-wise split "the ONLY valid evaluation split". |
| `train.py` | Adam + cosine schedule, class weights, per-class F1, confusion-matrix plots, MLflow logging, early stopping on macro F1. |
| `export.py`, `serve.py` | ONNX export and a FastAPI service. |

It is a genuine, coherent, deployment-shaped implementation. It also **cannot be
run in this environment as distributed** (torch, mlflow, onnx, fastapi, pandas;
only torch was installed here, into a separate venv, to run the audit).

### `impl/ml-transformer-faultformer` (~3 200 lines, PyTorch)

`model.py` (FaultFormer + pretraining/classification heads), `pretrain.py`,
`finetune.py`, and `cross_domain.py` — 529 lines aimed at precisely this
branch's question: CWRU→MFPT and CWRU→MIMII transfer, FaultFormer fine-tune
versus train-from-scratch, with the expected finding stated in the module
docstring.

## What they did not evaluate

Six findings. The first four were produced by executing their code; the last two
are static with the lines quoted.

### 1. Against CWRU as published, every record is labelled "Normal"

`CWRUBearingDataset._scan_directory` derives the label from **directory names**
(`'outer'`/`'inner'`/`'ball'` in the path, plus a `'0.007'`-style diameter
folder). CWRU is distributed as flat numbered `.mat` files. Run on the 40 real
files:

```
n_windows                 5 886
n_distinct_labels             1
label_counts         {'Normal': 5886}
majority-class accuracy    1.000
```

The 10-class problem silently becomes a one-class problem, and any model reports
100 %. The branch's own README documents an expected directory tree, so this is a
documentation-vs-data mismatch rather than a logic error — but it fails silently,
with a `logger.debug` for skipped files and no count of how many labels were
actually distinct. Fixing it is four lines (assert `n_distinct_labels > 1`).

Their README's example tree also mis-assigns files: it lists
`outer/0.007/ 105.mat 169.mat 197.mat 209.mat`, where 105 and 209 are inner-race
and 169 is inner-race 0.014". A user who filed the data as documented would train
on wrong labels.

### 2. `99.mat` returns `98`'s data

`_load_cwru_mat` returns the first variable whose name ends `_DE_time`. `99.mat`
as published holds five variables:

```
['X098_DE_time', 'X098_FE_time', 'X099_DE_time', 'X099_FE_time', 'ans']
```

Executed: the returned array has **483 903 samples, which is `X098`**, not
`X099`'s 485 063. So the 2 hp healthy baseline is the 1 hp one, duplicated, and
the 2 hp record is never read. `prototype/supervised_cnn/cwru.py` addresses the
channel by the explicit variable name its record table declares and raises if it
is absent.

### 3. The "bearing-wise split" is a file-wise split, and its own audit cannot fail

`bearing_id = f"cwru_{bearing_counter:04d}_{mat_file.stem}"` — one id per file.
Executed, with their loader fed this branch's *correct* record table so the
finding is about their split and not their directory scanner:

```
n_bearing_ids (theirs)                40
n_physical_units                      10
audit vs their own key          ok=True,  violations=[]
audit vs physical unit          ok=False, violations=['5 group(s) appear in both folds:
                                ball_0.021, healthy, inner_0.014, outer_0.007, outer_0.021']
```

**Half the physical bearings are on both sides of the partition.** Audited
against the key that produced it, the split is clean. That is the
check-that-cannot-fail pattern in its purest form, and it is doing real damage
here because the thing the split was introduced to prevent is exactly what
survives.

This is not a criticism that could have been made from reading the code alone, and
it is not a mistake unique to these branches: calling a per-file key
"bearing-wise" is the convention in a large part of the published literature.

### 4. 22 of 40 seeds drop a class from the test fold

`split_by_bearing` shuffles the ids and takes `test_fraction` of them. Nothing
keeps every class on both sides. Executed over seeds 0–39 on the 4-class label
space:

```
seeds with a class absent from test     22 / 40
seeds with a class absent from train     0 / 40
```

Their default `seed=42` happens to be one of the good ones. A class absent from
the test fold is simply not evaluated, and `f1_score(..., average="macro")`
without an explicit `labels=` averages only over classes present — so the
reported macro F1 for such a seed is a three-class score presented as a
four-class one.

`pdmcore.eval.splits.stratified_group_split` on the same groups and labels
succeeds and reports its test-fold composition:

```
sizes               [4470, 1416]
test_class_counts   {'Normal': 473, 'BPFO': 352, 'BPFI': 355, 'BSF': 236}
```

### 5. Model selection reads the test fold (static)

`train.py` has no validation loader. Quoted:

```
254: val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
284: if val_f1 > best_val_f1:
 62: parser.add_argument("--patience", default=15, ...)  # "Early stopping patience (val F1 epochs)"
```

So the checkpoint kept, and the epoch at which training stops, are chosen by
performance on the fold being reported. Over a 100-epoch schedule that is a
best-of-100, and the reported figure is optimistically biased by the
epoch-to-epoch variance of the test score.

In fairness: a separate validation fold is not actually available. Every fold of
CWRU that is not the test fold shares a physical bearing with it, which is this
branch's own headline structural finding. The honest options are a fixed epoch
budget (what this branch uses) or nested cross-validation over units (which CWRU
cannot support) — not selection on the test fold.

### 6. `cross_domain.py` produces a cross-domain result from synthetic sinusoids

The single most serious finding, and it is in the file aimed at this branch's
question. Quoted from `cross_domain.py`:

```
 99: print(f"  MFPT dir not found: {data_dir} — using synthetic data")
100: return _make_synthetic_labeled_dataset(
153: print("  No MFPT data found — using synthetic")
154: return _make_synthetic_labeled_dataset(
190: print(f"  MIMII dir not found: {data_dir} — using synthetic data")
```

Both target-domain loaders fall back to `_make_synthetic_labeled_dataset` when
their directory is absent — which it is in this environment and in any
environment where the data has not been fetched — print one line, and then the
script **runs the full CWRU→MFPT and CWRU→MIMII transfer experiment and logs the
F1 to MLflow**, under a docstring that states the expected finding in advance.

The generator, transcribed (not imported: `finetune.py` imports `mlflow`, which
is not installed here) and run:

```
signal = 0.5 * N(0,1) + sin(2*pi*(50 + 30*label)*t)
accuracy of an argmax over four FFT bins:   1.000
```

A four-bin argmax gets it perfectly. The "cross-domain transfer" comparison is
between two models on a problem solvable by one FFT bin per class. Worse, with
`n_samples=500`, `num_classes=4` and 20 bearing ids assigned by `i % 20` while
`label = i % 4`, each bearing id carries exactly one class — so the bearing-wise
split applied to the fallback data is also class-degenerate.

This is the artefact `hardware/rpi5_hailo/gateway/hailo_backend.py` on
`hw/rpi5-hailo-gateway` refuses to produce, in the same words: "a stub returning
plausible probabilities would make this kit look finished and would be the single
most misleading artefact in the repository". The fix is one line — raise instead
of returning synthetic data — and it is the one-line change in either upstream
branch with the largest effect on what a reader would believe.

### Also: eight of twelve output classes are unreachable

`model.FAULT_CLASSES` has 12 entries. The union of
`dataset.CWRU_LABEL_MAP.values()` and `dataset.MFPT_LABEL_MAP.values()` is
`[0, 1, 2, 3]`. The other eight — FTF, Unbalance, Misalignment, Lubrication,
Looseness, Cavitation, Gearbox, Electrical — can never receive a training
example from either dataset. The softmax therefore spends a third of its output
width on classes that are structurally untrainable, and `per_class_f1` is logged
to MLflow for all twelve with `labels=list(range(NUM_CLASSES))`, producing eight
metrics that are 0.0 by construction and will stay 0.0 for ever.

This is also the conceptually interesting one: the class list is the *platform's*
fault taxonomy (it matches `pdmcore.sim.machine.FAULT_MODES` closely), and the
mismatch between it and what public data can teach is a real, structural
statement about why supervised diagnosis is not deployable — see
`docs/09_supervised_cnn_crossdataset.md`.

## What this branch adds

1. **It runs their model.** `prototype/supervised_cnn/run_upstream.py` imports
   `BearingFaultCNN1D` and `segment_and_fft` unmodified and evaluates them under
   the split ladder, including under *their* `split_by_bearing`, so their number
   and the honest numbers are in one table.
2. **It evaluates on measured data at all**, with `pdmcore.eval.splits`' audit
   run and reported — which closes part of the gate in
   `docs/07_production_readiness.md`.
3. **It establishes the structural limit** that makes the audit findings matter:
   neither CWRU nor MFPT has more than one physical unit per class in any label
   space, so "bearing-wise" is not available to anybody on these datasets, and
   the honest split is the one that raises.
4. **It compares against good features.** Neither upstream branch has a
   non-deep baseline, so neither can answer whether the 1.35 M-parameter model
   earns its parameters.

## What this branch does not do to them

- It does not fix them. The findings are reported, with reproduction commands;
  no commit is made on either branch.
- It does not run the FaultFormer. That would need `mlflow`, `einops`, `timm`
  and `transformers`, and a transformer pre-training run on 2 CPU cores is not a
  thing this host can do. The `cross_domain.py` finding is therefore static —
  the control flow is read and quoted, and the fallback generator is
  transcribed and run, which is labelled as such everywhere it appears.
- It does not claim their accuracy numbers are wrong. **No accuracy number from
  either branch was found committed anywhere** — no results JSON, no MLflow
  store, no figure. So there is nothing to contradict; what is reported here is
  what their code would measure if it were run, not a correction to a published
  figure.
