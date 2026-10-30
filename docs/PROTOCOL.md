# First transformer experiment

This protocol describes the implemented benchmark. It is not publicly preregistered. Local smoke results are development evidence. No GPU/pretrained performance result is currently claimed.

## Questions

1. Does a persistent state help after relevant observations leave the current input?
2. Does the rotating gated update outperform a GRU or a similarly evaluated token memory?
3. Does state-conditioned routing add value beyond static adapters?
4. Do benefits survive longer delays and held-out dictionaries/scenes?

## Language

Four words map bijectively to four labels. Each episode samples two distinct dictionaries and assigns them to two workshop names. It supplies four teaching sentences for A, distractors, four queries; then the same structure for B; then distractors and four queries about A without reteaching. Queries include workshop and word, never the answer. Query answers are classification targets and are never fed back as observations.

All 24 permutations are partitioned into six orbits under cyclic output relabeling. Three orbits (12 dictionaries) train, one orbit (4) validates, and two orbits (8) test. Every word has uniform label marginals in every split. Complete test dictionaries are unseen; individual associations are not necessarily novel. Procedural episode RNG uses independent split namespaces. Both contexts and query orders are shuffled. Balanced query-label chance is 25%.

## Vision

Two colored squares have independently sampled cardinal directions. Four visible frames provide motion evidence; a configurable hidden gap follows; two query-only frames identify which object's last observed direction is requested. A colored marker occupies the top strip and conveys identity only. Both objects are absent in scored frames. A second phase changes both directions and supplies new evidence. A third phase supplies another segment under those new directions. Random positions, speed and object query order vary across episodes.

Coordinates wrap at scene boundaries; this introduces wrap events and is part of the synthetic task. Overlap can cause partial occlusion. The queried target is **last observed direction**, not arbitrary extrapolation after a hidden unobserved reversal. Direction balance is uniform in expectation. Split seed namespaces are disjoint; this is scene-seed generalization, not held-out object categories or real-world domain transfer.

## Models and budgets

Read the README mode table. All modes get the same current observation at each step, backbone initialization/model identity, train/validation/test streams and nominal update schedule. Models with memory retain only their declared state. Tiny models train end-to-end; pretrained backbones freeze by default. Frozen does not mean no autograd through their downstream blocks.

The loop has coordinate-pair rotation, learned decay, a contextual multiplicative write and bounded `tanh` state. One internal block receives a gated low-rank operator mixture plus a projected state. GRU uses the same internal read/operator interface and replaces only the recurrent write. No-rotation keeps the other loop parameters. Hard routing uses a straight-through estimator while the forward selects one operator. Static routing removes history from the router. Token memory uses learned cross-attention read/write at the same block. Window control stores the previous W encoded observations and applies a small temporal transformer.

Report trainable/total parameters, persistent state scalars, measured latency and peak CUDA memory. These are budget-audited baselines, not exact budget matches. A subsequent superiority study should tune each family on validation at a comparable total tuning budget and compare a cost/accuracy frontier. A larger raw-context pretrained transformer and actual RMT/Titans/TTT implementations remain future comparisons.

## Training and measurement

Cross-entropy is computed only on query events. Episode state resets before every batch; no cross-example/session memory leakage. Truncated BPTT retains state values but detaches the graph at configured boundaries. Default 32 events covers the default language and vision streams. Reducing this below the evidence/query distance can produce a misleading negative by preventing useful write credit assignment.

Choose a checkpoint by validation NLL, then run test once. Report normal accuracy/NLL and first/switch/return (language) or first/switch/late (vision) phase scores. Report three or more independent training seeds for comparative conclusions. Validation/test are generated identically across modes; the random query order makes phase accuracy an aggregate, not a calibrated change-detection latency metric.

Interventions: erase state before each event; shuffle state across batch streams at query time; increase blank/distractor gap to 3g+2; add visual pixel noise. A shuffled singleton tail is excluded from shuffle scoring. Batch one cannot support shuffle and is explicitly unavailable. Erasure/shuffle need not lower scores for an untrained or unused memory.

## Decision

The implementation is ready for experiments when tests, both training pipelines and checkpoint round trips pass. An architectural benefit requires repeated held-out gains beyond the conventional controls at an acceptable measured cost. A gain over current-input-only demonstrates use of temporal information; it does not establish a superior memory mechanism. Failure of rotation to beat no-rotation is a valid negative result.

Do not infer continual weight learning, protected memories, parallel Bayesian trajectories, general language understanding or real-video performance from these tests.
