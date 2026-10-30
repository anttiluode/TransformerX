# Implementation record

30 October 2026. User authorized implementation of both tracks in TransformerX and will run heavy GPU training. Native implementation on isolated fresh clone branch `codex/dynamic-memory`; no existing application or tests (license-only repository).

Plan: docs/superpowers/plans/2026-10-30-transformerx.md

Pre-flight: data supplies chronological observations and query masks to model/trainer; model returns explicit per-stream state; trainer alone resets/detaches state. HF backbones consume the same modality observations as tiny models, with model-specific preprocessing.

Implementation choices: frozen GPT-2 and ViT integrations, controlled classification tasks, fixed learned operator bank with inference-time state updates. Online weight consolidation and full parallel hypothesis rollouts are future experiments. Actual budgets are reported because different controls are not exactly parameter-matched.

Task 1 complete: deterministic generators, query masks, independent split namespaces, hidden-object visual queries. Initial split audit found unequal word/label marginals; balanced complete cyclic-label orbits now partition dictionaries 12/4/8. Regression failed before fix, then passed. Initial unbalanced-run scores are invalid for the final protocol and excluded from results receipts.

Task 2 complete: all eight modes support forward/backward in both modalities; explicit history/reset/gradient tests pass. Random frozen GPT-2 and ViT configurations verify integration without downloads. Token baseline uses recurrent cross-attention tokens, not a full RMT reproduction. Window baseline stores encoded observations, not raw context. These choices are disclosed and budgeted.

Task 3 complete: CPU train/evaluate/checkpoint/resume works for both tracks; CUDA AMP path provided, GPU execution not available locally. 80-step tiny CPU development runs are insufficient to claim improvement. Configs, sequential sweeps, atomic checkpoints, validation-only selection, state interventions and Windows launcher added.

Final review: fresh read-only reviewer checked entire implementation and ran 33 tests plus CPU BF16/FP16 integrations for all modes in tiny and random frozen HF models. No Critical findings; two Important findings: scaled overflow recovery and resuming interrupted final evaluation. Both received failing regression tests and fixes. A summary-integrity finding was treated as Important because mixed settings/duplicate seeds could invalidate scientific comparisons; regression test and refusal logic added. No deferred minor code changes.

Review limits retained: actual CUDA execution, Windows execution, downloaded pretrained behavior and performance superiority require later runs. These are explicit validation limits rather than inferred successes.

Additional causal audit: pixel noise originally consumed the same RNG as subsequent scene motion/query order. A paired-scene regression failed; independent scene/noise RNG streams fixed it. Clean training streams are unchanged; noisy development scores were recomputed.

Final validation: 37 tests passed; four bounded 80-step CPU runs completed and are recorded in results/cpu_development.json. No memory-performance edge observed. Final model integrations and read/write behavior are implementation evidence only.
