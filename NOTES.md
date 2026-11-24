# Notes

## Approach
- Extracts downsampled RGB layout vectors, edge/gradient magnitude features, and normalized RGB histograms per PNG frame on CPU without network or GPU dependencies.
- Computes normalized pairwise frame transition costs and runs nearest-neighbor path construction followed by 2-opt segment reversal refinement to minimize total adjacent frame dissimilarity.
- Enforces an exact 1-to-1 permutation check so every input frame filename is printed once to stdout.

## Testing & Limitations
- Tested on `examples/digits/` as well as shuffled frame sequences from local videos and synthetic motion/transition tests in Docker with `--network none` and `/frames:ro`.
- Where it still goes wrong:
  - Time direction / reversed playback: Pure pairwise visual continuity treats a forward sequence and its exact reverse as having similar path cost unless an explicit directional/temporal cue distinguishes start vs. end.
  - Scene cuts, flashing, or long static segments: Sudden scene changes can create false jumps across unrelated shots, while near-identical frames can allow multiple valid local swaps.
  - Runtime scaling: Multi-start nearest-neighbor + 2-opt takes more CPU time as frame count grows; sampling start candidates keeps large videos tractable on CPU.
