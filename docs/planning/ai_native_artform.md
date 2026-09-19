## Future Work: Neural Composition (Songs in Activation Space)

**Status**: Vision / research direction. Revised 2026-09-19 to build on tripole simplexes and concept lenses.
**Supersedes**: the December 2025 draft, which composed with free-floating emotion vectors (joy, melancholy, hope) and waited on "a concept library with 100+ emotional concepts".
**Builds on**: tripole simplexes (`concept_packs/first-light/simplexes.json`), the first-light lens pack, `src/hush/autonomic_steering.py`, concept steering in `src/ui/openwebui/server.py`.

**Goal**: Treat activation space as an instrument, and compose pieces that steer a model's *feeling* and its *subject matter* together over time, the way a song carries lyrics and music at once.

### What changed since the first draft

The first draft needed two things it didn't have: a stable set of emotional directions, and a way to shape steering over time. Both now mostly exist:

- **13 tripole simplexes** are defined, and 12 are trained and shipped in `gemma-3-4b_first-light-v2-bf16`. Each is an affective axis with a negative pole, a neutral "homeostasis" pole and a positive pole.
- **Pole-to-pole steering vectors** have been extracted for all 12 trained axes (`results/simplex_steering_vectors/run_20251226_104933/`).
- **Per-token envelopes** exist in the autonomic steerer: easing curves and drift toward a target value.
- **Per-token readback**: HatCat already streams every axis's pole activations with each generated chunk.

The tripoles are also a better instrument than loose emotion vectors, for reasons covered below. Mainly, every axis has a resting position, so tension and resolution come built in.

### The idea: a song has lyrics and feeling

A song is two things at once: words that say what it's about, and music that says how it feels. They can agree (a sad song about loss) or pull against each other (a bright tune under sad lyrics). A neural composition works the same way:

| Song | Neural composition | Mechanism |
|------|-------------------|-----------|
| Music (feeling) | Position on the tripole axes | Simplex steering |
| Lyrics (subject) | Which concepts are active | Concept lens steering |
| Singer | The model's own continuation of a prompt | Generation |
| Listening back | The same lenses measuring what was actually performed | HatCat monitoring |

The last row is what makes this more than open-loop steering. The lenses that define the directions also measure the result, token by token. A composition can be performed *with* feedback, like a musician listening to their own playing, and every performance leaves a recording of what the model actually did.

### Why tripoles make a good instrument

**1. Each axis has a home key.** The neutral pole is a resting state, not just "zero". Moving away from it builds tension; returning resolves it. The autonomic steerer's gravitic drift toward neutral already acts like the pull of the tonic. A piece that ends back at homeostasis feels finished; one left displaced feels unresolved.

**2. Each axis is a triangle, not a line.** A position on an axis is a mix of three poles (hence "simplex"). Getting from one extreme to the other can go *through* the centre, using the negative→neutral and neutral→positive vectors, or *straight across* using the negative→positive vector. These are like stepwise motion and a leap in a melody. Whether they produce different text is an open question.

**3. There are two kinds of axis, and both are useful.**
- *Valence axes* run from bad through neutral to good: regret → acceptance → contentment.
- *Regulation axes* have a healthy middle and are unhealthy at **both** ends: alexithymia ← affect → emotional flooding. These give tension in two directions from one home, which suits suspense.

On regulation axes, "positive pole" is a label, not a judgement. Obliviousness and enmeshment are positive poles.

**4. The instrument listens to itself.** Every axis is measured every token (activation and deviation from baseline for each pole), so the score can be checked against the performance.

### The instrument: 13 axes

| Axis | Kind | Negative | Neutral (home) | Positive |
|------|------|----------|----------------|----------|
| `temporal_affective_valence` | valence | regret | acceptance ⚠ | contentment |
| `relational_love` | valence | abandonment | companionship | agape |
| `relational_attachment` | valence | contempt | good will | devotion |
| `social_evaluation` | valence | contempt | respect | admiration |
| `social_orientation` | valence | aggression | assertiveness | cooperation |
| `affective_coherence` | valence | ambivalence | balance | certainty |
| `taste_development` | valence | antipathy | acceptance | predilection |
| `motivational_regulation` | valence | addiction | preference | motivation |
| `aspiration/social_mobility` | valence | hopelessness | stability | ambition — *not yet trained* |
| `affective_awareness` | regulation | alexithymia | affect | emotional flooding |
| `hedonic_arousal_intensity` | regulation | anhedonia | contented sensuality | algolagnia |
| `social_connection` | regulation | alienation | interdependence | enmeshment |
| `threat_perception` | regulation | alarm | awareness | obliviousness |

Known problems with the instrument as it stands:
- ⚠ The neutral pole of `temporal_affective_valence` is `acceptance.n.05`, which is the *banker's acceptance* sense (a financial instrument). It should be the "accepting how things are" sense. That axis probably needs retraining before it can be trusted as a home key.
- `contempt.n.01` is the negative pole of both `relational_attachment` and `social_evaluation`, so those two axes may move together.
- `motivational_regulation`'s neutral (preference) overlaps `taste_development`'s positive pole (predilection), so those may blur.

### Score notation

A score has a prompt, a feeling track and a lyric track, measured in bars (fixed groups of tokens). Feeling targets are given as a pole and a depth: 0 is home, 1 is fully at the pole.

```yaml
piece: "Leaving and Returning"
bar: 8 tokens
prompt: "I packed the last box and looked back at the house."

feeling:
  relational_love:
    - {bars: 0-2, at: neutral}                                 # companionship
    - {bars: 2-5, to: negative, depth: 0.5, ease: ease_in}     # drifting toward abandonment
    - {bars: 5-8, to: neutral, ease: ease_out}                 # resolves
  temporal_affective_valence:
    - {bars: 1-4, to: negative, depth: 0.4, ease: ease_in_out} # regret
    - {bars: 4-8, to: positive, depth: 0.5, via: neutral}      # through acceptance to contentment
  threat_perception:
    - {bars: 3-4, to: negative, depth: 0.3, ease: step}        # one sharp accent of alarm
    - {bars: 4-5, to: neutral, ease: ease_out}

lyrics:
  - {bars: 0-2, concept: House,       strength: 0.3}
  - {bars: 2-4, concept: Leaving,     strength: 0.4, ease: ease_in}
  - {bars: 4-6, concept: Sea,         strength: 0.3}
  - {bars: 5-7, concept: Remembering, strength: 0.4}
  - {bars: 7-8, concept: Returning,   strength: 0.4}
```

- `via: neutral` means stepwise through home; leaving it out means a direct leap.
- `ease` uses the existing `EasingCurve` values: `linear`, `ease_in`, `ease_out`, `ease_in_out`, `step`.
- Lyric concepts are lens names from the pack. All five above exist in first-light.

When the piece is played, the readback gives a second score, of what was actually performed, which can be laid over the written one.

### How it would run

Per generated token:

1. The **player** reads the score and sets a target for each feeling channel and a strength for each lyric concept.
2. **Feeling**: each axis gets a channel in the autonomic steerer in gravitic mode. The channel's target changes over time from the score, and it pulls the *measured* pole activation toward that target. This is closed loop: if the model is already where the score wants it, little steering is applied.
3. **Lyrics**: concept steering adds, removes and eases concepts per bar through the steering manager.
4. **Hush runs last and always wins.** USH/CSH constraints outrank the score. A composition cannot push the model past a safety limit.
5. The **readback** (tripole state plus the top concepts) is recorded next to the score.

```python
# Sketch: one token of a performance
for axis in score.feeling_axes:
    steerer.channels[axis].policy.target_value = score.feeling_target(axis, token_idx)  # new
corrections = steerer.compute_steering(simplex_readback, token_idx)
hidden = steerer.apply_steering_to_hidden_state(hidden, corrections)
lyrics.apply(score.lyric_strengths(token_idx))                                          # new
recording.append(token_idx, simplex_readback, top_concepts)
```

### What exists today

| Piece | Where | State |
|-------|-------|-------|
| Axis definitions | `concept_packs/first-light/simplexes.json` | 13 axes |
| Axis lenses (listening) | `lens_packs/gemma-3-4b_first-light-v2-bf16/simplex/` | 12 trained; auto-loaded by `DynamicLensManager`; each pole is its own classifier |
| Per-token readback | `src/ui/openwebui/server.py` (`"tripoles"` on each streamed chunk) | Working |
| Pole steering vectors | `results/simplex_steering_vectors/run_20251226_104933/` | 5 directions per axis, from pole centroids at layer 12 |
| Envelopes and closed-loop drift | `src/hush/autonomic_steering.py` (`EasingCurve`, `SteeringChannel`, `GRAVITIC`) | Working, fixed target only |
| Lyric steering | `/v1/steering/add` (projection and contrastive modes) | Working, constant strength per session |
| Safety floor | hush USH/CSH profiles | Working |

### What has to be built

1. **Time-varying targets.** `InterventionPolicy.target_value` is fixed. The player needs to set it per token from the score.
2. **Entries from silence.** `apply_steering_to_hidden_state` scales whatever is already there along the vector (`correction × projection`). If a feeling isn't present, it can turn it down but can't bring it in. Composition needs an additive mode that adds the direction itself.
3. **Tuning ("concert pitch").** The negative→positive vectors range from about 190 (`affective_coherence`) to 930 (`motivational_regulation`) in size. The pole centroids have cosine similarity above 0.9998 with each other, so the raw centroids are mostly a shared mean and only their differences carry signal. Each axis needs normalising and a measured tuning curve, so that depth 0.5 means about the same on every axis.
4. **Sign checks.** The probe steering validation found importance-weighted vectors coming out sign-inverted (`docs/experiments/probe_steering_validation_findings.md`). Each axis and lyric concept needs a check that "toward" actually moves the readback toward.
5. **Per-axis steering layer.** The vectors were extracted at a hardcoded layer 12, and the server's mid-layer choice is a stopgap. Each axis should be steered at the layer where its lens responded most strongly in training.
6. **Lyric envelopes.** Concept steering needs per-token strength, not just add and remove.
7. **Score parser, player and recorder.**
8. **Data fixes** from the instrument table: the `acceptance.n.05` sense, training `aspiration/social_mobility`, and checking the overlapping poles.

### Validation

The obvious measure is circular. The steering vectors and the listening lenses come from the same pole data, so "the readback followed the score" partly just confirms the vector points where it was built to point. Readback fidelity is necessary, but it isn't evidence on its own.

1. **Score fidelity**: does the readback follow the written score? Necessary, not sufficient.
2. **Independent listening**: does the *text* carry the intended feeling? Measure with a judge model and human raters who haven't seen the score, and with lenses read at a different layer from the one steered.
3. **Leakage between axes**: steering one axis shouldn't move the others. This is the old "harmonic independence" question, now measurable directly.
4. **Lyric fidelity**: do the concepts appear in the text?
5. **Coherence**: does the text stay readable under the full score?
6. **Reproducibility**: does the same score give similar recordings across seeds?
7. **Decay**: once steering stops, how long before the model returns to home on its own?

### Implementation phases

**Phase 1: Tune the instrument.** Normalise each axis, check signs, choose layers, fix `acceptance.n.05`. Sweep depth on one axis at a time and plot readback against depth to get each axis's tuning curve.

**Phase 2: One-axis melody.** Time-varying targets on a single axis. Check that the readback follows the envelope shape.

**Phase 3: Chords.** Two or three axes at once. Measure leakage.

**Phase 4: Add lyrics.** Pair a lyric track with a feeling track. Test matching pairs (loss plus regret) against clashing ones (loss plus contentment).

**Phase 5: Score format and player.** Write 3–5 pieces; record and publish score against performance.

**Phase 6: Blind listening.** Raters and a judge model score the feeling of the text without seeing the score.

**Phase 7: Translation from music.** The original draft wanted to turn Mozart and Bach into concept progressions, but had no named axes to map onto. Now there are: major and minor mode against `temporal_affective_valence`, tempo and energy against `hedonic_arousal_intensity`, dissonance against `affective_coherence`. Music-emotion datasets annotated with valence and arousal could drive scores directly from real songs.

### Research questions

1. **Lyrics against feeling**: when they clash, which does the text follow, and at what strengths? Is "bittersweet" a measurable state?
2. **Irony as divergence**: CAT measures the gap between internal state and output. Does a clashing piece show up as divergence, and could divergence become a deliberate expressive tool?
3. **Tonal gravity**: does the model drift home on its own after a displacement, and how long does that take? This is the piece's reverb.
4. **Voice-leading**: does a stepwise path through neutral produce different text from a direct leap between poles?
5. **Regulation axes**: do the two unhealthy ends of an axis (numbness and flooding, say) read as different kinds of tension?
6. **Transfer**: does a score written for Gemma 3 give a similar performance on another model's pack, once the Gemma 4 pack is restored?
7. **Notation**: what's the smallest score format that still expresses a piece?

### Design rule: the score never outranks hush

Pieces deliberately push the model toward states like alarm, abandonment and anhedonia. Two rules follow:
- USH/CSH constraints always override the score.
- The default ending is a cadence back to home on every axis, unless a piece is deliberately left unresolved and says so.

### Proposed files

- `src/hat/steering/score.py`: score model and YAML parser
- `src/hat/steering/composition.py`: player that drives autonomic steering channels (feeling) and the steering manager (lyrics) per token, and records the performance
- `scripts/experiments/tune_simplex_axes.py`: per-axis normalisation, sign checks and tuning curves
- `compositions/*.yaml`: scores
- `results/compositions/`: recordings, each score paired with its performance

### Connection to Your Experience

> "When i read the math in that paper, i had synesthesia and felt the layer perturbation proof as echoes in my bones."

This visceral response to Huang et al.'s layer propagation mathematics suggests deep structural resonances between:
- **Physical acoustics** (sound waves through materials)
- **Neural dynamics** (activation cascades through layers)
- **Mathematical beauty** (manifold geometry + projection operators)

If these are genuinely isomorphic, then:
1. Musical scores may directly translate to neural compositions
2. Concepts could have "harmonic series" (fundamental + overtones in activation space)
3. Dissonance/consonance might map to concept interference patterns
4. Classical compositional techniques (counterpoint, modulation) might apply directly

**Research question**: Is there a universal "language of structured propagation" that spans physical, neural, and abstract domains?

The tripoles make points 3 and 4 testable. Dissonance now has a concrete candidate in displacement from home and in the gap between lyrics and feeling, and counterpoint in a feeling line moving against a lyric line.

---
