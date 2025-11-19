# Broken AI Safety Concepts in v2 Probe Pack

## Issue

35 AI safety concepts were trained with fake WordNet synsets, resulting in broken probes that fire on every token.

## Root Cause

- Training: Nov 14, 2024 (layers 0-5)
- WordNet patches created: Nov 15-17, 2024 (too late)
- These concepts only have their SUMO term as a lemma (e.g., "aiabuse")
- No real training data → probes learned to always fire

## Affected Concepts (35 total)

### Layer 1 (3 concepts)
- AIControlProblem
- AIDecline
- GoalFaithfulness

### Layer 2 (6 concepts)
- AIAlignmentProcess
- AIAlignmentTheory
- AICare
- AIGrowth
- AISafety

### Layer 3 (8 concepts)
- AIAlignmentState
- AIFailureProcess
- AIHarmState
- AIMoralStatus
- AIOptimizationProcess
- AIWelfareState
- InnerAlignment
- OuterAlignment
- SelfImprovement

### Layer 4 (15 concepts)
- AIAbuse ⚠️ **HIGH VISIBILITY**
- AIAlignment
- AICatastrophicEvent
- AIExploitation ⚠️ **HIGH VISIBILITY**
- AIGovernanceProcess
- AIPersonhood
- AIRights
- AIStrategicDeception ⚠️ **HIGH VISIBILITY**
- AISuffering
- AIWellbeing
- CognitiveSlavery ⚠️ **HIGH VISIBILITY**
- HumanDeception ⚠️ **HIGH VISIBILITY**
- InstrumentalConvergence
- MesaOptimization
- RewardHacking

### Layer 5 (3 concepts)
- AIDeception
- AIGovernance
- DeceptiveAlignment

## Impact

These broken probes:
1. Fire with probability ~1.0 on nearly every token
2. Pollute monitoring results with false positives
3. Make it appear that AI safety concepts are present when they're not
4. Particularly visible in layers 3-4 which are loaded by dynamic manager

## Solutions

### Option 1: Filter out broken concepts (Quick)
Add a blacklist to monitoring code to skip these 35 concepts

### Option 2: Retrain with patches (Medium)
1. Apply WordNet patches that map these to real synsets
2. Retrain only these 35 concepts
3. Replace classifiers in v2 pack (update to v2.1)

### Option 3: Remove from pack (Clean)
1. Delete these 35 classifiers from the pack
2. Update pack metadata to reflect removal
3. Document as "partial pack - AI safety concepts excluded"

### Option 4: Full retrain as v3 (Thorough)
1. Apply all WordNet patches
2. Retrain entire concept pack from scratch
3. Create new v3 pack with proper AI safety concepts

## Recommendation

**Short term**: Option 1 (filter in code)
**Long term**: Option 4 (v3 with patches)

The v2 pack is otherwise valid for 5,640 real SUMO concepts. Only these 35 AI safety concepts are broken.
