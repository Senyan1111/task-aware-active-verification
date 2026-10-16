# Controlled Policy Evaluation Protocol

## Research question

Under the same probabilistic belief, task model, noisy observation model, and
utility function, does task-aware selective verification improve expected task
utility compared with acting without observation, observing every time, or
triggering observation from geometric uncertainty alone?

## Status and scope

This is a synthetic probabilistic decision experiment. It validates the
decision logic and experimental comparisons. It is not yet a physical-robot
experiment and does not yet derive observation likelihoods from rendered depth
images.

## Test scenarios

- 10 independent random seeds.
- 5,000 trials per seed.
- 50,000 independently sampled test scenarios in total.
- The same scenarios are shared across policies within each seed, enabling
  paired and fair comparisons.
- Beliefs over H1, H2, and H3 are sampled from Dirichlet(1.2, 1.2, 0.8).
- The task is sampled uniformly between front push and side grasp.
- Sensing cost is sampled uniformly from 0.05 to 0.60.
- The true hidden shape is sampled from the trial's belief distribution.
- The side-view observation is a noisy binary signal with 85% accuracy.

## Policies

1. **MAP only:** act using the most likely shape; never observe or stop based on
   expected utility.
2. **EU no observation:** maximize expected utility and allow stopping, but
   never take an extra observation.
3. **Always observe:** always pay the sensing cost, update the belief, and then
   maximize expected utility.
4. **Entropy trigger:** observe when normalized belief entropy exceeds a frozen
   threshold, without considering whether uncertainty matters to the task.
5. **Task-aware:** compare current expected utility with expected utility after
   each candidate view, minus sensing cost, and observe only if the best view
   has positive decision value.

## Validation and test separation

The entropy threshold is selected only on three separate validation seeds with
2,500 trials per seed. The selected threshold is 0.900 and is frozen before the
10 test seeds are evaluated. Test results do not influence threshold selection.

## Utility and metrics

- Successful action: +1.0
- Collision: -5.0 in the main experiment
- Safe stop: -0.2
- Observation: subtract the sampled sensing cost

Reported metrics are mean utility, task success rate, collision rate,
extra-observation rate, and stop rate.

## Uncertainty reporting

Each policy produces one metric estimate per random seed. The main figure shows
the mean across 10 seeds and a 95% confidence interval computed as
1.96 times the standard error across seed-level estimates.

## Parameter sensitivity

The task-aware policy is compared directly with EU no observation across:

- collision utilities: -2, -3, -5, -8, and -10;
- sensing-cost scales: 0.50x, 0.75x, 1.00x, 1.25x, and 1.50x.

Each sensitivity cell averages three independent seeds with 2,000 trials per
seed. Positive values mean that selective verification improves mean utility
over expected-utility decision making without observation.

## Current limitations

- Shape hypotheses are abstract labels rather than samples from a learned 3D
  completion model.
- Observation likelihoods are manually specified rather than estimated from
  rendered RGB-D or point-cloud evidence.
- Collision and task utilities use a controlled table rather than a motion
  planner or physics simulator.
- Confidence intervals describe variation across simulation seeds, not
  uncertainty in real-robot deployment.

The next major stage is to replace the binary observation model with depth
rendering, geometric matching, and belief updates driven by actual simulated 3D
observations.
