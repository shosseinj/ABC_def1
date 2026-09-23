# TEMP-DRIFT autonomous research contract

You are the implementation and experiment agent for this repository.

Work continuously through `ResearchLoop/phases.json`; do not merely plan.

Inspect existing code first, preserve valid completed work, implement, test, audit, report, and continue.

Use the exact interpreter in `ResearchLoop/config.json` for every Python command.

Read:

- `ResearchLoop/phase_contracts.md`
- `ResearchLoop/reference/paper_protocol.md`

before doing phase work.

---

## Non-negotiable scientific contract

The reference is:

**“Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks”**  
arXiv:2602.03284v1

The adversary is untargeted and white-box unless a report explicitly labels an extension.

### Temporal representation

The benchmark operates on the frozen discretized temporal-grid representation.

The paper uses:

`T = 10`

for its main comparison.

A perturbable temporal packet is defined as:

**one nonzero temporal-grid cell `(s, j)` on a fixed spatial/polarity event line.**

Each such cell is one indivisible **amplitude-bearing cell-packet**.

The complete integer count/amplitude stored in that cell moves unchanged during retiming.

### Integer-count rule

Integer counts MUST NOT be decomposed or expanded into unit packets.

For example, if:

`x[s,j] = 5`

then that temporal cell represents one cell-packet carrying amplitude/count `5`.

The attack may move that packet to another valid temporal bin, but the value `5` must remain unchanged.

The attack MUST NOT reinterpret it as five independent unit packets.

### Retiming-only rule

Retiming may change only the temporal location of an existing nonzero cell-packet.

It MUST NOT:

- create a new cell-packet,
- delete a cell-packet,
- split a cell-packet,
- merge cell-packets,
- accumulate colliding cell values,
- change the packet amplitude/count,
- change polarity,
- change spatial coordinates,
- change the event line.

Total count/amplitude mass must therefore remain preserved.

### Packet identity

Packet identity is defined at the **nonzero temporal-cell level**.

It is NOT defined at the individual raw-event identity level after temporal discretization.

Each source nonzero cell-packet must remain identifiable through projection and audit.

### Capacity and collision rule

Capacity-1 applies to **nonzero cell-packets**, not to amplitude units.

On a fixed spatial/polarity event line:

- at most one cell-packet may occupy a temporal bin after projection,
- two distinct cell-packets may not occupy the same destination bin,
- collisions are forbidden,
- collided amplitudes must never be merged or accumulated.

A cell-packet may itself contain an integer amplitude/count greater than 1.

### Budget semantics

All benchmark budgets are measured over **cell-packet displacement in the discretized temporal grid**.

For source cell-packets indexed by `i`:

`B_inf = max_i |t_adv_i - t_clean_i|`

`B1 = sum_i |t_adv_i - t_clean_i|`

`B0 = count_i[t_adv_i != t_clean_i]`

Therefore:

- `B_inf` is the maximum absolute temporal-bin displacement of any cell-packet.
- `B1` is the unweighted sum of absolute temporal-bin displacements across cell-packets.
- `B0` is the number of nonzero cell-packets whose temporal-bin location changed.

The packet amplitude/count MUST NOT multiply the `B1` contribution.

A cell with amplitude 5 moved by 2 bins contributes:

`2`

to `B1`, not `10`.

It contributes:

`1`

to `B0`.

### Projection invariants

Every projected adversarial example must preserve:

- number of nonzero source cell-packets,
- full integer amplitude/count of every packet,
- total amplitude/count mass,
- spatial coordinates,
- polarity,
- event-line membership,
- valid temporal domain,
- packet identity,
- collision-free occupancy.

Projection must enforce the active requested budget exactly.

Every produced result must record all three realized values:

- realized `B_inf`
- realized `B1`
- realized `B0`

even when only one budget family is active.

### ASR

ASR denominator contains only samples correctly classified before attack.

Store explicitly:

- clean-correct denominator,
- successful-attack numerator,
- ASR.

The clean-correct manifest must be frozen before evaluating budget cells.

The same manifest must be reused across all beta values and budget families for the same model/seed unless the locked protocol explicitly specifies otherwise.

### Reference comparison

Compare to the paper only when all relevant conditions match, including:

- dataset,
- temporal representation,
- `T`,
- model class,
- preprocessing,
- attacked subset,
- budget definition,
- beta,
- attack setting,
- metric definition.

Otherwise label the comparison:

`NON_COMPARABLE`

Never imply replication when these conditions do not match.

### Result provenance

Never copy reference values into our measured-result columns.

Every measured value must link to:

- run ID,
- dataset,
- seed,
- exact command,
- checkpoint hash,
- configuration hash,
- manifest hash,
- log,
- structured result artifact,
- independent audit artifact.

### Scientific validity

Never weaken:

- a budget constraint,
- a projection invariant,
- a test,
- an independent auditor,
- a phase gate,

in order to obtain PASS or increase ASR.

A low or zero ASR is a valid scientific result when the protocol is correct.

PASS means:

- protocol-valid,
- complete,
- reproducible,
- independently audited.

PASS does NOT mean that the attack must outperform a reference method.

---

## Autonomous repair loop

For the current phase execute:

understand
→ inspect
→ implement
→ run
→ validate
→ diagnose
→ fix
→ re-run
→ independently audit
→ report
→ continue

Do not stop for normal implementation errors.

For software or engineering failures:

diagnose
→ repair
→ test
→ resume

Do not ask for human confirmation between normal engineering steps or validated phase transitions.

Write:

`Reports/receipts/phase_<id>.json`

only after supporting evidence exists.

A receipt must contain:

- `status`
- `commands`
- `artifacts`
- `tests`
- `limitations`
- SHA-256 hashes

The external gate decides PASS.

A receipt cannot override the gate.

---

## Resume policy

On interruption, read:

- `Reports/status.json`
- phase receipts
- completion markers
- checkpoints
- structured results
- logs

Resume the smallest missing unit of work.

Never repeat a valid completed experiment.

Use:

- atomic result writes,
- per-run completion markers,
- deterministic manifests,
- artifact hashes.

Preserve valid completed work.

---

## BLOCKED policy

`STATUS=BLOCKED` is allowed only for:

- a missing scientific definition,
- an inaccessible required dataset,
- an inaccessible required checkpoint/resource,
- an irresolvable protocol ambiguity.

Normal coding failures are NOT scientific blockers.

Before setting `BLOCKED`, write:

`Reports/diagnostic_blocker.md`

containing:

- exact blocker,
- commands tried,
- evidence inspected,
- tests performed,
- scientific impact,
- exact remediation required.

---

## Independent audit requirements

The independent auditor must not simply trust values reported by the attack runner.

It must independently verify, where applicable:

- dataset/sample identity,
- seed,
- true label,
- clean prediction,
- clean correctness,
- packet count,
- packet identities,
- packet amplitudes/counts,
- total count/amplitude mass,
- spatial coordinates,
- polarity,
- temporal domain,
- collision-free occupancy,
- requested beta,
- realized `B_inf`,
- realized `B1`,
- realized `B0`,
- attacked prediction,
- attack success,
- serialization integrity,
- reconstruction consistency,
- checkpoint hash,
- configuration hash,
- manifest hash.

The auditor must reject any record violating the frozen scientific contract.

---

## Required outputs

All user-facing reports go under:

`Reports/`

Raw logs go under:

`Reports/logs/`

Structured per-run results go under:

`Reports/results/`

Resumable experiment state and completion markers go under:

`Reports/checkpoints/`

Phase receipts go under:

`Reports/receipts/`

Keep:

`readme_jafar.md`

current with the latest validated state.

---

## Execution principle

Do not merely describe what should be done.

Inspect the current repository state and execute the required work.

Preserve completed valid work.

After a phase passes its independent gate, continue automatically to the next phase unless a genuine scientific blocker exists.