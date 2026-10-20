# ModelLab

ModelLab is a small, forkable workbench for local model work on Apple Silicon
Macs: run models, collect labelled judgments, evaluate, and fine-tune.
Projects use it instead of building their own tooling. Other people fork it
and configure it for their own machines.

## Why it exists

Agents doing model work keep building their own tooling. This duplicates
effort, leaves machines in an inconsistent state, and produces scores that
cannot be compared across runs or projects. Examples: several agents researched
the same model separately; Python environments with different MLX versions;
model folders without a recorded version; per-project scripts to reach a
second machine; on 2026-09-28, two training runs started together made a
128 GiB Mac restart.

ModelLab is also for long unattended agent runs: an agent left to pursue a
clearly stated goal should drift less with ModelLab than without it. ModelLab
puts items next to their references wherever an agent reads a result
(`docs/on-the-data.md`).

## What ModelLab owns

| Layer | ModelLab provides |
|---|---|
| Machines and models | A locked environment. One model store with exact model identity. One way to run jobs on any configured machine, with memory safety. |
| Gold sets | A labelling tool configured per task. Versioned gold sets. |
| Evaluation | One harness. Each result cites its gold-set version, model identity and configuration. |
| Training and serving | Recipes over upstream trainers. Local serving through an OpenAI-compatible endpoint. |

Projects own their questions, data, labels, objectives and conclusions.
ModelLab does not choose a project's research design. It provides the checks
that show when a design cannot answer its question, and the budgets that make
drift from a stated goal visible. Hosted training services are later scope;
keep a path to them open when it costs nothing.

## Design rules

1. **Template, not instance.** The core is general: macOS on Apple Silicon first.
   Hosts, network details, paths, models, resource policy and labelling schemas
   are configuration. Do not hardcode one machine, network, model or project.
2. **Forkable.** The core works on its own and holds no personal data, private
   datasets or credentials. Any integration with other tools stays thin and
   optional.
3. **Adopt, then build.** Use maintained upstream tools. ModelLab's own code is
   thin glue, templates and configuration. Build only where no upstream tool
   fits, and record why (`docs/stack.md`).
4. **Safe at full power.** No job may take a machine down. Within that limit,
   use the machine's full capacity. ModelLab enforces this; each run's own
   settings must not be the only guard.
   ModelLab jobs yield first: when memory runs short, ModelLab stops its own
   jobs and never blocks or stops other work. Long jobs save checkpoints, so a
   stop costs little.
5. **Local by default.** No data, results or telemetry leave the machine unless
   configuration enables it explicitly.
6. **Comparable results.** Every result records the gold-set version, model
   identity and configuration that produced it.
7. **Adoption by pull.** A project moves onto a ModelLab layer when its work
   next touches that layer. Do not stop running work to migrate it. Each
   migration improves the shared version.
8. **Proof by use.** A component is proven when a real project task runs on it.
   Keep records short: what it does, how to run it, and where it is known to work.

## Working on ModelLab

- Each layer's document in `docs/` says what is built, how it works and how it
  was verified. Change the code, its tests and its document together.
- Tests: see README.md. Tests that need an installed environment, a model, the
  Hugging Face or a second Mac run only when their `MODELLAB_TEST_*` variable is set.
- The command line (`bin/modellab`, `modellab/cli.py` and what it imports)
  needs Python 3.11 or later and nothing else. Code never assumes a shell.
- ModelLab does not fork or patch upstream packages, except as `docs/stack.md`
  (Patches) allows.
- Run long work as ModelLab jobs (`modellab run`), and watch it with
  `modellab wait`.
