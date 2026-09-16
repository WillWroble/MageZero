"""
Config loading, validation, and per-generation curriculum resolution.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
import yaml


VALID_MODES = {"minimax", "mcts", "offline", "fixed"}
VALID_LOG_LEVELS = {"ALL", "GAMEPLAY", "ACTIONS", "ERROR"}


# ── dataclasses ──────────────────────────────────────────────

@dataclass
class Opponent:
    deck: str
    mode: str
    version: Optional[int] = None
    offline : bool = True


@dataclass
class TrainingFlags:
    analyze_dataset: bool = True
    generate_plots: bool = True
    eval_previous_model: bool = True


@dataclass
class RunConfig:
    deck: str
    version: int
    start_from_version: Optional[int]
    generations: int
    games_per_gen: int
    replay_buffer_gens: int
    opponents: list[Opponent]
    training: TrainingFlags
    log_level: str
    curriculum_path: str
    max_jvms: int = 1


@dataclass
class Priors:
    binary: bool = False
    priority: bool = False
    target: bool = False
    opponent: bool = False


@dataclass
class PriorSchedule:
    binary: Optional[int] = 2
    priority: Optional[int] = 3
    target: Optional[int] = 4
    opponent: Optional[int] = None


@dataclass
class GenSettings:
    """Fully resolved settings for a single generation."""
    gen: int
    td_discount: float
    prior_temperature: float
    priors: Priors


@dataclass
class CurriculumConfig:
    defaults_td_discount: float
    defaults_prior_temperature: float
    defaults_priors: Priors
    prior_schedule: PriorSchedule
    schedule: list[dict]  # raw sparse override entries


# ── loading ──────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_run(path: str = "configs/run.yml") -> RunConfig:
    raw = _load_yaml(path)

    opponents = []
    for o in raw.get("opponents", []):
        mode = o["mode"]
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid opponent mode '{mode}', must be one of {VALID_MODES}")
        opponents.append(Opponent(
            deck=o["deck"],
            mode=mode,
            version=o.get("version"),
            offline=o.get("offline", True)
        ))

    log_level = raw.get("log_level", "ACTIONS")
    if log_level not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log_level '{log_level}', must be one of {VALID_LOG_LEVELS}")

    training_raw = raw.get("training", {})

    return RunConfig(
        deck=raw["deck"],
        version=raw.get("version", 1),
        start_from_version=raw.get("start_from_version"),
        generations=raw["generations"],
        games_per_gen=raw["games_per_gen"],
        replay_buffer_gens=raw.get("replay_buffer_gens", 3),
        opponents=opponents,
        training=TrainingFlags(**training_raw),
        log_level=log_level,
        curriculum_path=raw.get("curriculum", "configs/curriculum.yml"),
        max_jvms=raw.get("max_jvms", 1),
    )


def load_curriculum(path: str = "configs/curriculum.yml") -> CurriculumConfig:
    raw = _load_yaml(path)
    defaults = raw.get("defaults", {})
    priors_raw = defaults.get("priors", {})
    ps_raw = raw.get("prior_schedule", {})

    return CurriculumConfig(
        defaults_td_discount=defaults.get("td_discount", 0.70),
        defaults_prior_temperature=defaults.get("prior_temperature", 1.5),
        defaults_priors=Priors(**priors_raw),
        prior_schedule=PriorSchedule(**ps_raw),
        schedule=raw.get("schedule", []),
    )


# ── resolution ───────────────────────────────────────────────

def resolve_gen(curriculum: CurriculumConfig, gen: int) -> GenSettings:
    """
    Walk the sparse schedule up to `gen`, inheriting forward.
    Then apply prior_schedule thresholds.
    """
    td = curriculum.defaults_td_discount
    temp = curriculum.defaults_prior_temperature

    # walk schedule in order, applying overrides
    for entry in curriculum.schedule:
        if entry["gen"] > gen:
            break
        if "td_discount" in entry:
            td = entry["td_discount"]
        if "prior_temperature" in entry:
            temp = entry["prior_temperature"]

    # resolve priors: start from defaults, then flip on by schedule
    priors = Priors(
        binary=curriculum.defaults_priors.binary,
        priority=curriculum.defaults_priors.priority,
        target=curriculum.defaults_priors.target,
        opponent=curriculum.defaults_priors.opponent,
    )
    ps = curriculum.prior_schedule
    if ps.binary is not None and gen >= ps.binary:
        priors.binary = True
    if ps.priority is not None and gen >= ps.priority:
        priors.priority = True
    if ps.target is not None and gen >= ps.target:
        priors.target = True
    if ps.opponent is not None and gen >= ps.opponent:
        priors.opponent = True

    return GenSettings(gen=gen, td_discount=td, prior_temperature=temp, priors=priors)


# ── convenience ──────────────────────────────────────────────

def load_all(run_path: str = "configs/run.yml") -> tuple[RunConfig, CurriculumConfig]:
    run = load_run(run_path)
    curriculum = load_curriculum(run.curriculum_path)
    return run, curriculum