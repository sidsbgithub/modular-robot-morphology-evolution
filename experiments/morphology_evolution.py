"""
Evolving robot bodies (Ariel TreeGenome) towards a set of target bodies.

Fitness (minimised): mean + std of the weighted tree-edit distance between the
candidate body and each of the five targets (tree_edit_distance.py).

The EA is a (mu + lambda) generational algorithm with mu = lambda = population
size: ~50 % of the population is selected as parents, each offspring is
produced by subtree crossover followed by mutation, and the best mu of
parents + offspring survive (elitist). All variation operators are Ariel's
built-in tree operators; the surrounding EA design is our own.

Modes
    (default)   one EA run with the final tuned configuration
    --final     the experiment reported in the paper: five variants that differ
                only in the mutation operator, on held-out seeds, plus a
                random-search baseline at the same evaluation budget, plus the
                analysis and figures (A1_final_analysis.py)
    --screen    parameter screening used to choose and justify the settings:
                one-factor-at-a-time axes (mutation / crossover / selection),
                a full mutation x crossover factorial, and an optional second
                stage that re-runs the top-K configurations on extra seeds and
                ranks them with Holm-corrected Mann-Whitney U tests
    --random-search   stand-alone random-search baseline

Options that exist only for the screening stages and are NOT used in the
reported final experiment: k-tournament and linear-ranking parent selection,
mutation_events > 1, crossover_rate < 1.

Screening runs are parallel (--workers), crash-tolerant and resumable: a run
whose result.csv / generation_results.csv already exist on disk is reused.
"""

import argparse
import copy
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from rich.console import Console
from rich.traceback import install

from ariel.ec import EA, EAOperation, Individual, Population

from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    get_tree_depth,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    validate_tree_depth,
)

from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)

from tree_edit_distance import mean_plus_std_tree_edit_distance


install()
console = Console()

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_DIR = SCRIPT_DIR / "target_bodies"

DATA_DIR = SCRIPT_DIR / "__data__" / "A1_evolution_tree"
DATA_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_POP_SIZE = 60
DEFAULT_GENERATIONS = 100
DEFAULT_MAX_MODULES = 20
DEFAULT_SEED = 42

# The experiment reported in the paper (--final): five EA variants that differ
# only in the mutation operator, everything else fixed to the tuned settings
# below, on seeds that were never used during tuning.
FINAL_OPERATORS = ["subtree", "mixed", "replace", "shrink", "hoist"]
FINAL_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
FINAL_SETTINGS = {
    "mutation_rate": 0.75,
    "mutation_events": 1,
    "crossover": "subtree",
    "crossover_rate": 1.0,
    "selection": "truncation",
    "tournament_k": 2,
    "rank_pressure": 1.5,
}

MAX_DEPTH = 12

# Per-axis figures become unreadable beyond this many configurations;
# larger screens (e.g. the factorial) plot only the top configurations.
MAX_CONFIGS_PER_FIGURE = 25

VALID_MUTATIONS = [
    "replace",
    "subtree",
    "shrink",
    "hoist",
    "mixed",
]

VALID_CROSSOVERS = [
    "subtree",
    "none",
]

VALID_SELECTIONS = [
    "truncation",
    "tournament",
    "rank",
]

# Shared baseline used to hold "nuisance" knobs fixed while screening one axis.
BASELINE = {
    "mutation_operator": "mixed",
    "mutation_rate": 0.5,
    "mutation_events": 1,
    "crossover": "subtree",
    "crossover_rate": 1.0,
    "selection": "truncation",
    "tournament_k": 2,
    "rank_pressure": 1.5,
}


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)

    return np.random.default_rng(seed)


def load_targets() -> list:
    target_files = sorted(TARGET_DIR.glob("*.json"))

    if not target_files:
        raise FileNotFoundError(f"No target JSON files found in: {TARGET_DIR}")

    targets = [load_graph_from_json(str(path)) for path in target_files]

    console.log(f"Loaded {len(targets)} target bodies from {TARGET_DIR}")

    return targets


def calculate_fitness(
    genome: TreeGenome,
    targets: list,
) -> float:
    """
    Convert a TreeGenome to a phenotype and calculate the assignment fitness.

    Lower is better.

    Fitness = mean tree-edit-distance + population standard deviation
    """

    try:
        body = genome.to_networkx()

        if body.number_of_nodes() == 0:
            return float("inf")

        return float(
            mean_plus_std_tree_edit_distance(
                body,
                targets,
            )
        )

    except Exception:
        return float("inf")


def is_valid_genome(genome: TreeGenome) -> bool:
    """
    Check whether a TreeGenome is valid.
    """

    try:
        if genome is None:
            return False

        if len(genome.nodes) == 0:
            return False

        validate_genome_dict(genome.to_dict())

        validate_tree_depth(
            genome,
            max_depth=MAX_DEPTH,
        )

        return True

    except Exception:
        return False


class TreeEvolution:
    """(mu + lambda) EA over Ariel tree genomes; one run = one seed (see module docstring)."""

    def __init__(
        self,
        targets: list,
        pop_size: int,
        generations: int,
        max_modules: int,
        seed: int,
        mutation_operator: str = "subtree",
        mutation_rate: float = 0.75,
        mutation_events: int = 1,
        crossover: str = "subtree",
        crossover_rate: float = 1.0,
        selection: str = "truncation",
        tournament_k: int = 2,
        rank_pressure: float = 1.5,
        run_name: str = "run",
        output_dir: Path | None = None,
    ):
        self.targets = targets

        self.pop_size = pop_size
        self.generations = generations
        self.max_modules = max_modules
        self.seed = seed

        self.mutation_operator = mutation_operator
        self.mutation_rate = mutation_rate
        self.mutation_events = mutation_events

        self.crossover = crossover
        self.crossover_rate = crossover_rate

        self.selection = selection
        self.tournament_k = tournament_k
        self.rank_pressure = rank_pressure

        self.run_name = run_name

        self.rng = seed_everything(seed)

        self.evaluations = 0
        self.generation_stats = []

        if output_dir is None:
            output_dir = DATA_DIR / run_name

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.db_path = self.output_dir / "evolution.sqlite"

    def create_individual(self) -> Individual:
        """
        Create a random valid tree individual.
        """

        genome = random_tree(max_modules=self.max_modules)

        if not is_valid_genome(genome):
            raise RuntimeError("random_tree() generated an invalid genome.")

        individual = Individual()

        individual.genotype = genome.to_dict()

        individual.tags["selected"] = False
        individual.tags["mutate"] = False
        individual.tags["valid"] = True

        individual.requires_eval = True

        return individual

    def initialize_population(self) -> Population:
        population = Population([self.create_individual() for _ in range(self.pop_size)])

        population = self.evaluate(population)

        return population

    def evaluate(
        self,
        population: Population,
    ) -> Population:
        for individual in population.unevaluated:
            try:
                genome = TreeGenome.from_dict(individual.genotype)

                if not is_valid_genome(genome):
                    individual.fitness = float("inf")
                    individual.tags["valid"] = False

                else:
                    individual.fitness = calculate_fitness(
                        genome,
                        self.targets,
                    )

                    individual.tags["valid"] = bool(np.isfinite(individual.fitness))

            except Exception:
                individual.fitness = float("inf")
                individual.tags["valid"] = False

            individual.requires_eval = False

            self.evaluations += 1

        return population

    def parent_selection(
        self,
        population: Population,
    ) -> Population:
        """
        Select approximately 50% of the population as parents.

        Lower fitness is better.
        """

        for individual in population:
            individual.tags["selected"] = False

        if self.selection == "truncation":
            ordered = population.sort(
                sort="min",
                attribute="fitness_",
            )

            cutoff = max(
                2,
                len(ordered) // 2,
            )

            for individual in ordered[:cutoff]:
                individual.tags["selected"] = True

        elif self.selection == "tournament":
            individuals = list(population.alive)

            target_count = max(
                2,
                len(individuals) // 2,
            )

            selected = []

            # Safety valve: with a small population and/or large
            # tournament_k, distinct winners can run out before
            # target_count is reached.
            max_attempts = target_count * 50

            attempts = 0

            while len(selected) < target_count and attempts < max_attempts:
                attempts += 1

                k = min(self.tournament_k, len(individuals))

                if k < 2:
                    break

                contestants = self.rng.choice(
                    individuals,
                    size=k,
                    replace=False,
                )

                winner = min(
                    contestants,
                    key=lambda ind: (float("inf") if ind.fitness_ is None else ind.fitness_),
                )

                if winner not in selected:
                    selected.append(winner)

            for individual in selected:
                individual.tags["selected"] = True

        elif self.selection == "rank":
            ordered = list(
                population.alive.sort(
                    sort="min",
                    attribute="fitness_",
                )
            )

            n = len(ordered)

            target_count = max(
                2,
                n // 2,
            )

            if n >= 2:
                # Linear ranking (Eiben & Smith): rank 0 is the best
                # individual. Selection pressure s in [1, 2]; s=1 is
                # uniform, s=2 maximally favours the best individual.
                s = self.rank_pressure

                ranks = np.arange(n)

                probabilities = (2 - s) / n + 2 * (n - 1 - ranks) * (s - 1) / (n * (n - 1))

                probabilities = probabilities / probabilities.sum()

                chosen_indices = self.rng.choice(
                    n,
                    size=target_count,
                    replace=False,
                    p=probabilities,
                )

                for index in chosen_indices:
                    ordered[index].tags["selected"] = True

        return population

    def apply_single_mutation(
        self,
        genome: TreeGenome,
        mutation_operator: str,
    ) -> TreeGenome:
        mutated = copy.deepcopy(genome)

        if mutation_operator == "replace":
            mutate_replace_node(mutated)

        elif mutation_operator == "subtree":
            mutate_subtree_replacement(
                mutated,
                max_modules=self.max_modules,
            )

        elif mutation_operator == "shrink":
            mutate_shrink(mutated)

        elif mutation_operator == "hoist":
            mutate_hoist(mutated)

        else:
            raise ValueError(f"Unknown mutation operator: {mutation_operator}")

        return mutated

    def choose_mixed_mutation(self) -> str:
        """
        Select one mutation operator for mixed mutation.

        The probabilities are intentionally kept fixed so that
        'mixed' represents a reproducible mutation strategy.
        """

        operators = [
            "replace",
            "subtree",
            "shrink",
            "hoist",
        ]

        probabilities = [
            0.40,
            0.40,
            0.10,
            0.10,
        ]

        return str(
            self.rng.choice(
                operators,
                p=probabilities,
            )
        )

    def mutate_genome(
        self,
        genome: TreeGenome,
    ) -> TreeGenome:
        """
        Apply mutation according to:

            mutation_rate
            mutation_operator
            mutation_events

        Example:

            rate=0.5, events=2

        means there is a 50% chance of mutation and,
        when mutation occurs, two mutation events are applied.
        """

        mutated = copy.deepcopy(genome)

        if self.rng.random() > self.mutation_rate:
            return mutated

        for _ in range(self.mutation_events):
            if self.mutation_operator == "mixed":
                operator = self.choose_mixed_mutation()

            else:
                operator = self.mutation_operator

            try:
                mutated = self.apply_single_mutation(
                    mutated,
                    operator,
                )

            except Exception:
                return copy.deepcopy(genome)

            if not is_valid_genome(mutated):
                return copy.deepcopy(genome)

        return mutated

    def crossover_genomes(
        self,
        parent_a: TreeGenome,
        parent_b: TreeGenome,
    ) -> TreeGenome:
        """
        Apply subtree crossover and return one valid child.

        crossover_rate gates how often crossover actually fires for a
        given mating event; when it does not fire (or crossover=="none"),
        the child starts as a copy of parent_a.
        """

        if self.crossover == "none":
            return copy.deepcopy(parent_a)

        if self.rng.random() > self.crossover_rate:
            return copy.deepcopy(parent_a)

        try:
            child_a, child_b = crossover_subtree(
                copy.deepcopy(parent_a),
                copy.deepcopy(parent_b),
            )

            candidates = [
                child_a,
                child_b,
            ]

            self.rng.shuffle(candidates)

            for child in candidates:
                if is_valid_genome(child):
                    return child

        except Exception:
            pass

        return copy.deepcopy(parent_a)

    def reproduction(
        self,
        population: Population,
    ) -> Population:
        """
        Generate offspring from selected parents.

        Population size doubles temporarily:
            N parents + N offspring

        Survivor selection then returns it to N.
        """

        parents = [
            individual for individual in population if individual.tags.get("selected", False)
        ]

        if not parents:
            return population

        offspring = []

        while len(offspring) < self.pop_size:
            parent_a = self.rng.choice(parents)
            parent_b = self.rng.choice(parents)

            genome_a = TreeGenome.from_dict(parent_a.genotype)

            genome_b = TreeGenome.from_dict(parent_b.genotype)

            child_genome = self.crossover_genomes(
                genome_a,
                genome_b,
            )

            original_genome = copy.deepcopy(child_genome)

            child_genome = self.mutate_genome(child_genome)

            if not is_valid_genome(child_genome):
                child_genome = original_genome

            child = Individual()

            child.genotype = child_genome.to_dict()

            child.tags["selected"] = False
            child.tags["mutate"] = True
            child.tags["valid"] = True

            child.requires_eval = True

            offspring.append(child)

        population.extend(offspring)

        return population

    def record_statistics(self, population: Population) -> Population:
        """Record population statistics after each generation."""

        alive = [
            ind
            for ind in population
            if ind.alive and ind.tags.get("valid", False) and ind.fitness_ is not None
        ]

        if not alive:
            return population

        fitnesses = np.array(
            [float(ind.fitness_) for ind in alive],
            dtype=float,
        )

        # Tree size/depth of the current best individual, so growth
        # ("bloat") or shrinkage over generations can be inspected directly
        # instead of inferred indirectly from fitness alone.
        best_individual = min(alive, key=lambda ind: float(ind.fitness_))

        try:
            best_genome = TreeGenome.from_dict(best_individual.genotype)
            best_modules = len(best_genome.nodes)
            best_depth = get_tree_depth(best_genome)
        except Exception:
            best_modules = None
            best_depth = None

        mean_modules = float(
            np.mean([len(TreeGenome.from_dict(ind.genotype).nodes) for ind in alive])
        )

        generation = len(self.generation_stats)

        self.generation_stats.append(
            {
                "generation": generation,
                "best_fitness": float(np.min(fitnesses)),
                "mean_fitness": float(np.mean(fitnesses)),
                "std_fitness": float(np.std(fitnesses)),
                "worst_fitness": float(np.max(fitnesses)),
                "best_modules": best_modules,
                "best_depth": best_depth,
                "mean_modules": mean_modules,
                "evaluations": int(self.evaluations),
            }
        )

        return population

    def survivor_selection(
        self,
        population: Population,
    ) -> Population:
        """
        Keep the best pop_size individuals.

        Lower fitness is better: (mu + lambda) elitist survival.
        """

        def fitness_key(individual: Individual) -> float:
            return float("inf") if individual.fitness_ is None else individual.fitness_

        alive = sorted(population.alive, key=fitness_key)

        survivors = set(id(individual) for individual in alive[: self.pop_size])

        for individual in alive:
            if id(individual) not in survivors:
                individual.alive = False

        return population

    def run(self):
        """
        Run the complete EA.
        """

        start_time = time.time()

        console.log(f"Starting {self.run_name}")

        console.log(
            f"Mutation: {self.mutation_operator}, "
            f"rate={self.mutation_rate}, "
            f"events={self.mutation_events}"
        )

        console.log(f"Crossover: {self.crossover}, crossover_rate={self.crossover_rate}")

        console.log(
            f"Selection: {self.selection}, "
            f"tournament_k={self.tournament_k}, "
            f"rank_pressure={self.rank_pressure}"
        )

        console.log(
            f"Population={self.pop_size}, "
            f"Generations={self.generations}, "
            f"Max modules={self.max_modules}, "
            f"Seed={self.seed}"
        )

        population = self.initialize_population()

        self.record_statistics(population)

        operations = [
            EAOperation(self.parent_selection),
            EAOperation(self.reproduction),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),
            EAOperation(self.record_statistics),
        ]

        ea = EA(
            population,
            operations=operations,
            num_steps=self.generations,
            is_maximisation=False,
            db_file_path=self.db_path,
            quiet=True,
        )

        self.engine = ea.engine

        ea.run()

        elapsed = time.time() - start_time

        best = ea.get_solution(
            "best",
            only_alive=False,
        )

        generation_df = pd.DataFrame(self.generation_stats)
        generation_df["run_name"] = self.run_name
        generation_df["mutation_operator"] = self.mutation_operator
        generation_df["mutation_rate"] = self.mutation_rate
        generation_df["mutation_events"] = self.mutation_events
        generation_df["crossover"] = self.crossover
        generation_df["crossover_rate"] = self.crossover_rate
        generation_df["selection"] = self.selection
        generation_df["tournament_k"] = self.tournament_k
        generation_df["rank_pressure"] = self.rank_pressure
        generation_df["seed"] = self.seed
        generation_df.to_csv(
            self.output_dir / "generation_results.csv",
            index=False,
        )

        if best is not None:
            best_genome = TreeGenome.from_dict(best.genotype)

            best_fitness = float(best.fitness_)

            best_modules = len(best_genome.nodes)

            best_edges = len(best_genome.edges)

        else:
            best_fitness = float("inf")
            best_modules = 0
            best_edges = 0

        if self.generation_stats:
            final_stats = self.generation_stats[-1]
            final_mean = float(final_stats["mean_fitness"])
            final_std = float(final_stats["std_fitness"])
        else:
            final_mean = np.nan
            final_std = np.nan

        result = {
            "run_name": self.run_name,
            "mutation_operator": self.mutation_operator,
            "mutation_rate": self.mutation_rate,
            "mutation_events": self.mutation_events,
            "crossover": self.crossover,
            "crossover_rate": self.crossover_rate,
            "selection": self.selection,
            "tournament_k": self.tournament_k,
            "rank_pressure": self.rank_pressure,
            "population": self.pop_size,
            "generations": self.generations,
            "max_modules": self.max_modules,
            "seed": self.seed,
            "best_fitness": best_fitness,
            "best_modules": best_modules,
            "best_edges": best_edges,
            "final_mean_fitness": final_mean,
            "final_std_fitness": final_std,
            "evaluations": self.evaluations,
            "runtime_seconds": elapsed,
        }

        pd.DataFrame([result]).to_csv(
            self.output_dir / "result.csv",
            index=False,
        )

        console.log(
            f"Finished {self.run_name}: "
            f"best={best_fitness:.4f}, "
            f"modules={best_modules}, "
            f"evaluations={self.evaluations}, "
            f"time={elapsed:.2f}s"
        )

        return result, generation_df

    def dispose_database(self) -> None:
        """
        Release the sqlite engine and delete the database file.

        The per-generation CSVs already hold everything the analysis needs;
        the sqlite file is ~9 MB per run at pop=60/gens=100, which is
        prohibitive across thousands of screening runs.
        """

        engine = getattr(self, "engine", None)

        if engine is not None:
            engine.dispose()

        try:
            self.db_path.unlink(missing_ok=True)
        except OSError:
            pass


def run_single_experiment(
    targets: list,
    args,
):
    """
    Run one EA configuration.
    """

    selection_suffix = args.selection

    if args.selection == "tournament":
        selection_suffix = f"tournament-k{args.tournament_k}"
    elif args.selection == "rank":
        selection_suffix = f"rank-sp{args.rank_pressure:g}"

    run_name = (
        f"single_"
        f"{args.mutation}_"
        f"r{args.mutation_rate}_"
        f"e{args.mutation_events}_"
        f"{args.crossover}-cxr{args.crossover_rate:g}_"
        f"{selection_suffix}_"
        f"s{args.seed}"
    )

    output_dir = DATA_DIR / run_name

    experiment = TreeEvolution(
        targets=targets,
        pop_size=args.pop,
        generations=args.generations,
        max_modules=args.max_modules,
        seed=args.seed,
        mutation_operator=args.mutation,
        mutation_rate=args.mutation_rate,
        mutation_events=args.mutation_events,
        crossover=args.crossover,
        crossover_rate=args.crossover_rate,
        selection=args.selection,
        tournament_k=args.tournament_k,
        rank_pressure=args.rank_pressure,
        run_name=run_name,
        output_dir=output_dir,
    )

    result, generation_df = experiment.run()

    console.rule("Final Result")

    console.print(pd.DataFrame([result]).to_string(index=False))

    return result, generation_df


def run_random_search(
    targets: list,
    budget: int,
    seed: int,
    max_modules: int,
    run_name: str = "random_search",
) -> pd.DataFrame:
    """
    Pure random search over the same genome space, at a fixed evaluation
    budget, so it can be compared directly against an EA run that used the
    same number of fitness evaluations.

    Returns a per-evaluation DataFrame with the running best-so-far fitness,
    which lines up with the EA's best_fitness-vs-evaluations curve.
    """

    seed_everything(seed)

    best_so_far = float("inf")
    records = []

    for evaluation in range(1, budget + 1):
        genome = random_tree(max_modules=max_modules)

        if is_valid_genome(genome):
            fitness = calculate_fitness(genome, targets)
        else:
            fitness = float("inf")

        if fitness < best_so_far:
            best_so_far = fitness

        records.append(
            {
                "evaluation": evaluation,
                "fitness": fitness,
                "best_so_far": best_so_far,
            }
        )

    df = pd.DataFrame(records)
    df["run_name"] = run_name
    df["seed"] = seed
    df["max_modules"] = max_modules

    return df


def build_mutation_axis_configs(args) -> list[dict]:
    """Sweep mutation operator x rate x events; crossover/selection fixed."""

    configs = []

    for operator in args.screen_mutations:
        for rate in args.screen_rates:
            for events in args.screen_events:
                configs.append(
                    {
                        "axis": "mutation",
                        "mutation_operator": operator,
                        "mutation_rate": float(rate),
                        "mutation_events": int(events),
                        "crossover": args.screen_crossover,
                        "crossover_rate": BASELINE["crossover_rate"],
                        "selection": args.screen_selection,
                        "tournament_k": BASELINE["tournament_k"],
                        "rank_pressure": BASELINE["rank_pressure"],
                    }
                )

    return configs


def build_crossover_axis_configs(args) -> list[dict]:
    """Sweep crossover mechanism x crossover_rate; mutation/selection fixed."""

    configs = []

    for crossover in args.screen_crossover_options:
        if crossover == "none":
            configs.append(
                {
                    "axis": "crossover",
                    "mutation_operator": BASELINE["mutation_operator"],
                    "mutation_rate": BASELINE["mutation_rate"],
                    "mutation_events": BASELINE["mutation_events"],
                    "crossover": "none",
                    "crossover_rate": 1.0,
                    "selection": args.screen_selection,
                    "tournament_k": BASELINE["tournament_k"],
                    "rank_pressure": BASELINE["rank_pressure"],
                }
            )

        else:
            for rate in args.screen_crossover_rates:
                configs.append(
                    {
                        "axis": "crossover",
                        "mutation_operator": BASELINE["mutation_operator"],
                        "mutation_rate": BASELINE["mutation_rate"],
                        "mutation_events": BASELINE["mutation_events"],
                        "crossover": crossover,
                        "crossover_rate": float(rate),
                        "selection": args.screen_selection,
                        "tournament_k": BASELINE["tournament_k"],
                        "rank_pressure": BASELINE["rank_pressure"],
                    }
                )

    return configs


def build_selection_axis_configs(args) -> list[dict]:
    """Sweep selection scheme (+ its own knob); mutation/crossover fixed."""

    configs = []

    for selection in args.screen_selection_options:
        if selection == "tournament":
            for k in args.screen_tournament_k:
                configs.append(
                    {
                        "axis": "selection",
                        "mutation_operator": BASELINE["mutation_operator"],
                        "mutation_rate": BASELINE["mutation_rate"],
                        "mutation_events": BASELINE["mutation_events"],
                        "crossover": args.screen_crossover,
                        "crossover_rate": BASELINE["crossover_rate"],
                        "selection": "tournament",
                        "tournament_k": int(k),
                        "rank_pressure": BASELINE["rank_pressure"],
                    }
                )

        elif selection == "rank":
            for pressure in args.screen_rank_pressure:
                configs.append(
                    {
                        "axis": "selection",
                        "mutation_operator": BASELINE["mutation_operator"],
                        "mutation_rate": BASELINE["mutation_rate"],
                        "mutation_events": BASELINE["mutation_events"],
                        "crossover": args.screen_crossover,
                        "crossover_rate": BASELINE["crossover_rate"],
                        "selection": "rank",
                        "tournament_k": BASELINE["tournament_k"],
                        "rank_pressure": float(pressure),
                    }
                )

        else:  # truncation
            configs.append(
                {
                    "axis": "selection",
                    "mutation_operator": BASELINE["mutation_operator"],
                    "mutation_rate": BASELINE["mutation_rate"],
                    "mutation_events": BASELINE["mutation_events"],
                    "crossover": args.screen_crossover,
                    "crossover_rate": BASELINE["crossover_rate"],
                    "selection": "truncation",
                    "tournament_k": BASELINE["tournament_k"],
                    "rank_pressure": BASELINE["rank_pressure"],
                }
            )

    return configs


def build_factorial_configs(args) -> list[dict]:
    """
    Full mutation x crossover factorial; selection pinned to --screen-selection.

    Unlike the one-factor-at-a-time axes, this captures interactions: the
    best mutation setting may depend on how much crossover is applied.
    """

    crossover_levels = []

    for crossover in args.screen_crossover_options:
        if crossover == "none":
            crossover_levels.append(("none", 1.0))
        else:
            for rate in args.screen_crossover_rates:
                crossover_levels.append((crossover, float(rate)))

    configs = []

    for operator in args.screen_mutations:
        for rate in args.screen_rates:
            for events in args.screen_events:
                for crossover, crossover_rate in crossover_levels:
                    configs.append(
                        {
                            "axis": "factorial",
                            "mutation_operator": operator,
                            "mutation_rate": float(rate),
                            "mutation_events": int(events),
                            "crossover": crossover,
                            "crossover_rate": crossover_rate,
                            "selection": args.screen_selection,
                            "tournament_k": BASELINE["tournament_k"],
                            "rank_pressure": BASELINE["rank_pressure"],
                        }
                    )

    return configs


AXIS_BUILDERS = {
    "mutation": build_mutation_axis_configs,
    "crossover": build_crossover_axis_configs,
    "selection": build_selection_axis_configs,
    "factorial": build_factorial_configs,
}


def get_screening_configurations(args) -> list[dict]:
    axes = (
        ["mutation", "crossover", "selection"] if args.screen_axis == "all" else [args.screen_axis]
    )

    configurations = []

    for axis in axes:
        configurations.extend(AXIS_BUILDERS[axis](args))

    return configurations


def mutation_label(config: dict) -> str:
    return (
        f"{config['mutation_operator']} r={config['mutation_rate']:g} e={config['mutation_events']}"
    )


def crossover_level_label(config: dict) -> str:
    if config["crossover"] == "none":
        return "none"
    return f"cxr={config['crossover_rate']:g}"


def config_label(config: dict) -> str:
    """Short human-readable label for the knob(s) that vary on this axis."""

    axis = config["axis"]

    if axis == "mutation":
        return mutation_label(config)

    if axis == "crossover":
        if config["crossover"] == "none":
            return "none"
        return f"subtree {crossover_level_label(config)}"

    if axis == "selection":
        if config["selection"] == "tournament":
            return f"tournament k={config['tournament_k']}"
        if config["selection"] == "rank":
            return f"rank s={config['rank_pressure']:g}"
        return "truncation"

    if axis == "factorial":
        return f"{mutation_label(config)} | {crossover_level_label(config)}"

    return "config"


def safe_name(text: str) -> str:
    """Filesystem-safe run name ('|' is forbidden on Windows)."""

    text = text.replace(" | ", "__").replace("=", "").replace(" ", "_")

    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def build_job(
    config: dict,
    seed: int,
    args,
    screening_dir: Path,
    pop_size: int | None = None,
    generations: int | None = None,
) -> dict:
    run_name = safe_name(f"{config['axis']}_{config_label(config)}_s{seed}")

    return {
        "config": config,
        "seed": int(seed),
        "run_name": run_name,
        "output_dir": screening_dir / config["axis"] / run_name,
        "pop_size": args.screen_pop if pop_size is None else pop_size,
        "generations": args.screen_generations if generations is None else generations,
        "max_modules": args.max_modules,
        "keep_db": args.keep_db,
        "force_rerun": args.force_rerun,
    }


def _load_cached_job(job: dict) -> tuple[dict, pd.DataFrame] | None:
    """
    Reuse a finished run's CSVs if they exist and were produced by exactly
    this configuration/budget/seed. Makes long screens resumable.
    """

    output_dir = Path(job["output_dir"])
    result_path = output_dir / "result.csv"
    generation_path = output_dir / "generation_results.csv"

    if not (result_path.exists() and generation_path.exists()):
        return None

    try:
        result_df = pd.read_csv(result_path)
        generation_df = pd.read_csv(generation_path)
    except Exception:
        return None

    if len(result_df) != 1 or generation_df.empty:
        return None

    result = result_df.iloc[0].to_dict()
    config = job["config"]

    try:
        same = (
            int(result["population"]) == job["pop_size"]
            and int(result["generations"]) == job["generations"]
            and int(result["max_modules"]) == job["max_modules"]
            and int(result["seed"]) == job["seed"]
            and str(result["mutation_operator"]) == config["mutation_operator"]
            and np.isclose(float(result["mutation_rate"]), config["mutation_rate"])
            and int(result["mutation_events"]) == config["mutation_events"]
            and str(result["crossover"]) == config["crossover"]
            and np.isclose(float(result["crossover_rate"]), config["crossover_rate"])
            and str(result["selection"]) == config["selection"]
            and int(result["tournament_k"]) == config["tournament_k"]
            and np.isclose(float(result["rank_pressure"]), config["rank_pressure"])
        )
    except (KeyError, ValueError, TypeError):
        return None

    if not same:
        return None

    return result, generation_df


def _run_screening_job(job: dict) -> tuple[dict, pd.DataFrame]:
    config = job["config"]
    label = config_label(config)

    cached = None if job.get("force_rerun", False) else _load_cached_job(job)

    if cached is not None:
        result, generation_df = cached
        result["cached"] = True

    else:
        targets = load_targets()

        experiment = TreeEvolution(
            targets=targets,
            pop_size=job["pop_size"],
            generations=job["generations"],
            max_modules=job["max_modules"],
            seed=job["seed"],
            mutation_operator=config["mutation_operator"],
            mutation_rate=config["mutation_rate"],
            mutation_events=config["mutation_events"],
            crossover=config["crossover"],
            crossover_rate=config["crossover_rate"],
            selection=config["selection"],
            tournament_k=config["tournament_k"],
            rank_pressure=config["rank_pressure"],
            run_name=job["run_name"],
            output_dir=job["output_dir"],
        )

        result, generation_df = experiment.run()
        result["cached"] = False

        if not job.get("keep_db", False):
            experiment.dispose_database()

    result["axis"] = config["axis"]
    result["config_label"] = label

    generation_df["axis"] = config["axis"]
    generation_df["config_label"] = label

    return result, generation_df


def _execute_jobs(
    jobs: list[dict],
    workers: int,
) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    """
    Run jobs (in parallel if workers > 1). A failing run is logged and
    skipped rather than aborting the whole screen; progress lines carry an
    ETA so multi-hour screens can be monitored.
    """

    results: list[dict] = []
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []

    total = len(jobs)
    start = time.time()

    def log_progress(done: int, result: dict) -> None:
        elapsed = time.time() - start
        remaining = elapsed / done * (total - done) if done else 0.0
        cached = " (cached)" if result.get("cached") else ""
        console.log(
            f"[{done}/{total}] {result['run_name']}: "
            f"best={float(result['best_fitness']):.4f}{cached} | "
            f"elapsed={elapsed / 60:.1f} min, eta={remaining / 60:.1f} min"
        )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_job = {pool.submit(_run_screening_job, job): job for job in jobs}

            for done, future in enumerate(as_completed(future_to_job), start=1):
                job = future_to_job[future]

                try:
                    result, generation_df = future.result()
                except Exception as exc:
                    failures.append({"run_name": job["run_name"], "error": repr(exc)})
                    console.log(
                        f"[{done}/{total}] FAILED {job['run_name']}: {exc!r}",
                        style="red",
                    )
                    continue

                results.append(result)
                frames.append(generation_df)
                log_progress(done, result)

    else:
        for done, job in enumerate(jobs, start=1):
            console.rule(f"Run {done}/{total}")

            try:
                result, generation_df = _run_screening_job(job)
            except Exception as exc:
                failures.append({"run_name": job["run_name"], "error": repr(exc)})
                console.log(
                    f"FAILED {job['run_name']}: {exc!r}",
                    style="red",
                )
                continue

            results.append(result)
            frames.append(generation_df)
            log_progress(done, result)

    return results, frames, failures


def run_screening(
    targets: list,
    args,
):
    """
    Stage 1: run every configuration of the chosen axis (or the factorial)
    on the screening seeds. Stage 2 (optional): re-run the top-K
    configurations on extra seeds and rank them with statistical tests.
    """

    screening_dir = DATA_DIR / args.screen_name
    screening_dir.mkdir(parents=True, exist_ok=True)

    configurations = get_screening_configurations(args)

    jobs = [
        build_job(config, seed, args, screening_dir)
        for config in configurations
        for seed in args.screen_seeds
    ]

    console.rule("EA Configuration Screening - Stage 1")

    console.log(f"Output: {screening_dir}")
    console.log(f"Axis: {args.screen_axis}")
    console.log(f"Configurations: {len(configurations)}")
    console.log(f"Seeds: {args.screen_seeds}")
    console.log(f"Stage 1 runs: {len(jobs)}")
    console.log(
        f"Population={args.screen_pop}, "
        f"Generations={args.screen_generations}, "
        f"Max modules={args.max_modules}"
    )
    console.log(f"Workers: {args.workers}")

    if args.refine_top_k > 0:
        console.log(
            f"Stage 2 planned: top {args.refine_top_k} configurations x "
            f"{len(args.refine_seeds)} refinement seeds"
        )

    results, frames, failures = _execute_jobs(jobs, args.workers)

    if not results:
        raise RuntimeError("Every screening run failed - see console output.")

    raw_results = pd.DataFrame(results)
    generation_results = pd.concat(frames, ignore_index=True)

    finalists: list[tuple[str, str]] = []

    if args.refine_top_k > 0:
        refine_results, refine_frames, refine_failures, finalists = run_refinement(
            raw_results, configurations, args, screening_dir
        )

        failures.extend(refine_failures)

        if not refine_results.empty:
            raw_results = pd.concat([raw_results, refine_results], ignore_index=True)
            generation_results = pd.concat([generation_results, refine_frames], ignore_index=True)

    raw_results.to_csv(screening_dir / "raw_results.csv", index=False)
    generation_results.to_csv(screening_dir / "generation_results.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(screening_dir / "failed_runs.csv", index=False)
        console.log(
            f"{len(failures)} run(s) failed - see failed_runs.csv",
            style="red",
        )

    console.rule("Screening Summary by Axis")

    effect_sizes = []

    for axis in sorted(raw_results["axis"].unique()):
        axis_dir = screening_dir / axis
        axis_dir.mkdir(parents=True, exist_ok=True)

        axis_results = raw_results[raw_results["axis"] == axis]
        axis_generations = generation_results[generation_results["axis"] == axis]

        summary = summarize_configs(axis_results).drop(columns="axis")

        summary.to_csv(axis_dir / "summary.csv", index=False)

        console.rule(f"Axis: {axis} ({len(summary)} configurations)")
        console.print(summary.head(MAX_CONFIGS_PER_FIGURE).to_string(index=False))

        if len(summary) > MAX_CONFIGS_PER_FIGURE:
            console.log(
                f"... {len(summary) - MAX_CONFIGS_PER_FIGURE} more rows in "
                f"{axis_dir / 'summary.csv'}"
            )

        generate_axis_figures(
            axis_results,
            axis_generations,
            axis,
            axis_dir,
        )

        if axis == "factorial":
            analyze_factorial(axis_results, axis_dir)

        effect_sizes.append(
            {
                "axis": axis,
                "range_mean_best_fitness": float(
                    summary["mean_best_fitness"].max() - summary["mean_best_fitness"].min()
                ),
                "best_config": summary.iloc[0]["config_label"],
                "best_mean_fitness": float(summary.iloc[0]["mean_best_fitness"]),
            }
        )

    effect_df = pd.DataFrame(effect_sizes).sort_values(
        "range_mean_best_fitness",
        ascending=False,
    )

    effect_df.to_csv(
        screening_dir / "axis_effect_sizes.csv",
        index=False,
    )

    console.rule("Which axis moved fitness the most?")
    console.print(effect_df.to_string(index=False))
    console.log(
        "Larger 'range_mean_best_fitness' means that axis separated its "
        "configurations more clearly at this screening budget: a stronger "
        "candidate for your one research question. This is a screening "
        "signal, not a conclusion - confirm with a proper final experiment."
    )

    if finalists:
        final_ranking(raw_results, generation_results, finalists, screening_dir)

    write_screening_readme(
        args,
        configurations,
        raw_results,
        screening_dir,
    )

    console.rule("Screening Complete")

    console.log(f"Results saved to: {screening_dir}")

    return (
        raw_results,
        generation_results,
        effect_df,
    )


def generate_axis_figures(
    axis_results: pd.DataFrame,
    axis_generations: pd.DataFrame,
    axis: str,
    output_dir: Path,
):
    """
    Generate a final-fitness comparison and a convergence plot for one axis,
    faceted by config_label (the knob(s) that varied on this axis).
    """

    summary = (
        axis_results.groupby("config_label", as_index=False)
        .agg(
            mean_best=("best_fitness", "mean"),
            std_best=("best_fitness", "std"),
        )
        .sort_values("mean_best")
    )

    total_configs = len(summary)
    summary = summary.head(MAX_CONFIGS_PER_FIGURE)

    suffix = f" (top {len(summary)} of {total_configs})" if total_configs > len(summary) else ""

    axis_generations = axis_generations[
        axis_generations["config_label"].isin(summary["config_label"])
    ]

    plt.figure(figsize=(10, 6))

    x = np.arange(len(summary))

    plt.errorbar(
        x,
        summary["mean_best"],
        yerr=summary["std_best"],
        fmt="o",
        capsize=4,
    )

    plt.xticks(
        x,
        summary["config_label"],
        rotation=45,
        ha="right",
    )

    plt.ylabel("Best fitness")
    plt.xlabel("Configuration")
    plt.title(f"{axis.capitalize()} Screening: Final Best Fitness{suffix}")
    plt.tight_layout()

    plt.savefig(
        output_dir / "final_fitness_comparison.png",
        dpi=300,
    )

    plt.close()

    convergence = axis_generations.groupby(["config_label", "generation"], as_index=False).agg(
        mean_best=("best_fitness", "mean"),
        std_best=("best_fitness", "std"),
    )

    plt.figure(figsize=(10, 6))

    for label, group in convergence.groupby("config_label"):
        group = group.sort_values("generation")

        plt.plot(
            group["generation"],
            group["mean_best"],
            label=label,
        )

        if len(group) > 1:
            lower = group["mean_best"] - group["std_best"].fillna(0)
            upper = group["mean_best"] + group["std_best"].fillna(0)

            plt.fill_between(
                group["generation"],
                lower,
                upper,
                alpha=0.15,
            )

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title(f"{axis.capitalize()} Screening: Convergence{suffix}")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()

    plt.savefig(
        output_dir / "convergence.png",
        dpi=300,
    )

    plt.close()


def summarize_configs(raw_results: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_results.groupby(["axis", "config_label"], as_index=False)
        .agg(
            mean_best_fitness=("best_fitness", "mean"),
            std_best_fitness=("best_fitness", "std"),
            median_best_fitness=("best_fitness", "median"),
            min_best_fitness=("best_fitness", "min"),
            mean_evaluations=("evaluations", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            number_of_runs=("seed", "count"),
        )
        .sort_values("mean_best_fitness")
        .reset_index(drop=True)
    )


def run_refinement(
    raw_results: pd.DataFrame,
    configurations: list[dict],
    args,
    screening_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[tuple[str, str]]]:
    """
    Stage 2 of the racing design: re-run only the top-K stage-1
    configurations on additional seeds, so near-tied candidates are
    compared with enough statistical power to separate them (or to show
    they cannot be separated).
    """

    top = summarize_configs(raw_results).head(args.refine_top_k)

    lookup = {(c["axis"], config_label(c)): c for c in configurations}

    finalists: list[tuple[str, str]] = []
    jobs: list[dict] = []

    for row in top.itertuples(index=False):
        key = (row.axis, row.config_label)
        finalists.append(key)

        done_seeds = set(
            raw_results.loc[
                (raw_results["axis"] == row.axis)
                & (raw_results["config_label"] == row.config_label),
                "seed",
            ].astype(int)
        )

        for seed in args.refine_seeds:
            if int(seed) not in done_seeds:
                jobs.append(build_job(lookup[key], seed, args, screening_dir))

    console.rule("Stage 2: Refinement of Finalists")
    console.log(f"Finalists ({len(finalists)}):")

    for axis, label in finalists:
        console.log(f"  [{axis}] {label}")

    console.log(f"Stage 2 runs: {len(jobs)}")

    results, frames, failures = _execute_jobs(jobs, args.workers)

    refine_results = pd.DataFrame(results)
    refine_frames = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return refine_results, refine_frames, failures, finalists


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values; NaN entries pass through."""

    indexed = [(p, i) for i, p in enumerate(p_values) if not np.isnan(p)]
    m = len(indexed)

    adjusted = [float("nan")] * len(p_values)
    running = 0.0

    for rank, (p, i) in enumerate(sorted(indexed)):
        running = max(running, (m - rank) * p)
        adjusted[i] = min(1.0, running)

    return adjusted


def final_ranking(
    raw_results: pd.DataFrame,
    generation_results: pd.DataFrame,
    finalists: list[tuple[str, str]],
    screening_dir: Path,
) -> pd.DataFrame | None:
    """
    Rank finalists on all their seeds with mean, 95% CI and a two-sided
    Mann-Whitney U test of each finalist against the best one (Holm-corrected).
    """

    groups: list[tuple[str, str, np.ndarray]] = []

    for axis, label in finalists:
        values = raw_results.loc[
            (raw_results["axis"] == axis) & (raw_results["config_label"] == label),
            "best_fitness",
        ].to_numpy(dtype=float)

        values = values[np.isfinite(values)]

        if len(values) >= 2:
            groups.append((axis, label, values))

    if not groups:
        return None

    groups.sort(key=lambda group: group[2].mean())

    best_axis, best_label, best_values = groups[0]

    rows = []

    for rank, (axis, label, values) in enumerate(groups, start=1):
        n = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        half_width = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))

        if (axis, label) == (best_axis, best_label):
            p_value = float("nan")
        else:
            p_value = float(stats.mannwhitneyu(values, best_values, alternative="two-sided").pvalue)

        rows.append(
            {
                "rank": rank,
                "axis": axis,
                "config_label": label,
                "number_of_runs": n,
                "mean_best_fitness": mean,
                "std_best_fitness": std,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "median_best_fitness": float(np.median(values)),
                "min_best_fitness": float(values.min()),
                "p_value_vs_best": p_value,
            }
        )

    ranking = pd.DataFrame(rows)

    ranking["p_value_vs_best_holm"] = holm_correction(ranking["p_value_vs_best"].tolist())

    ranking["tied_with_best"] = ranking["p_value_vs_best_holm"].isna() | (
        ranking["p_value_vs_best_holm"] >= 0.05
    )

    ranking.to_csv(screening_dir / "final_ranking.csv", index=False)

    console.rule("Final Ranking of Finalists (all seeds)")
    console.print(ranking.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    tied = ranking[(ranking["rank"] > 1) & ranking["tied_with_best"]]

    if tied.empty:
        console.log(
            f"CLEAR WINNER: {best_label} - significantly better than every "
            f"other finalist (Holm-corrected Mann-Whitney p < 0.05).",
            style="green",
        )
    else:
        console.log(
            f"NO SINGLE CLEAR WINNER: {best_label} has the lowest mean but is "
            f"statistically tied (Holm-corrected p >= 0.05) with "
            f"{len(tied)} finalist(s): " + "; ".join(tied["config_label"]),
            style="yellow",
        )

    labels = [label for _, label, _ in groups]
    samples = [values for _, _, values in groups]

    plt.figure(figsize=(max(8.0, 0.7 * len(labels) + 4.0), 6))
    plt.boxplot(samples, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
    plt.ylabel("Best fitness (lower is better)")
    plt.title("Finalists: best fitness across all seeds")
    plt.tight_layout()
    plt.savefig(screening_dir / "finalists_boxplot.png", dpi=300)
    plt.close()

    finalist_curves = generation_results.merge(
        pd.DataFrame(finalists, columns=["axis", "config_label"]),
        on=["axis", "config_label"],
    )

    curves = finalist_curves.groupby(["config_label", "generation"], as_index=False).agg(
        mean_best=("best_fitness", "mean"),
        std_best=("best_fitness", "std"),
    )

    plt.figure(figsize=(10, 6))

    for label in labels:
        group = curves[curves["config_label"] == label].sort_values("generation")

        plt.plot(group["generation"], group["mean_best"], label=label)

        plt.fill_between(
            group["generation"],
            group["mean_best"] - group["std_best"].fillna(0),
            group["mean_best"] + group["std_best"].fillna(0),
            alpha=0.12,
        )

    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Finalists: convergence (mean +/- std over all seeds)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(screening_dir / "finalists_convergence.png", dpi=300)
    plt.close()

    return ranking


def analyze_factorial(
    axis_results: pd.DataFrame,
    axis_dir: Path,
) -> None:
    """
    Marginal (main) effects of each factor and the mutation x crossover
    interaction, which a one-factor-at-a-time screen cannot show.
    """

    df = axis_results.copy()
    records = df.to_dict("records")

    df["mutation_config"] = [mutation_label(r) for r in records]
    df["crossover_level"] = [crossover_level_label(r) for r in records]

    tables = []

    for factor in [
        "mutation_operator",
        "mutation_rate",
        "mutation_events",
        "crossover_level",
    ]:
        table = (
            df.groupby(factor)["best_fitness"]
            .agg(["mean", "std", "median", "count"])
            .reset_index()
            .rename(columns={factor: "level"})
        )
        table.insert(0, "factor", factor)
        table["level"] = table["level"].astype(str)
        tables.append(table)

    marginal = pd.concat(tables, ignore_index=True)
    marginal.to_csv(axis_dir / "marginal_effects.csv", index=False)

    console.rule("Factorial: marginal (main) effects")
    console.print(marginal.to_string(index=False))

    def crossover_order(level: str) -> float:
        return -1.0 if level == "none" else float(level.split("=")[1])

    operator_table = (
        df.groupby(["mutation_operator", "crossover_level"])["best_fitness"].mean().unstack()
    )
    operator_table = operator_table[sorted(operator_table.columns, key=crossover_order)]
    operator_table.to_csv(axis_dir / "interaction_operator_x_crossover.csv")

    console.rule("Factorial: operator x crossover (mean best fitness)")
    console.print(operator_table.to_string(float_format=lambda v: f"{v:.3f}"))

    pivot = df.pivot_table(
        index="mutation_config",
        columns="crossover_level",
        values="best_fitness",
        aggfunc="mean",
    )
    pivot = pivot[sorted(pivot.columns, key=crossover_order)]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    pivot.to_csv(axis_dir / "interaction_mutation_x_crossover.csv")

    plt.figure(figsize=(10, max(6.0, 0.28 * len(pivot) + 2.0)))
    image = plt.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    plt.colorbar(image, label="Mean best fitness (lower is better)")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=6)
    plt.xlabel("Crossover level")
    plt.ylabel("Mutation configuration (sorted best to worst)")
    plt.title("Mutation x Crossover interaction")
    plt.tight_layout()
    plt.savefig(axis_dir / "factorial_heatmap.png", dpi=300)
    plt.close()


def write_screening_readme(
    args,
    configurations,
    raw_results,
    output_dir,
):
    """
    Write a human-readable description of the screening experiment.
    """

    axes_run = sorted(set(config["axis"] for config in configurations))

    text = f"""
A1 EA Configuration Screening
==============================

Purpose
-------
Screen one or more EA axes (mutation, crossover, selection) to decide
which single aspect to make the research question, and to pick
defensible fixed values for the other (nuisance) parameters.

Representation
--------------
TreeGenome

Screen name
-----------
{args.screen_name}

Axes screened this run
-----------------------
{", ".join(axes_run)}

Stage 2 refinement (racing)
---------------------------
{
        (
            f"top {args.refine_top_k} configurations re-run on seeds "
            + ", ".join(str(x) for x in args.refine_seeds)
            if args.refine_top_k > 0
            else "disabled"
        )
    }

Shared baseline (used to hold non-screened axes fixed)
-------------------------------------------------------
{BASELINE}

Maximum modules
---------------
{args.max_modules}

Population (screening budget, NOT the final budget)
-----------------------------------------------------
{args.screen_pop}

Generations (screening budget, NOT the final budget)
-------------------------------------------------------
{args.screen_generations}

Seeds
-----
{", ".join(str(x) for x in args.screen_seeds)}

Number of configurations
------------------------
{len(configurations)}

Total independent runs
----------------------
{len(raw_results)}

Files
-----
raw_results.csv
    One row per independent run, tagged with its axis and config_label.

generation_results.csv
    Generation-by-generation convergence statistics, all axes combined.

axis_effect_sizes.csv
    Per-axis spread of mean best fitness across its configurations -
    use this to decide which axis is your research question.

<axis>/summary.csv, <axis>/final_fitness_comparison.png, <axis>/convergence.png
    Per-axis aggregated results and figures.

factorial/marginal_effects.csv, factorial/interaction_*.csv, factorial/factorial_heatmap.png
    Main effects and the mutation x crossover interaction (factorial axis only).

final_ranking.csv, finalists_boxplot.png, finalists_convergence.png
    Stage 2 output: finalists on all seeds with 95% CIs and Holm-corrected
    Mann-Whitney U tests against the best configuration.

failed_runs.csv
    Only written if some runs raised an exception.

Interpretation
--------------
This is a screening experiment at a reduced budget, not the final
statistical experiment. Its purpose is to (1) identify which single
aspect shows the clearest, most tellable effect, and (2) provide
justified values for whichever parameters are NOT your research
question. The final assignment experiments must vary only ONE aspect
between your two EA variants, use at least five independent runs per
variant at the full budget, and include a random-search baseline at
the same evaluation budget (see run_random_search / --random-search).
"""

    with open(
        output_dir / "README.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(text.strip())


def _random_search_job(job: dict) -> pd.DataFrame:
    targets = load_targets()

    return run_random_search(
        targets,
        job["budget"],
        job["seed"],
        job["max_modules"],
        run_name=job["run_name"],
    )


def run_final_experiment(
    targets: list,
    args,
):
    """
    Five EA variants that differ only in the mutation operator, on held-out
    seeds, plus a random-search baseline at the identical evaluation budget,
    followed by the statistical analysis and the report figures.

    Finished runs found on disk are reused; pass --force-rerun to recompute.
    """

    experiment_dir = DATA_DIR / args.final_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    budget = args.pop * (args.generations + 1)

    configurations = [
        {"axis": "mutation", "mutation_operator": operator, **FINAL_SETTINGS}
        for operator in args.final_operators
    ]

    jobs = [
        build_job(
            config,
            seed,
            args,
            experiment_dir,
            pop_size=args.pop,
            generations=args.generations,
        )
        for config in configurations
        for seed in args.final_seeds
    ]

    console.rule("Final Experiment")
    console.log(f"Output: {experiment_dir}")
    console.log(f"Variants (mutation operator): {args.final_operators}")
    console.log(f"Fixed settings: {FINAL_SETTINGS}")
    console.log(
        f"Population={args.pop}, Generations={args.generations}, "
        f"Max modules={args.max_modules}, budget={budget} evaluations per run"
    )
    console.log(f"Held-out seeds: {args.final_seeds}")
    console.log(
        f"EA runs: {len(jobs)} ({len(configurations)} variants x "
        f"{len(args.final_seeds)} seeds); workers: {args.workers}"
    )

    results, frames, failures = _execute_jobs(jobs, args.workers)

    if not results:
        raise RuntimeError("Every EA run failed - see console output.")

    raw_results = pd.DataFrame(results)
    generation_results = pd.concat(frames, ignore_index=True)

    raw_results.to_csv(experiment_dir / "raw_results.csv", index=False)
    generation_results.to_csv(experiment_dir / "generation_results.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(experiment_dir / "failed_runs.csv", index=False)
        console.log(f"{len(failures)} run(s) failed - see failed_runs.csv", style="red")

    random_dir = DATA_DIR / "random_search"
    random_dir.mkdir(parents=True, exist_ok=True)

    random_path = random_dir / f"random_search_budget{budget}.csv"
    random_df = None

    if random_path.exists() and not args.force_rerun:
        existing = pd.read_csv(random_path)

        if (
            set(args.final_seeds) <= set(existing["seed"].unique())
            and int(existing["evaluation"].max()) == budget
        ):
            random_df = existing[existing["seed"].isin(args.final_seeds)]
            console.log(f"Reusing random search from {random_path}")

    if random_df is None:
        console.rule("Random Search Baseline")
        console.log(f"{len(args.final_seeds)} seeds x {budget} evaluations")

        random_jobs = [
            {
                "seed": int(seed),
                "budget": budget,
                "max_modules": args.max_modules,
                "run_name": f"random_search_b{budget}_s{seed}",
            }
            for seed in args.final_seeds
        ]

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                random_frames = list(pool.map(_random_search_job, random_jobs))
        else:
            random_frames = [_random_search_job(job) for job in random_jobs]

        random_df = pd.concat(random_frames, ignore_index=True)
        random_df.to_csv(random_path, index=False)

        console.log(f"Saved random search to {random_path}")

    from analyze_results import analyze

    analyze(experiment=args.final_name, random_csv=random_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Ariel tree-genome evolutionary robot morphology experiments")
    )

    parser.add_argument(
        "--pop",
        type=int,
        default=DEFAULT_POP_SIZE,
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
    )

    parser.add_argument(
        "--max-modules",
        type=int,
        default=DEFAULT_MAX_MODULES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--mutation",
        choices=VALID_MUTATIONS,
        default="subtree",
    )

    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--mutation-events",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--crossover",
        choices=VALID_CROSSOVERS,
        default="subtree",
    )

    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=1.0,
        help="Probability that crossover fires per mating event.",
    )

    parser.add_argument(
        "--selection",
        choices=VALID_SELECTIONS,
        default="truncation",
    )

    parser.add_argument(
        "--tournament-k",
        type=int,
        default=2,
        help="Tournament size when --selection=tournament.",
    )

    parser.add_argument(
        "--rank-pressure",
        type=float,
        default=1.5,
        help="Linear ranking selection pressure in [1, 2] when --selection=rank.",
    )

    parser.add_argument(
        "--final",
        action="store_true",
        help="Run the reported experiment: five mutation-operator variants "
        "on held-out seeds + random-search baseline + analysis.",
    )

    parser.add_argument(
        "--final-name",
        default="final_experiment",
        help="Output folder name under __data__/A1_evolution_tree.",
    )

    parser.add_argument(
        "--final-operators",
        nargs="+",
        choices=VALID_MUTATIONS,
        default=FINAL_OPERATORS,
    )

    parser.add_argument(
        "--final-seeds",
        nargs="+",
        type=int,
        default=FINAL_SEEDS,
        help="Held-out seeds (never used during tuning).",
    )

    parser.add_argument(
        "--screen",
        action="store_true",
        help="Run automated EA configuration screening.",
    )

    parser.add_argument(
        "--screen-axis",
        choices=["mutation", "crossover", "selection", "all", "factorial"],
        default="mutation",
        help="Which axis to screen. 'all' runs mutation, crossover, and "
        "selection as three separate one-factor-at-a-time screens; "
        "'factorial' runs the full mutation x crossover grid.",
    )

    parser.add_argument(
        "--screen-name",
        default="screening",
        help="Output folder name under __data__/A1_evolution_tree.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for screening runs.",
    )

    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep each run's evolution.sqlite (deleted by default in "
        "screening to save disk; CSV outputs are always kept).",
    )

    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore finished runs found on disk (default: resume/reuse them).",
    )

    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=0,
        help="Stage 2: re-run the top-K stage-1 configurations on "
        "--refine-seeds and rank them statistically. 0 disables.",
    )

    parser.add_argument(
        "--refine-seeds",
        nargs="+",
        type=int,
        default=[],
        help="Extra seeds for the stage 2 finalists.",
    )

    parser.add_argument(
        "--screen-pop",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--screen-generations",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--screen-mutations",
        nargs="+",
        choices=VALID_MUTATIONS,
        default=[
            "replace",
            "subtree",
            "shrink",
            "hoist",
            "mixed",
        ],
    )

    parser.add_argument(
        "--screen-rates",
        nargs="+",
        type=float,
        default=[
            0.25,
            0.50,
            0.75,
            1.00,
        ],
    )

    parser.add_argument(
        "--screen-events",
        nargs="+",
        type=int,
        default=[
            1,
            2,
        ],
    )

    parser.add_argument(
        "--screen-seeds",
        nargs="+",
        type=int,
        default=[
            42,
            123,
            456,
        ],
    )

    parser.add_argument(
        "--screen-crossover",
        choices=VALID_CROSSOVERS,
        default="subtree",
    )

    parser.add_argument(
        "--screen-selection",
        choices=VALID_SELECTIONS,
        default="truncation",
    )

    parser.add_argument(
        "--screen-crossover-options",
        nargs="+",
        choices=VALID_CROSSOVERS,
        default=["none", "subtree"],
    )

    parser.add_argument(
        "--screen-crossover-rates",
        nargs="+",
        type=float,
        default=[0.3, 0.6, 1.0],
    )

    parser.add_argument(
        "--screen-selection-options",
        nargs="+",
        choices=VALID_SELECTIONS,
        default=["truncation", "tournament", "rank"],
    )

    parser.add_argument(
        "--screen-tournament-k",
        nargs="+",
        type=int,
        default=[2, 3, 5, 7],
    )

    parser.add_argument(
        "--screen-rank-pressure",
        nargs="+",
        type=float,
        default=[1.2, 1.5, 2.0],
    )

    parser.add_argument(
        "--random-search",
        action="store_true",
        help="Run a random-search baseline instead of an EA.",
    )

    parser.add_argument(
        "--random-search-budget",
        type=int,
        default=None,
        help="Total evaluations for the random search. Defaults to "
        "pop * (generations + 1), matching a single-run EA's budget.",
    )

    parser.add_argument(
        "--random-search-seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 1011],
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not 0.0 <= args.mutation_rate <= 1.0:
        raise ValueError("Mutation rate must be between 0 and 1.")

    if args.mutation_events < 1:
        raise ValueError("Mutation events must be >= 1.")

    if not 0.0 <= args.crossover_rate <= 1.0:
        raise ValueError("Crossover rate must be between 0 and 1.")

    if args.tournament_k < 2:
        raise ValueError("Tournament k must be >= 2.")

    if not 1.0 <= args.rank_pressure <= 2.0:
        raise ValueError("Rank pressure must be between 1.0 and 2.0.")

    if args.pop < 2:
        raise ValueError("Population size must be >= 2.")

    if args.generations < 1:
        raise ValueError("Generations must be >= 1.")

    if args.workers < 1:
        raise ValueError("Workers must be >= 1.")

    if args.refine_top_k < 0:
        raise ValueError("--refine-top-k must be >= 0.")

    if args.refine_top_k > 0 and not args.refine_seeds:
        raise ValueError("--refine-seeds is required when --refine-top-k > 0.")

    targets = load_targets()

    if args.final:
        run_final_experiment(targets, args)

        return

    if args.random_search:
        budget = (
            args.random_search_budget
            if args.random_search_budget is not None
            else args.pop * (args.generations + 1)
        )

        console.rule("Random Search Baseline")
        console.log(f"Budget (evaluations): {budget}")

        random_dir = DATA_DIR / "random_search"
        random_dir.mkdir(parents=True, exist_ok=True)

        all_random = []

        for seed in args.random_search_seeds:
            run_name = f"random_search_b{budget}_s{seed}"

            df = run_random_search(
                targets,
                budget,
                seed,
                args.max_modules,
                run_name=run_name,
            )

            all_random.append(df)

            console.log(f"seed={seed}: final best_so_far={df['best_so_far'].iloc[-1]:.4f}")

        combined = pd.concat(all_random, ignore_index=True)

        out_path = random_dir / f"random_search_budget{budget}.csv"
        combined.to_csv(out_path, index=False)

        console.log(f"Saved to {out_path}")

        return

    if args.screen:
        run_screening(
            targets,
            args,
        )

    else:
        run_single_experiment(
            targets,
            args,
        )


if __name__ == "__main__":
    main()
