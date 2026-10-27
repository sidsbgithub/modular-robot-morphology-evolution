"""Baseline EA using the Tree representation for Assignment 1."""

import random
from pathlib import Path
import time
import copy
import contextlib
import numpy as np

from ariel import console
from rich.progress import track
from ariel.ec import EA, EAOperation, EASettings, Individual, Population
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.operators import (
    random_tree,
    crossover_subtree,
    mutate_replace_node,
    mutate_subtree_replacement,
    mutate_shrink,
    mutate_hoist,
    _prune_invalid_edges,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.body_phenotypes.robogen_lite.config import ALLOWED_ROTATIONS, ModuleType, IDX_OF_CORE

# Import fitness function and target loading from the template
from A1_template_2026 import load_targets, fitness_function, show_body, NUM_OF_MODULES

# Reproducibility
SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)

# Data saving
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# EA configuration
POP_SIZE = 50
BUDGET = 50  # Generations
MAX_DEPTH = 12

class MorphologicalEA:
    def __init__(self, targets):
        self.targets = targets
        self.config = EASettings(
            is_maximisation=False,  # We want to minimize distance
            num_steps=BUDGET,
            target_population_size=POP_SIZE,
            output_folder=DATA,
            db_file_name="baseline_ea.db",
        )

    def create_individual(self) -> Individual:
        """Create a random tree genome."""
        while True:
            genome = random_tree(NUM_OF_MODULES)
            if len(genome.nodes) > 0:
                break
        ind = Individual()
        ind.genotype = genome.to_dict()
        ind.tags["ps"] = False
        ind.tags["valid"] = True
        return ind

    def mutate_tree(self, genome: TreeGenome) -> TreeGenome:
        """Apply various structural and parametric mutations."""
        new = copy.deepcopy(genome)
        
        # 1. Structural/Node mutations (standard GP)
        mutation_type = RNG.choice(
            ["point", "subtree", "shrink", "hoist"], p=[0.4, 0.4, 0.1, 0.1]
        )
        if mutation_type == "point":
            mutate_replace_node(new)
        elif mutation_type == "subtree":
            mutate_subtree_replacement(new, max_modules=NUM_OF_MODULES)
        elif mutation_type == "shrink":
            mutate_shrink(new)
        elif mutation_type == "hoist":
            mutate_hoist(new)

        # 2. Parametric mutation (rotation)
        if RNG.random() < 0.2:
            noncore = [nid for nid in new.nodes if nid != IDX_OF_CORE]
            if noncore:
                nid = random.choice(noncore)
                mtype = ModuleType[new.nodes[nid]["type"]]
                rots = [r.name for r in ALLOWED_ROTATIONS[mtype]]
                if rots:
                    new.nodes[nid]["rotation"] = random.choice(rots)

        _prune_invalid_edges(new)
        with contextlib.suppress(ValueError):
            validate_genome_dict(new.to_dict())
        return new

    def crossover_trees(self, parent1: Individual, parent2: Individual) -> TreeGenome:
        """Subtree crossover."""
        t1 = TreeGenome.from_dict(parent1.genotype)
        t2 = TreeGenome.from_dict(parent2.genotype)

        child1, child2 = crossover_subtree(t1, t2)
        # Return first valid child randomly
        chosen = child1 if RNG.random() < 0.5 else child2
        return chosen

    def reproduction(self, population: Population) -> Population:
        """Create offspring."""
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        if not parents:
            parents = population
        
        new_offspring = []
        target_pool = self.config.target_population_size * 2
        
        while len(population) + len(new_offspring) < target_pool:
            if len(parents) >= 2 and RNG.random() < 0.5:
                # 50% chance of Crossover
                p1, p2 = random.sample(parents, 2)
                child_morph = self.crossover_trees(p1, p2)
            else:
                # 50% chance of Mutation only
                parent = random.choice(parents)
                child_morph = TreeGenome.from_dict(parent.genotype)

            # Always mutate
            child_morph = self.mutate_tree(child_morph)

            # Ensure valid size/depth
            valid = False
            attempts = 0
            while not valid and attempts < 10:
                has_modules = len(child_morph.nodes) > 0
                valid_depth = validate_tree_depth(child_morph, MAX_DEPTH)
                if has_modules and valid_depth:
                    valid = True
                else:
                    child_morph = self.mutate_tree(child_morph)
                attempts += 1

            ind = Individual()
            ind.genotype = child_morph.to_dict()
            ind.tags["ps"] = False
            ind.tags["valid"] = True
            new_offspring.append(ind)
            
        population.extend(new_offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        """Evaluate fitness for all individuals requiring it."""
        to_eval = [
            ind for ind in population 
            if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]
        
        if not to_eval:
            return population

        for ind in track(to_eval, description="Evaluating Fitness"):
            genome = TreeGenome.from_dict(ind.genotype)
            body = genome.to_networkx()
            if body.number_of_nodes() == 0:
                ind.fitness = float("inf")
            else:
                try:
                    ind.fitness = fitness_function(body, self.targets)
                except Exception:
                    ind.fitness = float("inf")
            ind.requires_eval = False
            
        return population

    def parent_selection(self, population: Population) -> Population:
        """Tournament selection."""
        for ind in population:
            ind.tags["ps"] = False
            
        alive = [ind for ind in population if ind.alive]
        k = 3 # Tournament size
        selected = []
        
        # Select target_population_size parents
        for _ in range(self.config.target_population_size):
            tournament = random.sample(alive, k)
            winner = min(tournament, key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf"))
            selected.append(winner)
            
        for ind in selected:
            ind.tags["ps"] = True
            
        return population

    def survivor_selection(self, population: Population) -> Population:
        """Elitism: Keep the best N individuals."""
        population = population.sort(sort="min", attribute="fitness_")
        survivors = population[:self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False
                
        # Log generation stats
        avg_fit = np.mean([i.fitness_ for i in survivors if i.fitness_ is not None])
        min_fit = min([i.fitness_ for i in survivors if i.fitness_ is not None])
        console.log(f"[green]Survivors:[/green] Avg Fitness = {avg_fit:.4f}, Best Fitness = {min_fit:.4f}")
        return population

    def run(self) -> Individual:
        console.log("Initializing population...")
        population = Population([self.create_individual() for _ in range(POP_SIZE)])
        population = self.evaluate(population)

        ops = [
            EAOperation(self.parent_selection),
            EAOperation(self.reproduction),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),
        ]

        ea = EA(
            population,
            operations=ops,
            num_steps=self.config.num_steps,
            is_maximisation=self.config.is_maximisation,
            db_file_path=self.config.db_file_path,
            db_handling=self.config.db_handling,
            quiet=self.config.quiet,
        )
        
        start = time.time()
        ea.run()
        elapsed = time.time() - start
        console.log(f"Evolution finished in {elapsed:.2f}s")
        
        return ea.get_solution("best", only_alive=False)

def main():
    console.rule("[bold cyan]Running Baseline EA[/bold cyan]")
    targets = load_targets()
    console.log(f"Loaded {len(targets)} targets.")
    
    ea = MorphologicalEA(targets)
    best_ind = ea.run()
    
    if best_ind:
        best_genome = TreeGenome.from_dict(best_ind.genotype)
        best_body = best_genome.to_networkx()
        fitness = fitness_function(best_body, targets)
        
        console.rule("[bold green]Best Body Found[/bold green]")
        console.log(f"Best Fitness: {fitness:.4f}")
        console.log(f"Number of Modules: {best_body.number_of_nodes()}")
        
        # Save frame
        show_body(best_body, mode="frame", file_name="baseline_best")
        # To launch viewer automatically, change mode="launcher"

if __name__ == "__main__":
    main()
