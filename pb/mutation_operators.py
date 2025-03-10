import os
import re
import json
import copy
import random
from rich import print
from typing import List
from pathlib import Path

# from sentence_transformers import SentenceTransformer, util

from pb import gsm
from pb.types import Population, EvolutionUnit
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

from . import ModelWrapper

import ipdb
from dotenv import load_dotenv

load_dotenv()

EVALUATION_DATA_DIR = os.getenv("EVALUATION_DATA_DIR")

# gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')

# need below for estimation_distribution_mutation, not currently using.
# model = SentenceTransformer('multi-qa-distilbert-cos-v1')
# print(model) 

# Direct Mutation mutators
def zero_order_prompt_gen(unit: EvolutionUnit, problem_description: str, model: ModelWrapper, **kwargs) -> EvolutionUnit:
    """Generates a new task-prompt P by concatenating the problem description D with the prompt 
    'a list of 100 hints:'. New task-prompt P is the first generated hint.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    result = model.generate(problem_description + " An ordered list of 100 hints: ")
    # search for the pattern "anything after 1. and before 2."
    pattern = r"1\.(.*?)2\."
    match = re.search(pattern, result, re.DOTALL)
    if match: 
        # return the first match
        unit.P = match.group(1).strip()
    else: 
        unit.P = ""
    
    return unit 

def first_order_prompt_gen(unit: EvolutionUnit, problem_description: str, model: ModelWrapper, **kwargs) -> EvolutionUnit:
    """Concatenate the mutation prompt M to the parent task-prompt P and pass it to the LLM to produce P'
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    unit.P = model.generate(unit.M + " " + unit.P)
    # unit.P = model.generate(unit.M + " " + problem_description)
    return unit
    
# Estimation of Distribution Mutation - there is a variation of this called EDA rank
# and index mutation. I didn't implement it.
def estimation_distribution_mutation(unit: EvolutionUnit, population_units: List[EvolutionUnit], **kwargs) -> EvolutionUnit:
    """ Provide a filtered and numbered list of the current population of task-prompts to the LLM and ask it to continue this list with new task-prompts.
    The List is filtered via ensuring that no two task-prompts have a score of >0.95 via BERT embedding cosine similarities.
    The List is randomly ordered.  

    NOTE: I am confused by this one. Does this mutate the entire population? What values of the continued list from the LLM do I use as prompts? randomly sampled?
    Not going to implement this one yet. Maybe should email someone. 
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    pass

def eda_rank_based_mutation(unit: EvolutionUnit, sorted_elites: List[EvolutionUnit], model: ModelWrapper, **kwargs) -> EvolutionUnit:
    """Using the stored history of best units, provide the LLM this list in chronological order to produce a novel prompt as continuation.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    HEADING = f"INSTRUCTION: {unit.M}\n A List of Responses in descending order of score. ({len(sorted_elites)+1}) is the best response. It resembles ({len(sorted_elites)}) more than it does (1)."
    ITEMS = "\n".join(["({}) {}".format(i+1, x.P) for i, x in enumerate(sorted_elites)])
    unit.P = model.generate(HEADING + ITEMS)
    return unit

def lineage_based_mutation(unit: EvolutionUnit, elites: List[EvolutionUnit], model: ModelWrapper, **kwargs) -> EvolutionUnit:
    """Using the stored history of best units, provide the LLM this list in chronological order to produce a novel prompt as continuation.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    HEADING = "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY \n "
    # made a choice not to format it with newlines, could change later.
    ITEMS = "\n".join(["{}. {}".format(i+1, x.P) for i, x in enumerate(elites)])
    unit.P = model.generate(HEADING + ITEMS)
    return unit

# Hypermutation
def zero_order_hypermutation(unit: EvolutionUnit, problem_description: str, model: ModelWrapper, **kwargs) -> EvolutionUnit:
    """ Concatenate the original problem_description to a randomly sampled thinking-style and feed it to the LLM to generate a new mutation-prompt.
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    RANDOM_THINKING_STYLE = random.sample(thinking_styles, 1)[0]
    unit.M = model.generate(problem_description + " " + RANDOM_THINKING_STYLE)
    unit.P = model.generate(unit.M + " " + unit.P)
    return unit

def first_order_hypermutation(unit: EvolutionUnit, model: ModelWrapper, **kwargs) -> EvolutionUnit:
    """ Concatenate the hyper-mutation prompt "Please summarize and improve the following instruction:"
    to a mutation-prompt to that the LLM generates a new mutation-prompt. This new mutation-prompt is then 
    instantly applied to the task-prompt of that unit.

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    HYPER_MUTATION_PROMPT = "Please summarize and improve the following instruction: "
    unit.M = model.generate(HYPER_MUTATION_PROMPT + unit.M)
    unit.P = model.generate(unit.M + " " + unit.P)
    return unit 

# Lamarckian Mutation
def working_out_task_prompt(unit: EvolutionUnit, model: ModelWrapper, target_model: str, dataset_name: str, **kwargs) -> EvolutionUnit:
    """ A 'lamarckian' mutation operator similar to instruction induction in APE.

    As far as I can understand, give it both the Q and A from the gsm8k dataset, 
    concatenated between 'I gave a friend an instruction and some advice. Here
    are the correct examples of his workings out ' and 'The instruction was: '
    The idea is to let the LLM reverse-engineer the task-prompt.

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    # TODO Update this to use workings out from mutations
    dataset_path = Path(EVALUATION_DATA_DIR) / target_model / f"{dataset_name}.json"
    with open(dataset_path, "r") as fp:
        dataset = json.load(fp)

    correct_samples = []
    for _, sample in dataset.items():
        if sample["sample_details"]["source"] not in ("xsum",):
            if sample["result"]["score"] == 1.:
                correct_samples.append((sample["eval_details"]["prompt"], sample["eval_details"]["origin_prediction"]))
        else:
            correct_samples.append((sample["eval_details"]["prompt"], sample["eval_details"]["references"]))

    random_samples = random.sample(correct_samples, 2)

    HEADING = f"I gave a friend an instruction and some advice. Here are the correct examples of his workings out\n\n"
    ENDING = "The instruction was: "
    examples = ""
    for sample in random_samples:
        prompt, answer = sample
        examples += f"Q: {prompt}\nA: {answer}.\n\n"

    unit.P = model.generate(HEADING + examples + ENDING)
    return unit 

# Prompt crossover and context shuffling. These happen AFTER mutation operators. 
def prompt_crossover(unit: EvolutionUnit, crossover_unit: EvolutionUnit, **kwargs):
    """
    After a mutation operator is applied, 

    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """
    if random.randint(1, 10) == 1:
        unit.P = copy.deepcopy(crossover_unit.P)

def context_shuffling(**kwargs):
    """
    
    Returns: 
        EvolutionUnit: the evolution unit to replace the loser unit.
    """

# omitting the estimation_distribution_mutation
MUTATORS = [
    zero_order_prompt_gen,
    first_order_prompt_gen,
    eda_rank_based_mutation,
    #estimation_distribution_mutation,
    lineage_based_mutation,
    zero_order_hypermutation,
    first_order_hypermutation,
    working_out_task_prompt,
]

POST_MUTATORS = [
    prompt_crossover,
    # context_shuffling
]

def mutate(population: Population, model: ModelWrapper, target_model: str, dataset_name: str) -> Population:
    """Select and apply a random mutator"""
    # steps
    # 1. parse through the population, grouping each evo unit by 2
    # 2. for each pair of evo units, using a uniform distribution, select a random mutator (of the 9)
    # 3. mutate and populate population.units

    # make index pairs
    indices = [i for i in range(len(population.units))]
    random.shuffle(indices)
    pairs = [indices[2*x:2*x+2] for x in range(len(indices) // 2)]

    # binary tourmanent genetic algorithm
    for i in range(len(pairs)):

        first_unit = population.units[pairs[i][0]]
        second_unit = population.units[pairs[i][1]]

        print("%"*77)
        print("First unit: \n")
        print(first_unit)
        print("%"*77)
        print("Second unit: \n")
        print(second_unit)

        # determine which unit has the higher fitness. Since I am currently testing and want to preserve the # of calls I am making to the LLM, there 
        # is a decent chance that I will hit equal fitness levels. in that case, first unit wins and second unit loses.
        
        if first_unit.fitness >=  second_unit.fitness:
            # loser gets mutated.
            mutation_input = second_unit
        else:
            mutation_input = first_unit

        sorted_elites = sorted(population.elites, key=lambda x: x[0])
        sorted_elite_units = [e[1] for e in sorted_elites]

        elites = [e[1] for e in population.elites]

        mutation_data = {
            'unit' : mutation_input,
            'model' : model,
            'sorted_elites': sorted_elite_units,
            'elites' : elites,
            'problem_description': population.problem_description,
            'dataset_name': dataset_name,
            'target_model': target_model
        }

        # uniformly pick and call a random mutation operator on the losing unit
        random_mutator = random.sample(MUTATORS, 1)[0]
        print(f"MUTATING: {mutation_input} with {random_mutator.__name__}")
        random_mutator(**mutation_data)

        crossover_weights = [unit.fitness + 1e-6 for unit in population.units]
        crossover_unit = random.choices(population.units, weights=crossover_weights, k=1)[0]

        post_mutation_data = {
            'unit' : mutation_input,
            'crossover_unit' : crossover_unit,
        }

        random_post_mutator = random.sample(POST_MUTATORS, 1)[0]
        print(f"CROSS MUTATING: {mutation_input} with {random_post_mutator.__name__}")
        random_post_mutator(**post_mutation_data)

    return population
