# Copyright 2022 Technion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""This module defines the social law synthesis functionality."""

from collections import defaultdict, deque
import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.social_law.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from unified_planning.social_law.social_law import SocialLaw
from unified_planning.social_law.robustness_checker import SocialLawRobustnessChecker, SocialLawRobustnessResult, SocialLawRobustnessStatus
from unified_planning.model import Parameter, Fluent, InstantaneousAction, problem_kind
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, InstantaneousAction, DurativeAction, Action
from typing import Type, List, Dict, Callable, OrderedDict, Set
from enum import Enum, auto
from unified_planning.io import PDDLWriter, PDDLReader
from unified_planning.engines import Credits
from unified_planning.model.multi_agent import *
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
import unified_planning.engines as engines
from unified_planning.plans import Plan, SequentialPlan
import unified_planning.engines.results 
from unified_planning.engines.meta_engine import MetaEngine
import unified_planning.engines.mixins as mixins
from unified_planning.engines.mixins.oneshot_planner import OptimalityGuarantee
from unified_planning.engines.results import *
from unified_planning.engines.sequential_simulator import SequentialSimulator
from unified_planning.model.multi_agent.ma_centralizer import MultiAgentProblemCentralizer
from functools import partial
from unified_planning.engines.compilers.utils import replace_action
import queue
from dataclasses import dataclass, field
from typing import Any
from itertools import chain, combinations
from unified_planning.model.walkers import FreeVarsExtractor

credits = Credits('Social Law Synthesis',
                  'Technion Cognitive Robotics Lab (cf. https://github.com/TechnionCognitiveRoboticsLab)',
                  'karpase@technion.ac.il',
                  'https://https://cogrob.net.technion.ac.il/',
                  'Apache License, Version 2.0',
                  'Provides the ability to automatically generate a robust social law.',
                  'Provides the ability to automatically generate a robust social law.')

class SocialLawGeneratorSearch(Enum):
    BFS = auto()
    DFS = auto()
    GBFS = auto()


def get_gbfs_social_law_generator():
    return SocialLawGenerator(SocialLawGeneratorSearch.GBFS, 
                            heuristic = StatisticsHeuristic(), 
                            preferred_operator_heuristics = [EarlyPOHeuristic(), PublicActionsPOHeuristic()])

@dataclass(order=True)
class SearchNode:
    """ This class represents a node in the search for a robust social law."""

    priority: int
    social_law : SocialLaw = field(compare=False)

    def __init__(self, social_law : SocialLaw, priority : int = 0):
        self.priority = priority
        self.social_law = social_law

class Heuristic:
    def __init__(self):
        pass

    def get_priority(self, node : SearchNode):
        raise NotImplementedError()

    def report_current_problem(self, problem : MultiAgentProblemWithWaitfor):
        pass

    def report_current_node(self, node : SearchNode, robustness_result : SocialLawRobustnessResult):
        pass

class StatisticsHeuristic(Heuristic):
    def __init__(self, before_fail_weight = 2, after_fail_weight = 1):
        Heuristic.__init__(self)
        self.before_fail_weight = before_fail_weight
        self.after_fail_weight = after_fail_weight
        self.action_count_map = {}

    def get_priority(self, node : SearchNode) -> int:
        h = 0
        for agent_name, action_name, args in node.social_law.disallowed_actions:
            h = h + self.action_count_map[(agent_name, action_name, args)]
        return -h
                            
    def report_current_node(self, node : SearchNode, robustness_result : SocialLawRobustnessResult):
        if robustness_result.counter_example is not None:
            before_fail = True
            for i, ai in enumerate(robustness_result.counter_example_orig_actions.actions):
                compiled_action_instance = robustness_result.counter_example.actions[i]
                parts = compiled_action_instance.action.name.split("_")
                agent_name = parts[1]                                                    
                if parts[0][0] in ["w","f"]:
                    before_fail = False

                args_as_str = tuple(map(str, ai.actual_parameters))
                if (agent_name, ai.action.name, args_as_str) not in self.action_count_map:
                    self.action_count_map[agent_name, ai.action.name, args_as_str] = 0    
                if before_fail:
                    self.action_count_map[agent_name, ai.action.name, args_as_str] = self.action_count_map[agent_name, ai.action.name, args_as_str] + self.before_fail_weight
                else:
                    self.action_count_map[agent_name, ai.action.name, args_as_str] =self. action_count_map[agent_name, ai.action.name, args_as_str] + self.after_fail_weight

class EarlyPOHeuristic(Heuristic):
    def __init__(self):
        Heuristic.__init__(self)
        self.early_actions = set()

    def get_priority(self, node : SearchNode) -> int:
        if node.social_law.disallowed_actions.issubset(self.early_actions):
            # 0 Indicates preferred
            return 0
        else:
            # 1 Indicates not preferred
            return 1
                            
    def report_current_node(self, node : SearchNode, robustness_result : SocialLawRobustnessResult):
        self.early_actions.clear()
        
        if robustness_result.counter_example is not None:            
            for i, ai in enumerate(robustness_result.counter_example_orig_actions.actions):
                compiled_action_instance = robustness_result.counter_example.actions[i]
                parts = compiled_action_instance.action.name.split("_")
                agent_name = parts[1]                                                    
                args_as_str = tuple(map(str, ai.actual_parameters))
                if parts[0][0] in ["w","f"]:
                    break
                self.early_actions.add( (agent_name, ai.action.name, args_as_str) )

class PublicActionsPOHeuristic(Heuristic):
    def __init__(self):
        Heuristic.__init__(self)
        self.public_actions = set()

    def get_priority(self, node : SearchNode) -> int:
        for agent_name, action_name, _ in node.social_law.disallowed_actions:
            if (agent_name, action_name) not in self.public_actions:
                # 1 Indicates not preferred
                return 1
        # 0 Indicates preferred
        return 0

    def report_current_problem(self, problem: MultiAgentProblemWithWaitfor):
        fve = FreeVarsExtractor()
        for agent in problem.agents:
            for action in agent.actions:
                public = False
                for precondition in action.preconditions:
                    fluents = fve.get(precondition)
                    for f in fluents:
                        if f.fluent() in problem.ma_environment.fluents:
                            public = True
                            break
                for effect in action.effects:
                    if effect.fluent.fluent() in problem.ma_environment.fluents:
                        public = True
                        break
                if public:
                    self.public_actions.add( (agent.name, action.name) )




class POQueue:
    def get_single_queue(self):
        if self.search == SocialLawGeneratorSearch.BFS:
            return queue.Queue()
        elif self.search == SocialLawGeneratorSearch.DFS:
            return queue.LifoQueue()
        elif self.search == SocialLawGeneratorSearch.GBFS:
            return queue.PriorityQueue()

    def __init__(self, num_po_heuristics = 0, search : SocialLawGeneratorSearch = SocialLawGeneratorSearch.BFS):
        self.search = search        
        self.queues = [self.get_single_queue()] * (2 ** num_po_heuristics)
        self.current_queue = 0

    def empty(self) -> bool:
        for queue in self.queues:
            if not queue.empty():
                return False
        return True            

    def get(self):       
        self.current_queue = (self.current_queue + 1) % len(self.queues)
        while self.queues[self.current_queue].empty():
            self.current_queue = (self.current_queue + 1) % len(self.queues)
        return self.queues[self.current_queue].get()

    def put(self, node : SearchNode, pref_profile : List[int]):
        preferred_queues = []
        for queue_index in range(len(pref_profile)):
            if pref_profile[queue_index] == 0:
                preferred_queues.append(queue_index)        

        for prefs in chain.from_iterable(combinations(preferred_queues, r) for r in range(len(preferred_queues)+1)):
            queue_index = 0
            for pref_index in prefs:
                queue_index = queue_index + (2 ** pref_index)
            self.queues[queue_index].put(node)
            
class SocialLawGenerator:
    """ This class takes in a multi agent problem (possibly with social laws), and searches for a social law which will turn it robust."""
    def __init__(self, 
                    search : SocialLawGeneratorSearch = SocialLawGeneratorSearch.BFS, 
                    heuristic : Optional[Heuristic] = None,
                    preferred_operator_heuristics : List[Heuristic] = []):
        self.search = search
        self.heuristic = heuristic
        self.po = preferred_operator_heuristics
        self.all_heuristics = set(preferred_operator_heuristics)
        if self.heuristic is not None:
            self.all_heuristics.add(self.heuristic)

    
    def init_counters(self):
        self.generated = 0
        self.expanded = 0

    def generate_successors(self, current_sl : SocialLaw, action_index_in_plan : int, original_action_instance : ActionInstance, compiled_action_instance : ActionInstance):
        parts = compiled_action_instance.action.name.split("_")
        agent_name = parts[1]
        action_name = original_action_instance.action.name

        # Generate a successor which disallows this action
        succ_sl = current_sl.clone()
        succ_sl.disallow_action(agent_name, action_name, tuple(map(str, original_action_instance.actual_parameters)))

        # TODO: generate other possible successors (add preconditions, ...)

        return [succ_sl]


    def generate_social_law(self, initial_problem : MultiAgentProblemWithWaitfor):
        robustness_checker = SocialLawRobustnessChecker()        
        self.init_counters()

        for h in self.all_heuristics:
            h.report_current_problem(initial_problem)

        open = POQueue(len(self.po), self.search)
        closed : Set[SocialLaw] = set()
        infeasible_sap : Set[SocialLaw] = set()

        empty_social_law = SocialLaw()
        open.put( SearchNode(empty_social_law), [1] * len(self.po) )
        self.generated = self.generated + 1

        while not open.empty():
            current_node = open.get()
            current_sl = current_node.social_law
            if current_sl not in closed:
                closed.add(current_sl)
                self.expanded = self.expanded + 1
                
                # Check that this isn't stricter than a social law for while the single agent projection is not solvable
                for infeasible_sl in infeasible_sap:
                    if current_sl.is_stricter_than(infeasible_sl):
                        continue

                current_problem = current_sl.compile(initial_problem).problem
                robustness_result = robustness_checker.is_robust(current_problem)
                for h in self.all_heuristics:
                    h.report_current_node(current_node, robustness_result)

                if robustness_result.status == SocialLawRobustnessStatus.ROBUST_RATIONAL:
                    # We found a robust social law - return
                    return current_node.social_law
                elif robustness_result.status == SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT:
                    # We made one of the single agent problems unsolvable - this is a dead end (for this simple search)
                    infeasible_sap.add(current_sl)                    
                else:
                    # We have a counter example, generate a successor for removing each of the actions that appears there                    
                    for i, ai in enumerate(robustness_result.counter_example_orig_actions.actions):
                        compiled_action_instance = robustness_result.counter_example.actions[i]
                        for succ_sl in self.generate_successors(current_sl, i, ai, compiled_action_instance):
                            succ_node = SearchNode(succ_sl)
                            
                            pref = list(map(lambda poh: poh.get_priority(succ_node), self.po))
                            
                            if self.heuristic is not None:
                                succ_node.priority = self.heuristic.get_priority(succ_node)
                            open.put(succ_node, pref)
                            self.generated = self.generated + 1

                        



    
