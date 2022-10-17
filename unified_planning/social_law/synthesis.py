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

from collections import defaultdict
import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.social_law.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from unified_planning.social_law.social_law import SocialLaw
from unified_planning.social_law.robustness_checker import SocialLawRobustnessChecker, SocialLawRobustnessStatus
from unified_planning.model import Parameter, Fluent, InstantaneousAction, problem_kind
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, InstantaneousAction, DurativeAction, Action
from typing import Type, List, Dict, Callable, OrderedDict
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

credits = Credits('Social Law Synthesis',
                  'Technion Cognitive Robotics Lab (cf. https://github.com/TechnionCognitiveRoboticsLab)',
                  'karpase@technion.ac.il',
                  'https://https://cogrob.net.technion.ac.il/',
                  'Apache License, Version 2.0',
                  'Provides the ability to automatically generate a robust social law.',
                  'Provides the ability to automatically generate a robust social law.')


class SearchNode:
    """ This class represents a node in the search for a robust social law."""

    def __init__(self, sl : SocialLaw):
        self.sl = sl

    @property
    def social_law(self) -> SocialLaw:
        return self.sl

class Queue:
    """ This class represents a queue for the open list"""

    def __init__(self):
        self.items : List[SearchNode] = []

    def push(self, n : SearchNode):
        self.items.append(n)
    
    def pop(self) -> SearchNode:
        return self.items.pop()

    def empty(self) -> bool:
        return len(self.items) == 0


class SocialLawGenerator:
    """ This class takes in a multi agent problem (possibly with social laws), and searches for a social law which will turn it robust."""
    def __init__(self, initial_problem : MultiAgentProblemWithWaitfor):
        self._initial_problem = initial_problem
        self.robustness_checker = SocialLawRobustnessChecker()

    @property
    def initial_problem(self) -> MultiAgentProblemWithWaitfor:
        return self._initial_problem

    def generate_social_law(self):
        open = Queue()
        closed = set()
        empty_social_law = SocialLaw()
        open.push( SearchNode(empty_social_law) )

        while not open.empty():
            current_node = open.pop()
            current_sl = current_node.social_law
            if current_sl not in closed:
                closed.add(current_sl)
                
                current_problem = current_sl.compile(self.initial_problem).problem
                robustness_result = self.robustness_checker.is_robust(current_problem)

                if robustness_result.status == SocialLawRobustnessStatus.ROBUST_RATIONAL:
                    # We found a robust social law - return
                    return current_node
                elif robustness_result.status == SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT:
                    # We made one of the single agent problems unsolvable - this is a dead end (for this simple search)
                    continue
                else:
                    # We have a counter example, generate a successor for removing each of the actions that appears there
                    for i, ai in enumerate(robustness_result.counter_example_orig_actions.actions):
                        compiled_action_instance = robustness_result.counter_example.actions[i]
                        parts = compiled_action_instance.action.name.split("_")
                        agent_name = parts[1]
                        action_name = ai.action.name

                        succ_sl = current_sl.clone()
                        succ_sl.disallow_action(agent_name, action_name, tuple(map(str, ai.actual_parameters)))                        
                        open.push(SearchNode(succ_sl))



    
