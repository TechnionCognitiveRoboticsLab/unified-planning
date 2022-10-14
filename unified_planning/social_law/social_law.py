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
"""This module defines the social law class."""

from unified_planning.social_law.single_agent_projection import SingleAgentProjection
from unified_planning.social_law.robustness_verification import SimpleInstantaneousActionRobustnessVerifier
from unified_planning.social_law.waitfor_specification import WaitforSpecification
from unified_planning.social_law.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from unified_planning.model import Parameter, Fluent, InstantaneousAction
from unified_planning.shortcuts import *
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, InstantaneousAction, DurativeAction, Action
from typing import Type, List, Dict
from enum import Enum, auto
from unified_planning.io import PDDLWriter, PDDLReader
from unified_planning.engines import Credits
from unified_planning.model.multi_agent import *
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
import unified_planning.engines as engines
from unified_planning.plans.plan import Plan
import unified_planning.engines.results 
from unified_planning.engines.meta_engine import MetaEngine

credits = Credits('Social Law',
                  'Technion Cognitive Robotics Lab (cf. https://github.com/TechnionCognitiveRoboticsLab)',
                  'karpase@technion.ac.il',
                  'https://https://cogrob.net.technion.ac.il/',
                  'Apache License, Version 2.0',
                  'Represents a social law, which is a tranformation of a multi-agent problem + waitfor specification to a new multi-agent problem + waitfor.',
                  'Represents a social law, which is a tranformation of a multi-agent problem + waitfor specification to a new multi-agent problem + waitfor.')

class SocialLawRobustnessStatus(Enum):
    ROBUST_RATIONAL = auto() # Social law was proven to be robust
    NON_ROBUST_SINGLE_AGENT = auto() # Social law is not robust because one of the single agent projections is unsolvable
    NON_ROBUST_MULTI_AGENT_FAIL = auto() # Social law is not robust because the compilation achieves fail
    NON_ROBUST_MULTI_AGENT_DEADLOCK = auto() # Social law is not robust because the compilation achieves a deadlock



class SocialLawRobustnessChecker():
    '''social law robustness checker class:
    This class checks if a given MultiAgentProblemWithWaitfor is robust or not.
    '''
    def __init__(self, planner_name : str = 'fast-downward', robustness_verifier_name : str = None, save_pddl = False):
        self._planner_name = planner_name
        self._robustness_verifier_name = robustness_verifier_name
        self._save_pddl = save_pddl
        self._status = None
        self._counter_example = None

    @property
    def counter_example(self) -> Plan:
        return self._counter_example

    @property
    def status(self) -> SocialLawRobustnessStatus:
        return self._status

    def is_single_agent_solvable(self, problem : MultiAgentProblem) -> bool:
        for agent in problem.agents:
            sap = SingleAgentProjection(agent)        
            result = sap.compile(problem)

            with OneshotPlanner(name=self._planner_name, problem_kind=result.problem.kind) as planner:
                result = planner.solve(result.problem)
                if result.status not in unified_planning.engines.results.POSITIVE_OUTCOMES:
                    self._status = SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT
                    return False
        return True

    def is_multi_agent_robust(self, problem : MultiAgentProblemWithWaitfor) -> bool:
        self._counter_example = None
        
        rbv = Compiler(
            name = self._robustness_verifier_name,
            problem_kind = problem.kind, 
            compilation_kind=CompilationKind.MA_SL_ROBUSTNESS_VERIFICATION)
        rbv_result = rbv.compile(problem)

        if self._save_pddl:
            w = PDDLWriter(rbv_result.problem)
            w.write_domain("domain.pddl")
            w.write_problem("problem.pddl")            
        
        with OneshotPlanner(name=self._planner_name) as planner:
            result = planner.solve(rbv_result.problem)
            if result.status in unified_planning.engines.results.POSITIVE_OUTCOMES:
                self._counter_example = result.plan
                return False
        return True         


    def is_robust(self, problem : MultiAgentProblemWithWaitfor) -> SocialLawRobustnessStatus:
        self._status =  SocialLawRobustnessStatus.ROBUST_RATIONAL
        if not self.is_single_agent_solvable(problem):
            self._status = SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT
            return self._status            
        if not self.is_multi_agent_robust(problem):
            for action_occurence in self.counter_example.actions:
                parts = action_occurence.action.name.split("_")
                if parts[0] == "f":
                    self._status = SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL
                    break
                elif parts[0] == "w":
                    self._status = SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK            
                    break
            assert(self._status in [SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL, SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK])        
        return self._status
