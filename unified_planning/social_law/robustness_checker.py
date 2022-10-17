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

from collections import defaultdict
import unified_planning as up
from unified_planning.social_law.single_agent_projection import SingleAgentProjection
from unified_planning.social_law.robustness_verification import SimpleInstantaneousActionRobustnessVerifier
from unified_planning.social_law.waitfor_specification import WaitforSpecification
from unified_planning.social_law.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from unified_planning.model import Parameter, Fluent, InstantaneousAction, problem_kind
from unified_planning.shortcuts import *
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

credits = Credits('Social Law Robustness Checker',
                  'Technion Cognitive Robotics Lab (cf. https://github.com/TechnionCognitiveRoboticsLab)',
                  'karpase@technion.ac.il',
                  'https://https://cogrob.net.technion.ac.il/',
                  'Apache License, Version 2.0',
                  'Does all the robustness checks, and returns the reason for failure.',
                  'Does all the robustness checks, and returns the reason for failure.')

class SocialLawRobustnessStatus(Enum):
    ROBUST_RATIONAL = auto() # Social law was proven to be robust
    NON_ROBUST_SINGLE_AGENT = auto() # Social law is not robust because one of the single agent projections is unsolvable
    NON_ROBUST_MULTI_AGENT_FAIL = auto() # Social law is not robust because the compilation achieves fail
    NON_ROBUST_MULTI_AGENT_DEADLOCK = auto() # Social law is not robust because the compilation achieves a deadlock
    UNKNOWN = auto() # unknown, for example if the planner failed to prove unsolvability

class SocialLawRobustnessResult(Result):
    status : SocialLawRobustnessStatus
    counter_example : Optional["up.plans.Plan"]
    counter_example_orig_actions : Optional["up.plans.Plan"]

    def __init__(self, 
                status : SocialLawRobustnessStatus, 
                counter_example : Optional["up.plans.Plan"], 
                counter_example_orig_actions : Optional["up.plans.Plan"]):
        self.status = status
        self.counter_example = counter_example
        self.counter_example_orig_actions = counter_example_orig_actions
    


class SocialLawRobustnessChecker(engines.engine.Engine, mixins.OneshotPlannerMixin):
    '''social law robustness checker class:
    This class checks if a given MultiAgentProblemWithWaitfor is robust or not.
    '''
    def __init__(self, planner_name : str = None, robustness_verifier_name : str = None, save_pddl = False):
        engines.engine.Engine.__init__(self)
        mixins.OneshotPlannerMixin.__init__(self)
        self._planner_name = planner_name
        self._robustness_verifier_name = robustness_verifier_name
        self._save_pddl = save_pddl
        

    @property
    def name(self) -> str:
        return f"SocialLawRobustnessChecker[{self._planner_name}]"

    @staticmethod
    def get_credits(**kwargs) -> Optional['Credits']:
        return credits

    @staticmethod
    def satisfies(optimality_guarantee: OptimalityGuarantee) -> bool:
        if optimality_guarantee == OptimalityGuarantee.SATISFICING:
            return True
        return False

    @staticmethod
    def supported_kind() -> "ProblemKind":
        supported_kind = ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED_MULTI_AGENT")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_numbers("CONTINUOUS_NUMBERS")
        supported_kind.set_numbers("DISCRETE_NUMBERS")
        supported_kind.set_fluents_type("NUMERIC_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITY")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_time("CONTINUOUS_TIME")
        supported_kind.set_time("DISCRETE_TIME")
        supported_kind.set_time("INTERMEDIATE_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("TIMED_EFFECT")
        supported_kind.set_time("TIMED_GOALS")
        supported_kind.set_time("DURATION_INEQUALITIES")
        supported_kind.set_expression_duration("STATIC_FLUENTS_IN_DURATION")
        supported_kind.set_expression_duration("FLUENTS_IN_DURATION")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        final_supported_kind = supported_kind.intersection(SingleAgentProjection.supported_kind())
        
        return final_supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= SocialLawRobustnessChecker.supported_kind()

    @property
    def status(self) -> SocialLawRobustnessStatus:
        return self._status

    def is_single_agent_solvable(self, problem : MultiAgentProblem) -> bool:
        for agent in problem.agents:
            sap = SingleAgentProjection(agent)        
            result = sap.compile(problem)

            with OneshotPlanner(name=self._planner_name, problem_kind=result.problem.kind) as planner:
                presult = planner.solve(result.problem)
                if presult.status not in unified_planning.engines.results.POSITIVE_OUTCOMES:                    
                    return False
        return True

    def multi_agent_robustness_counterexample(self, problem : MultiAgentProblemWithWaitfor) -> SocialLawRobustnessResult:
        rbv = Compiler(
            name = self._robustness_verifier_name,
            problem_kind = problem.kind, 
            compilation_kind=CompilationKind.MA_SL_ROBUSTNESS_VERIFICATION)
        rbv_result = rbv.compile(problem)

        if self._save_pddl:
            w = PDDLWriter(rbv_result.problem)
            w.write_domain("domain.pddl")
            w.write_problem("problem.pddl")            
        
        with OneshotPlanner(name=self._planner_name, problem_kind=rbv_result.problem.kind) as planner:
            result = planner.solve(rbv_result.problem)
            if result.status in unified_planning.engines.results.POSITIVE_OUTCOMES:                
                for action_occurence in result.plan.actions:
                    parts = action_occurence.action.name.split("_")
                    if parts[0][0] == "f":
                        status = SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL
                        break
                    elif parts[0][0] == "w":
                        status = SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK            
                        break

                orig_plan = result.plan.replace_action_instances(rbv_result.map_back_action_instance)

                return SocialLawRobustnessResult(status, result.plan, orig_plan)
            elif result.status in [PlanGenerationResultStatus.UNSOLVABLE_PROVEN, PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY]:
                return SocialLawRobustnessResult(SocialLawRobustnessStatus.ROBUST_RATIONAL, None, None)
            else:
                return SocialLawRobustnessResult(SocialLawRobustnessStatus.UNKNOWN, None, None)


    def is_robust(self, problem : MultiAgentProblemWithWaitfor) -> SocialLawRobustnessResult:
        status =  SocialLawRobustnessStatus.ROBUST_RATIONAL
        # Check single agent solvability
        if not self.is_single_agent_solvable(problem):
            return SocialLawRobustnessResult(SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT, None, None)
        
        # Check for rational robustness
        result = self.multi_agent_robustness_counterexample(problem)        
        if result.counter_example is not None:            
            assert(result.status in [SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL, SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK])
        return result

    def _solve(self, problem: 'up.model.AbstractProblem',
            callback: Optional[Callable[['up.engines.results.PlanGenerationResult'], None]] = None,
            timeout: Optional[float] = None,
            output_stream: Optional[IO[str]] = None) -> 'up.engines.results.PlanGenerationResult':
        assert isinstance(problem, MultiAgentProblemWithWaitfor)
        plans = {}
        current_step = {}
        for agent in problem.agents:
            sap = SingleAgentProjection(agent)        
            result = sap.compile(problem)

            with OneshotPlanner(name=self._planner_name, problem_kind=result.problem.kind) as planner:
                presult = planner.solve(result.problem, timeout=timeout, output_stream=output_stream)
                if presult.status not in unified_planning.engines.results.POSITIVE_OUTCOMES:
                    return unified_planning.engines.results.PlanGenerationResult(
                        unified_planning.engines.results.UNSOLVABLE_INCOMPLETELY,
                        plan=None,
                        engine_name = self.name)
                plans[agent] = presult.plan
                current_step[agent] = 0

        mac = MultiAgentProblemCentralizer()
        cresult = mac.compile(problem)        
        simulator = SequentialSimulator(cresult.problem)
        current_state: "COWState" = UPCOWState(cresult.problem.initial_values)

        plan = SequentialPlan([])

        active_agents = problem.agents.copy()
        active_agents_next = []
                
        while len(active_agents) > 0:
            action_performed = False
            for agent in active_agents:
                if current_step[agent] < len(plans[agent].actions):
                    active_agents_next.append(agent)
                    ai = plans[agent].actions[current_step[agent]]
                    action = cresult.problem.action(agent.name + "__" + ai.action.name)
                    assert isinstance(action, unified_planning.model.InstantaneousAction)

                    applicable = True
                    events = simulator.get_events(action, ai.actual_parameters)
                    for event in events:
                        if not simulator.is_applicable(event, current_state):
                            applicable = False
                            break
                    
                    if applicable:
                        plan.actions.append(ActionInstance(ai.action, ai.actual_parameters, agent))
                        action_performed = True
                        for event in events:                
                            current_state = simulator.apply_unsafe(event, current_state)
                        current_step[agent] = current_step[agent] + 1
            if not action_performed and len(active_agents_next) > 0:
                # deadlock occurred
                return PlanGenerationResult(unified_planning.engines.results.PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
                                            plan=None,
                                            engine_name = self.name)
            active_agents = active_agents_next.copy()
            active_agents_next = []

        unsatisfied_goals = simulator.get_unsatisfied_goals(current_state)
        if len(unsatisfied_goals) == 0:            
            return unified_planning.engines.results.PlanGenerationResult(
                unified_planning.engines.results.PlanGenerationResultStatus.SOLVED_SATISFICING,
                plan=plan,
                engine_name = self.name      
            )
        else:
            # Goal not achieved at the end
            return PlanGenerationResult(unified_planning.engines.results.PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
                                            plan=None,
                                            engine_name = self.name)
        