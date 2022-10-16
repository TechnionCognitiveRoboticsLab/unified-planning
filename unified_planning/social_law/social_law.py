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

import unified_planning as up
from unified_planning.social_law.single_agent_projection import SingleAgentProjection
from unified_planning.social_law.robustness_verification import SimpleInstantaneousActionRobustnessVerifier
from unified_planning.social_law.waitfor_specification import WaitforSpecification
from unified_planning.social_law.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from unified_planning.model import Parameter, Fluent, InstantaneousAction, problem_kind
from unified_planning.shortcuts import *
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, InstantaneousAction, DurativeAction, Action
from typing import Type, List, Dict, Callable
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

class SocialLawRobustnessResult(Result):
    status : SocialLawRobustnessStatus
    counter_example : Optional["up.plans.Plan"]

    def __init__(self, status : SocialLawRobustnessStatus, counter_example : Optional["up.plans.Plan"]):
        self.status = status
        self.counter_example = counter_example
    


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
                presult = planner.solve(result.problem)
                if presult.status not in unified_planning.engines.results.POSITIVE_OUTCOMES:                    
                    return False
        return True

    def multi_agent_robustness_counterexample(self, problem : MultiAgentProblemWithWaitfor) -> Plan:
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
        
        with OneshotPlanner(name=self._planner_name, problem_kind=rbv_result.problem.kind) as planner:
            result = planner.solve(rbv_result.problem)
            if result.status in unified_planning.engines.results.POSITIVE_OUTCOMES:
                return result.plan
        return None         


    def is_robust(self, problem : MultiAgentProblemWithWaitfor) -> SocialLawRobustnessResult:
        status =  SocialLawRobustnessStatus.ROBUST_RATIONAL
        # Check single agent solvability
        if not self.is_single_agent_solvable(problem):
            return SocialLawRobustnessResult(SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT, None)
        
        # Check for rational robustness
        counter_example = self.multi_agent_robustness_counterexample(problem)        
        if counter_example is not None:
            for action_occurence in counter_example.actions:
                parts = action_occurence.action.name.split("_")
                if parts[0] == "f":
                    status = SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL
                    break
                elif parts[0] == "w":
                    status = SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK            
                    break
            assert(status in [SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL, SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK])        
        return SocialLawRobustnessResult(status, counter_example)

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
        
class SocialLaw(engines.engine.Engine, CompilerMixin):
    '''Social Law abstract class:
    this class requires a (multi agent) problem with waitfors, and applies itself to restrict some actions, resulting in a modified multi agent problem with waitfors.'''
    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.MA_SL_SOCIAL_LAW)
        self.added_waitfors = []
                
    @staticmethod
    def get_credits(**kwargs) -> Optional['Credits']:
        return credits

    @property
    def name(self):
        return "sl"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED_MULTI_AGENT")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_numbers("CONTINUOUS_NUMBERS")
        supported_kind.set_numbers("DISCRETE_NUMBERS")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
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
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= SocialLaw.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.MA_SL_SOCIAL_LAW

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = ProblemKind(problem_kind.features)    
        new_kind.set_problem_class("ACTION_BASED")    
        new_kind.unset_problem_class("ACTION_BASED_MULTI_AGENT")
        return new_kind


    def _compile(self, problem: "up.model.AbstractProblem", compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        '''Creates a problem that is a copy of the original problem, modified by the self social law.'''
        assert isinstance(problem, MultiAgentProblem)

        #Represents the map from the new action to the old action
        new_to_old: Dict[Action, Action] = {}

        new_problem = MultiAgentProblemWithWaitfor()
        new_problem.name = f'{self.name}_{problem.name}'

        for f in problem.ma_environment.fluents:
            default_val = problem.ma_environment.fluents_defaults[f]
            new_problem.ma_environment.add_fluent(f, default_initial_value=default_val)
        for ag in problem.agents:
            new_ag = up.model.multi_agent.Agent(ag.name, new_problem)
            for f in ag.fluents:
                default_val = ag.fluents_defaults[f]
                new_ag.add_fluent(f, default_initial_value=default_val)
            for a in ag.actions:
                new_action = a.clone()
                new_ag.add_action(new_action)
                new_to_old[new_action] = (ag, a)
            new_problem.add_agent(new_ag)
        new_problem._user_types = problem._user_types[:]
        new_problem._user_types_hierarchy = problem._user_types_hierarchy.copy()
        new_problem._objects = problem._objects[:]
        new_problem._initial_value = problem._initial_value.copy()
        new_problem._goals = problem._goals[:]
        new_problem._initial_defaults = problem._initial_defaults.copy()
        if isinstance(problem, MultiAgentProblemWithWaitfor):
            new_problem.waitfor.waitfor_map = problem.waitfor.waitfor_map.copy()

        for agent_name, action_name, precondition_fluent_name, pre_condition_args in self.added_waitfors:
            agent = new_problem.agent(agent_name)
            action = agent.action(action_name)
            if agent.has_fluent(precondition_fluent_name):
                precondition_fluent = agent.fluent(precondition_fluent_name)
            else:
                assert(new_problem.ma_environment.has_fluent(precondition_fluent_name))
                precondition_fluent = new_problem.ma_environment.fluent(precondition_fluent_name)
            pre_condition_arg_objs = []
            for arg in pre_condition_args:                
                if arg in action._parameters:                
                    arg_obj = action.parameter(arg)
                elif new_problem.has_object(arg):
                    arg_obj = new_problem.object(arg)
                else:
                    raise UPUsageError("Don't know what parameter " + arg + " is in waitfor(" + agent_name + ", " + action_name + ")")
                pre_condition_arg_objs.append(arg_obj)

            precondition = precondition_fluent(pre_condition_arg_objs)

            new_problem.waitfor.annotate_as_waitfor(agent_name, action_name, precondition)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )

    def add_waitfor_annotation(self, agent_name : str, action_name : str, precondition_fluent_name : str, pre_condition_args: List[str]):
        self.added_waitfors.append( (agent_name, action_name, precondition_fluent_name, pre_condition_args))