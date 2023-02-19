# Copyright 2023 Technion
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
"""This module defines a helper class which converts a single agent (classical) problem into a multi agent planning problem."""

import unified_planning as up
import unified_planning.engines as engines
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.model.multi_agent import *
from unified_planning.model import *
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPExpressionDefinitionError, UPProblemDefinitionError
from typing import List, Dict, Union, Optional
from unified_planning.engines.compilers.utils import replace_action, get_fresh_name
from functools import partial
import unified_planning as up
from unified_planning.social_law.robustness_verification import FluentMap, FluentMapSubstituter
from unified_planning.engines import Credits
from unified_planning.model.fnode import FNode
from unified_planning.model.operators import OperatorKind
from unified_planning.model.expression import Expression
from unified_planning.exceptions import UPTypeError
from typing import List, Dict
from unified_planning.exceptions import UPUsageError
import unified_planning.model.walkers as walkers
from unified_planning.model.walkers.identitydag import IdentityDagWalker

credits = Credits('Single Agent to Multi Agent Converter',
                  'Technion Cognitive Robotics Lab (cf. https://github.com/TechnionCognitiveRoboticsLab)',
                  'karpase@technion.ac.il',
                  'https://https://cogrob.net.technion.ac.il/',
                  'Apache License, Version 2.0',
                  'Compilation from a single agent planning problem to a multi agent planning problem.',
                  'Compilation from a single agent planning problem to a multi agent planning problem. Uses simple rules to allocate actions/goals to agents.')


class PartialGrounder(IdentityDagWalker):
    """
    This walker is used to partially ground an expression - it takes a list of parameter names and the values to assign to them
    and replaces every occurence of the found parameter name by its assigned value.
    """

    def __init__(self, env):
        self._env = env
        IdentityDagWalker.__init__(self, self._env, True)        

    def ground(
        self, expression: FNode, partial_grounding_map: Dict[str, Object]
    ) -> FNode:
        """
        This method takes in input an expression and performs partial grounding.

        :param expression: The target expression to partially ground.
        :param partial_grounding_map: The dictionary mapping parameter names to objects
        :return: An expression which is partially grounded
        """
        self._partial_grounding_map = partial_grounding_map
        return self.walk(expression)
    
    def walk_param_exp(self, expression: FNode, args: List[FNode], **kwargs) -> FNode:  
        if expression.parameter().name in self._partial_grounding_map:
            return self.manager.ObjectExp(self._partial_grounding_map[expression.parameter().name])
        else:      
            return self.manager.ParameterExp(expression.parameter())



class SingleAgentToMultiAgentConverter(engines.engine.Engine, CompilerMixin):
    '''Single Agent to Multi Agent Converter class:
    this class requires a single agent problem and generates a multi agent problem.'''
    def __init__(self, agent_types : List[str]):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.SA_MA_CONVERSION)                
        self.agent_types = agent_types
        
    @staticmethod
    def get_credits(**kwargs) -> Optional['Credits']:
        return credits

    @property
    def name(self):
        return "samac"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED")
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
        supported_kind.set_quality_metrics("PLAN_LENGTH")
        supported_kind.set_quality_metrics("ACTIONS_COST")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= SingleAgentToMultiAgentConverter.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.SA_MA_CONVERSION

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = ProblemKind(problem_kind.features)    
        new_kind.set_problem_class("ACTION_BASED_MULTI_AGENT")    
        new_kind.unset_problem_class("ACTION_BASED")
        return new_kind

    def agent_name(self, obj : Object):
        return "agent__" + obj.name


    def _compile(self, problem: "up.model.AbstractProblem", compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        '''Creates a problem that is a multi-agent version of the original problem'''
        assert isinstance(problem, Problem)

        #Represents the map from the new action to the old action
        new_to_old: Dict[Action, Action] = {}

        new_problem = MultiAgentProblem()
        new_problem.name = f'{self.name}_{problem.name}'
                
        new_problem.add_objects(problem.all_objects)

        for agent_type in self.agent_types:
            utype = problem.user_type(agent_type)
            for agent_obj in problem.objects(utype):
                agent = Agent(self.agent_name(agent_obj), new_problem)
                new_problem.add_agent(agent)                

        fmap = FluentMap("c")
        for f in problem.fluents:
            new_problem.ma_environment.add_fluent(f)
        
        eiv = problem.explicit_initial_values     
        for fluent in eiv:            
            new_problem.set_initial_value(fluent, eiv[fluent])
        
        for action in problem.actions:
            d = {}
            param = None
            for p in action.parameters:                
                if param is None and p.type.name in self.agent_types:
                    param = p
                else:
                    d[p.name] = p.type
            if param is None:
                raise UPUsageError("No parameter of action '" + action.name + "' matches any specified agent type")
            
            for agent_obj in problem.objects(param.type):
                agent = new_problem.agent(self.agent_name(agent_obj))
                pg = PartialGrounder(new_problem.env)

                if isinstance(action, InstantaneousAction):                    
                    new_action = InstantaneousAction(action.name, _parameters=d)        
                    for p in action.preconditions:
                        new_action.add_precondition(  pg.ground(p, {param.name : agent_obj})  )
                    for e in action.effects:
                        new_action.add_effect(   pg.ground(e.fluent, {param.name : agent_obj}), e.value)
                elif isinstance(action, DurativeAction):
                    new_action = DurativeAction(action.name, _parameters=d)     
                    new_action.set_duration_constraint(action.duration)
                    
                    for timing in action.conditions.keys():
                        for c in action.conditions[timing]:                            
                            new_action.add_condition(timing, pg.ground(c, {param.name : agent_obj}))

                    for timing in action.effects.keys():
                        for e in action.effects[timing]:
                            new_action.add_effect(timing, pg.ground(e.fluent, {param.name : agent_obj}), e.value)

                agent.add_action(new_action)
            
            

        for goal in problem.goals:            
            new_problem.add_goal(goal)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )



env = up.environment.get_env()
env.factory.add_engine('SingleAgentToMultiAgentConverter', __name__, 'SingleAgentToMultiAgentConverter')
