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
"""This module defines the single agent projection compiler class."""

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
from unified_planning.engines import Credits

credits = Credits('Single Agent Projection',
                  'Technion Cognitive Robotics Lab (cf. https://github.com/TechnionCognitiveRoboticsLab)',
                  'karpase@technion.ac.il',
                  'https://https://cogrob.net.technion.ac.il/',
                  'Apache License, Version 2.0',
                  'Projects a given multi-agent planning problem into the single agent planning problem of a given agent.',
                  'Projects a given multi-agent planning problem into the single agent planning problem of a given agent.')


class SingleAgentProjection(engines.engine.Engine, CompilerMixin):
    '''Single agent projection class:
    this class requires a (multi agent) problem and an agent, and offers the capability
    to produce the single agent projection planning problem for the given agent.

    This is done by only including the actions of the given agent, changing waitfor preconditions to regular preconditions, and setting the goal to the agent's goal.'''
    def __init__(self, agent: Agent):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.MA_SINGLE_AGENT_PROJECTION)                
        self._agent = agent
        
    @staticmethod
    def get_credits(**kwargs) -> Optional['Credits']:
        return credits

    @property
    def name(self):
        return "sap"

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
        return problem_kind <= SingleAgentProjection.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.MA_SINGLE_AGENT_PROJECTION

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = ProblemKind(problem_kind.features)    
        new_kind.set_problem_class("ACTION_BASED")    
        new_kind.unset_problem_class("ACTION_BASED_MULTI_AGENT")
        return new_kind

    @property
    def agent(self) -> Agent:
        """Returns the agent."""
        return self._agent


    def _compile(self, problem: "up.model.AbstractProblem", compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        '''Creates a problem that is a copy of the original problem
        but actions are modified and filtered.'''
        

        #Represents the map from the new action to the old action
        new_to_old: Dict[Action, Action] = {}

        new_problem = Problem()
        new_problem.name = f'{self.name}_{problem.name}'

        for fluent in problem.ma_environment.fluents:                        
            if fluent in problem.ma_environment.fluents_defaults:      
                default_val = problem.ma_environment.fluents_defaults[fluent]
                new_problem.add_fluent(fluent, default_initial_value=default_val)
            else:
                new_problem.add_fluent(fluent)

        for fluent in self.agent.fluents:
            if fluent in self.agent.fluents_defaults:
                default_val = self.agent.fluents_defaults[fluent]
                new_problem.add_fluent(fluent, default_initial_value=default_val)
            else:
                new_problem.add_fluent(fluent)

        eiv = problem.explicit_initial_values     
        for fluent in eiv:
            if fluent.is_dot():
                if fluent.agent().name == self.agent.name:
                    new_problem.set_initial_value(fluent.args[0], eiv[fluent])
            else:
                new_problem.set_initial_value(fluent, eiv[fluent])
            

        for action in self.agent.actions:            
            new_problem.add_action(action)
            new_to_old[action] = action

        for object in problem.all_objects:
            new_problem.add_object(object)


        # TODO: add agent goals when these are available
        for goal in problem.goals:
            if goal.is_dot():
                if goal.agent().name == self.agent.name:  # Compare agent names to handle social laws which change agents
                    new_problem.add_goal(goal.args[0])            
            else:
                new_problem.add_goal(goal)            

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )

    # def get_original_action(self, action: Action) -> Action:
    #     '''After the method get_rewritten_problem is called, this function maps
    #     the actions of the transformed problem into the actions of the original problem.'''
    #     return self._new_to_old[action]

    # def get_transformed_actions(self, action: Action) -> List[Action]:
    #     '''After the method get_rewritten_problem is called, this function maps
    #     the actions of the original problem into the actions of the transformed problem.'''
    #     return self._old_to_new[action]


env = up.environment.get_env()
env.factory.add_engine('SingleAgentProjection', __name__, 'SingleAgentProjection')
