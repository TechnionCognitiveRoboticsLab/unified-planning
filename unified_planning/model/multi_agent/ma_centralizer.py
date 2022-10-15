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
"""This module defines the compilation from multi agent planning problem to a centralized single agent problem."""

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
from unified_planning.social_law.robustness_verification import FluentMap
from unified_planning.engines import Credits
import unified_planning.model.walkers as walkers
from unified_planning.model.walkers.identitydag import IdentityDagWalker
from unified_planning.model.fnode import FNode
from unified_planning.model.operators import OperatorKind
from unified_planning.model.expression import Expression
from unified_planning.exceptions import UPTypeError
from typing import List, Dict

credits = Credits('Multi Agent Problem Centralizer',
                  'Technion Cognitive Robotics Lab (cf. https://github.com/TechnionCognitiveRoboticsLab)',
                  'karpase@technion.ac.il',
                  'https://https://cogrob.net.technion.ac.il/',
                  'Apache License, Version 2.0',
                  'Compilation from a multi agent planning problem to a centralized single agent problem.',
                  'Compilation from a multi agent planning problem to a centralized single agent problem.')


class FluentMapSubstituter(IdentityDagWalker):
    """Performs substitution according to the given FluentMap"""

    def __init__(self, env: "unified_planning.environment.Environment"):
        IdentityDagWalker.__init__(self, env, True)
        self.env = env
        self.manager = env.expression_manager
        self.type_checker = env.type_checker

    def _get_key(self, expression, **kwargs):
        return expression

    def substitute(self, expression: FNode, fmap: FluentMap, agent: Agent) -> FNode:
        """
        Performs substitution into the given expression, according to the given FluentMap
        """
        return self.walk(expression, fmap=fmap, agent = agent)

    def walk_fluent_exp(self, expression: FNode, args: List[FNode], **kwargs) -> FNode:
        return kwargs["fmap"].get_correct_version(kwargs["agent"], expression)



class MultiAgentProblemCentralizer(engines.engine.Engine, CompilerMixin):
    '''Multi Agent Problem Centralizer class:
    this class requires a (multi agent) problem and generates a centralized single agnet problem.'''
    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.MA_CENTRALIZATION)                
        
    @staticmethod
    def get_credits(**kwargs) -> Optional['Credits']:
        return credits

    @property
    def name(self):
        return "mac"

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
        return problem_kind <= MultiAgentProblemCentralizer.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.MA_CENTRALIZATION

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = ProblemKind(problem_kind.features)    
        new_kind.set_problem_class("ACTION_BASED")    
        new_kind.unset_problem_class("ACTION_BASED_MULTI_AGENT")
        return new_kind

    def _compile(self, problem: "up.model.AbstractProblem", compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        '''Creates a problem that is a centralized single agent version of the original problem'''
        assert isinstance(problem, MultiAgentProblem)

        #Represents the map from the new action to the old action
        new_to_old: Dict[Action, (Agent, Action)] = {}

        new_problem = Problem()
        new_problem.name = f'{self.name}_{problem.name}'
                
        new_problem.add_objects(problem.all_objects)


        fmap = FluentMap("c")
        fmap.add_facts(problem, new_problem)

        eiv = problem.explicit_initial_values     
        for fluent in eiv:
            if fluent.is_dot():                
                new_problem.set_initial_value(fmap.get_agent_version(fluent.agent(), fluent.args[0]), eiv[fluent])
            else:
                new_problem.set_initial_value(fmap.get_environment_version(fluent), eiv[fluent])
            

        fsub = FluentMapSubstituter(new_problem.env)
        for agent in problem.agents:
            for action in agent.actions:
                d = {}
                for p in action.parameters:
                    d[p.name] = p.type
                if isinstance(action, InstantaneousAction):                    
                    new_action = InstantaneousAction(agent.name + "__" + action.name, _parameters=d)        
                    for p in action.preconditions:
                        new_action.add_precondition(fsub.substitute(p, fmap, agent))                        
                    for e in action.effects:
                        new_action.add_effect(fsub.substitute(e.fluent, fmap, agent), e.value)                    
                elif isinstance(action, DurativeAction):
                    new_action = DurativeAction(agent.name + "__" + action.name, _parameters=d)     
                    new_action.set_duration_constraint(action.duration)
                    
                    for timing in action.conditions.keys():
                        for c in action.conditions[timing]:                            
                            new_action.add_condition(timing, fsub.substitute(c, fmap, agent))

                    for timing in action.effects.keys():
                        for e in action.effects[timing]:
                            new_action.add_effect(timing, fsub.substitute(e.fluent, fmap, agent), e.value)
                new_problem.add_action(new_action)
                new_to_old[new_action] = (agent, action)

        for goal in problem.goals:
            if goal.is_dot():                
                new_problem.add_goal(fmap.get_agent_version(goal.agent(), goal.args[0]))
            else:
                new_problem.add_goal(fmap.get_environment_version(goal))

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )



env = up.environment.get_env()
env.factory.add_engine('MultiAgentProblemCentralizer', __name__, 'MultiAgentProblemCentralizer')
