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

credits = Credits('Social Law',
                  'Technion Cognitive Robotics Lab (cf. https://github.com/TechnionCognitiveRoboticsLab)',
                  'karpase@technion.ac.il',
                  'https://https://cogrob.net.technion.ac.il/',
                  'Apache License, Version 2.0',
                  'Represents a social law, which is a tranformation of a multi-agent problem + waitfor specification to a new multi-agent problem + waitfor.',
                  'Represents a social law, which is a tranformation of a multi-agent problem + waitfor specification to a new multi-agent problem + waitfor.')

class SocialLaw(engines.engine.Engine, CompilerMixin):
    '''Social Law abstract class:
    this class requires a (multi agent) problem with waitfors, and applies itself to restrict some actions, resulting in a modified multi agent problem with waitfors.'''
    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.MA_SL_SOCIAL_LAW)
        self.added_waitfors = []
        self.disallowed_actions = {}
        self.new_fluents = []
        self.new_fluent_initial_val = []
        self.added_action_parameters = {}
        self.added_preconditions = []
        self.new_objects = []

    def __repr__(self) -> str:
        s = "added_waitfors: " + str(self.added_waitfors) + "\n" + \
            "disallowd actions: " + str(self.disallowed_actions) + "\n" + \
            "new fluents: " + str(self.new_fluents) + "\n" + \
            "new fluents initial vals: " + str(self.new_fluent_initial_val) + "\n" + \
            "added action parameters: " + str(self.added_action_parameters) + "\n" + \
            "added preconditions: " + str(self.added_preconditions) + "\n" + \
            "new objects: " + str(self.new_objects)
        return s
                
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

        # New objects
        for obj_name, obj_type_name in self.new_objects:
            if obj_type_name is not None:
                obj_type = new_problem.user_type(obj_type_name)
                new_problem.add_object(obj_name, obj_type)
            else:
                new_problem.add_object(obj_name)

        # New fluents
        for agent_name, fluent_name, signature, default_val in self.new_fluents:
            new_signature = OrderedDict()
            for pname, ptype_name in signature.items():
                if ptype_name is not None:
                    ptype = new_problem.user_type(ptype_name)
                new_signature[pname] = ptype
            new_f = Fluent(fluent_name, _signature = new_signature)
            if agent_name is not None:
                agent = new_problem.agent(agent_name)
                agent.add_fluent(new_f, default_initial_value = default_val)    
            else:
                new_problem.ma_environment.add_fluent(new_f, default_initial_value = default_val)

        # new fluent initial values
        for agent_name, fluent_name, args, val in self.new_fluent_initial_val:
            if agent_name is not None:
                agent = new_problem.agent(agent_name)
                fluent = agent.fluent(fluent_name)
            else:
                fluent = new_problem.ma_environment.fluent(fluent_name)
            arg_objs = []
            for arg in args:
                arg_obj = new_problem.object(arg)
                arg_objs.append(arg_obj)
            new_problem.set_initial_value(
                FluentExp(fluent, arg_objs), 
                val)

        # Added action parameters
        for agent_name, action_name in self.added_action_parameters.keys():
            agent = new_problem.agent(agent_name)
            action = agent.action(action_name)
            for param_name, param_type_name in self.added_action_parameters.get( (ag.name, a.name), []):
                if param_type_name is not None:
                    param_type = new_problem.user_type(param_type_name)
                    action._parameters[param_name] = Parameter(param_name, param_type)
                else:
                    action._parameters[param_name] = Parameter(param_name)
            
        # Add action preconditions
        for agent_name, action_name, precondition_fluent_name, pre_condition_args in self.added_preconditions:
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
                    raise UPUsageError("Don't know what parameter " + arg + " is in new precondition for (" + agent_name + ", " + action_name + ")")
                pre_condition_arg_objs.append(arg_obj)
            precondition = FluentExp(precondition_fluent, pre_condition_arg_objs)
            action.add_precondition(precondition)

        # Add waitfor annotations
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

            precondition = FluentExp(precondition_fluent, pre_condition_arg_objs)
            new_problem.waitfor.annotate_as_waitfor(agent_name, action_name, precondition)

        # Disallow actions
        for agent_name, action_name in self.disallowed_actions.keys():
            agent = new_problem.agent(agent_name)
            action = agent.action(action_name)
            
            allowed_name = "allowed__" + action.name
            if not agent.has_fluent(allowed_name): # this could have been added by a previously applied SL
                allowed_fluent = Fluent("allowed__" + action.name, _signature = action.parameters)
                agent.add_fluent(allowed_fluent, default_initial_value = True)
                action.add_precondition(
                    FluentExp(allowed_fluent, action.parameters)
                )
            else:
                allowed_fluent = agent.fluent(allowed_name)
            for disallowed_args in self.disallowed_actions[(agent_name, action_name)]:
                arg_objs = []
                for arg in disallowed_args:
                    arg_obj = new_problem.object(arg)
                    arg_objs.append(arg_obj)
                new_problem.set_initial_value(
                    FluentExp(allowed_fluent, arg_objs), 
                    False
                )

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )

    def add_waitfor_annotation(self, agent_name : str, action_name : str, precondition_fluent_name : str, pre_condition_args: List[str]):
        self.added_waitfors.append( (agent_name, action_name, precondition_fluent_name, pre_condition_args) )

    def disallow_action(self, agent_name : str, action_name : str, args: List[str]):
        if (agent_name, action_name) not in self.disallowed_actions:
            self.disallowed_actions[(agent_name, action_name)] = list()
        self.disallowed_actions[(agent_name, action_name)].append(args)

    def add_new_fluent(self, agent_name : Optional[str], fluent_name : str, signature: OrderedDict[str, Optional[str]], default_val):
        self.new_fluents.append( (agent_name, fluent_name, signature, default_val) )

    def set_initial_value_for_new_fluent(self, agent_name : Optional[str], fluent_name : str, args: List[str], val):      
        is_new_fluent = False
        for n_agent_name, n_fluent_name, _, _ in self.new_fluents:
            if agent_name == n_agent_name and fluent_name == n_fluent_name:
                is_new_fluent = True
                break
        if not is_new_fluent:
            raise UPUsageError("must only set initial values for new fluents")
        self.new_fluent_initial_val.append( ( agent_name, fluent_name, args, val) )
        
    def add_parameter_to_action(self, agent_name : str, action_name : str, parameter_name : str, parameter_type_name : Optional[str]):
        if (agent_name, action_name) not in self.added_action_parameters:
            self.added_action_parameters[(agent_name, action_name)] = list()      
        self.added_action_parameters[(agent_name, action_name)].append(  (parameter_name, parameter_type_name) )

    def add_precondition_to_action(self, agent_name : str, action_name : str, precondition_fluent_name : str, pre_condition_args: List[str]):
        self.added_preconditions.append( (agent_name, action_name, precondition_fluent_name, pre_condition_args) )

    def add_new_object(self, obj_name : str, obj_type_name : Optional[str]):
        self.new_objects.append( (obj_name, obj_type_name))