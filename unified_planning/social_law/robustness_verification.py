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
"""This module defines the robustness verification compiler classes"""

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
from operator import neg
from unified_planning.model import Parameter, Fluent, InstantaneousAction
from unified_planning.shortcuts import *
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, InstantaneousAction, DurativeAction, Action
from typing import List, Dict
from itertools import product


class FluentMap():
    """ This class maintains a copy of each fluent in the given problem (environment and agent specific)"""
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.env_fluent_map = {}
        self.agent_fluent_map = {}

    def get_environment_version(self, fact):
        """get a copy of given environment fact
        """
        # TODO: there must be a cleaner way to do this...
        negate = False
        if fact.is_not():
            negate = True
            fact = fact.arg(0)
        gfact = FluentExp(
            self.env_fluent_map[fact.fluent().name],
            fact.args)
        if negate:
            return Not(gfact)
        else:
            return gfact

    def get_agent_version(self, agent, fact):
        """get the copy of given agent-specific agent.fact 
        """
        # TODO: there must be a cleaner way to do this...
        negate = False
        if fact.is_not():
            negate = True
            fact = fact.arg(0)
        gfact = FluentExp(
            self.agent_fluent_map[agent.name, fact.fluent().name],
            fact.args)
        if negate:
            return Not(gfact)
        else:
            return gfact   

    def get_correct_version(self, agent, fact):
        cfact = fact
        if fact.is_not():
            cfact = fact.arg(0)
        if cfact.fluent() in agent.fluents:
            return self.get_agent_version(agent, fact)
        else:
            return self.get_environment_version(fact)

    def add_facts(self, problem, new_problem):
        # Add copy for each fact
        for f in problem.ma_environment.fluents:
            g_fluent = Fluent(self.prefix + "-" + f.name, f.type, f.signature)            
            self.env_fluent_map[f.name] = g_fluent        
            default_val = problem.ma_environment.fluents_defaults[f]    
            new_problem.add_fluent(g_fluent, default_initial_value=default_val)            

        for agent in problem.agents:
            for f in agent.fluents:
                g_fluent = Fluent(self.prefix + "-" + agent.name + "-" + f.name, f.type, f.signature)                
                self.agent_fluent_map[agent.name, f.name] = g_fluent      
                default_val = agent.fluents_defaults[f]          
                new_problem.add_fluent(g_fluent, default_initial_value=default_val)                

    

class RobustnessVerifier(engines.engine.Engine, CompilerMixin):
    '''Robustness verifier (abstract) class:
    this class requires a (multi agent) problem, and creates a classical planning problem which is unsolvable iff the multi agent problem is not robust.'''
    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.MA_SL_ROBUSTNESS_VERIFICATION)
        self.act_pred = None
        
    @property
    def name(self):
        return "rbv"

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.MA_SL_ROBUSTNESS_VERIFICATION

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = ProblemKind(problem_kind.features)    
        new_kind.set_problem_class("ACTION_BASED")    
        new_kind.unset_problem_class("ACTION_BASED_MULTI_AGENT")
        return new_kind

    def get_agent_obj(self, agent):
        return Object(agent.name, self.agent_type)

    def get_agent_goal(self, problem, agent):
        """ Returns the individual goal of the given agent"""
        #TODO: update when new API is available
        l = []
        for goal in problem.goals:
            if goal.is_dot() and goal.agent() == agent:
                l.append(goal)
        return l



    def initialize_problem(self, problem):
        new_problem = Problem(f'{self.name}_{problem.name}')

        # Add types
        for type in problem.user_types:
            new_problem._add_user_type(type)

        self.agent_type = UserType("agent")
        new_problem._add_user_type(self.agent_type)

        # Add objects 
        new_problem.add_objects(problem.all_objects)
        for agent in problem.agents:
            new_problem.add_object(Object(agent.name, self.agent_type))

        # Add global and local copy for each fact
        self.global_fluent_map = FluentMap("g")
        self.global_fluent_map.add_facts(problem, new_problem)

        self.local_fluent_map = {}
        for agent in problem.agents:        
            self.local_fluent_map[agent] = FluentMap("l-" + agent.name)
            self.local_fluent_map[agent].add_facts(problem, new_problem)

        return new_problem

            

class InstantaneousActionRobustnessVerifier(RobustnessVerifier):
    '''Robustness verifier class for instanteanous actions:
    this class requires a (multi agent) problem, and creates a classical planning problem which is unsolvable iff the multi agent problem is not robust.'''
    def __init__(self):
        RobustnessVerifier.__init__(self)
    
    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED_MULTI_AGENT")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_time("DISCRETE_TIME")        
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= InstantaneousActionRobustnessVerifier.supported_kind()

    def create_action_copy(self, agent, action, suffix):
        """Create a new copy of an action, with name action_name_suffix, and duplicates the local preconditions/effects
        """
        d = {}
        for p in action.parameters:
            d[p.name] = p.type

        new_action = InstantaneousAction(action.name + suffix, _parameters=d)
        new_action.add_precondition(self.act_pred)
        # TODO: can probably do this better with a substitution walker
        for fact in action.preconditions:
            if fact.is_and():
                for f in fact.args:
                    new_action.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, f))
            else:
                new_action.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, fact))
        for effect in action.effects:
            new_action.add_effect(self.local_fluent_map[agent].get_correct_version(agent, effect.fluent), effect.value)

        return new_action


    def _compile(self, problem: "up.model.AbstractProblem", compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        '''Creates a the robustness verification problem.'''

        #Represents the map from the new action to the old action
        new_to_old: Dict[Action, Action] = {}
        
        new_problem = self.initialize_problem(problem)

        self.waiting_fluent_map = FluentMap("w")
        self.waiting_fluent_map.add_facts(problem, new_problem)

        # Add fluents
        failure = Fluent("failure")
        crash = Fluent("crash")
        act = Fluent("act")
        fin = Fluent("fin", _signature=[Parameter("a", self.agent_type)])
        waiting = Fluent("waiting", _signature=[Parameter("a", self.agent_type)])

        self.act_pred = act

        new_problem.add_fluent(failure, default_initial_value=False)
        new_problem.add_fluent(crash, default_initial_value=False)
        new_problem.add_fluent(act, default_initial_value=True)
        new_problem.add_fluent(fin, default_initial_value=False)
        new_problem.add_fluent(waiting, default_initial_value=False)


        # Add actions
        for agent in problem.agents:
            end_s = InstantaneousAction("end_s_" + agent.name)
            end_s.add_precondition(Not(fin(self.get_agent_obj(agent))))            
            for goal in self.get_agent_goal(problem, agent):
                end_s.add_precondition(self.global_fluent_map.get_correct_version(agent, goal.args[0]))
                end_s.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, goal.args[0]))
            end_s.add_effect(fin(self.get_agent_obj(agent)), True)
            end_s.add_effect(act, False)
            new_problem.add_action(end_s)

            for i, goal in enumerate(self.get_agent_goal(problem, agent)):
                end_f = InstantaneousAction("end_f_" + agent.name + "_" + str(i))
                end_f.add_precondition(Not(fin(self.get_agent_obj(agent))))
                end_f.add_precondition(Not(self.global_fluent_map.get_correct_version(agent, goal.args[0])))
                for g in self.get_agent_goal(problem, agent):                    
                    end_f.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, g.args[0]))
                end_f.add_effect(fin(self.get_agent_obj(agent)), True)
                end_f.add_effect(act, False)
                end_f.add_effect(failure, True)
                new_problem.add_action(end_f)

            for action in agent.actions:
                # Success version - affects globals same way as original
                a_s = self.create_action_copy(agent, action, "_s_" + agent.name)
                a_s.add_precondition(Not(waiting(self.get_agent_obj(agent))))
                a_s.add_precondition(Not(crash))
                for effect in action.effects:
                    if effect.value.is_true():
                        a_s.add_precondition(Not(self.waiting_fluent_map.get_correct_version(agent, effect.fluent)))
                for fact in action.preconditions:
                    if fact.is_and():
                        for f in fact.args:
                            a_s.add_precondition(self.global_fluent_map.get_correct_version(agent, f))
                    else:
                        a_s.add_precondition(self.global_fluent_map.get_correct_version(agent, fact))
                for effect in action.effects:
                    a_s.add_effect(self.global_fluent_map.get_correct_version(agent, effect.fluent), effect.value)
                new_problem.add_action(a_s)

                real_preconds = []
                for fact in action.preconditions:
                    if fact.is_and():
                        real_preconds += fact.args
                    else:
                        real_preconds.append(fact)
                
                # Fail version
                for i, fact in enumerate(real_preconds):
                    a_f = self.create_action_copy(agent, action, "_f_" + agent.name + "_" + str(i))
                    a_f.add_precondition(Not(waiting(self.get_agent_obj(agent))))
                    a_f.add_precondition(Not(crash))
                    #TODO: get wait preconditions
                    for pre in action.preconditions:
                        a_f.add_precondition(self.global_fluent_map.get_correct_version(agent, pre))
                    a_f.add_precondition(Not(self.global_fluent_map.get_correct_version(agent, fact)))
                    a_f.add_effect(failure, True)
                    a_f.add_effect(crash, True)
                    new_problem.add_action(a_f)

                # Wait version
                #TODO: get wait preconditions
                for i, fact in enumerate(action.preconditions):
                    a_w = self.create_action_copy(agent, action, "_w_" + agent.name + str(i))
                    a_w.add_precondition(Not(crash))
                    a_w.add_precondition(Not(waiting(self.get_agent_obj(agent))))
                    a_w.add_precondition(Not(self.global_fluent_map.get_correct_version(agent,fact)))
                    assert not fact.is_not()
                    a_w.add_effect(self.waiting_fluent_map.get_correct_version(agent,fact), True)  # , action.agent.obj), True)
                    a_w.add_effect(waiting(self.get_agent_obj(agent)), True)
                    a_w.add_effect(failure, True)
                    new_problem.add_action(a_w)

                # Phantom version            
                a_pc = self.create_action_copy(agent, action, "_pc_" + agent.name)
                a_pc.add_precondition(crash)
                new_problem.add_action(a_pc)

                # Phantom version            
                a_pw = self.create_action_copy(agent, action, "_pw_" + agent.name)
                a_pw.add_precondition(waiting(self.get_agent_obj(agent)))
                new_problem.add_action(a_pw)

        # Initial state
        eiv = problem.explicit_initial_values     
        for fluent in eiv:
            if fluent.is_dot():                
                new_problem.set_initial_value(self.global_fluent_map.get_correct_version(fluent.agent(), fluent.args[0]), eiv[fluent])
                for a in problem.agents:
                    new_problem.set_initial_value(self.local_fluent_map[a].get_correct_version(fluent.agent(), fluent.args[0]), eiv[fluent])
            else:
                new_problem.set_initial_value(self.global_fluent_map.get_environment_version(fluent), eiv[fluent])
                for a in problem.agents:
                    new_problem.set_initial_value(self.local_fluent_map[a].get_environment_version(fluent), eiv[fluent])

        # Goal
        new_problem.add_goal(failure)
        for agent in problem.agents:
            new_problem.add_goal(fin(self.get_agent_obj(agent)))

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
