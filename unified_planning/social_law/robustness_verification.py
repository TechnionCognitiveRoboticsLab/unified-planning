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

#TODO - make sure handling of agent specific facts in goals (.args[0]) is correct


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
from unified_planning.social_law.waitfor_specification import WaitforSpecification
from unified_planning.social_law.ma_problem_waitfor import MultiAgentProblemWithWaitfor
import unified_planning as up
from unified_planning.engines import Credits
from unified_planning.io.pddl_writer import PDDLWriter


credits = Credits('Robustness Verification',
                  'Technion Cognitive Robotics Lab (cf. https://github.com/TechnionCognitiveRoboticsLab)',
                  'karpase@technion.ac.il',
                  'https://https://cogrob.net.technion.ac.il/',
                  'Apache License, Version 2.0',
                  'Creates a planning problem which verifies the robustness of a multi-agent planning problem with given waitfor specification.',
                  'Creates a planning problem which verifies the robustness of a multi-agent planning problem with given waitfor specification.')



class FluentMap():
    """ This class maintains a copy of each fluent in the given problem (environment and agent specific). Default value (if specified) is the default value for the new facts."""
    def __init__(self, prefix: str, default_value = None, override_type = None):
        self.prefix = prefix
        self.env_fluent_map = {}
        self.agent_fluent_map = {}
        self._default_value = default_value
        self._override_type = override_type

    @property
    def default_value(self):
        return self._default_value

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

    def create_fluent(self, f, agent=None):
        if self._override_type is not None:
            ftype = self._override_type
        else:
            ftype = f.type
        
        if agent is None:
            name = self.prefix + "-" + f.name
        else:
            name = self.prefix + "-" + agent.name + "-" + f.name
        g_fluent = Fluent(name, ftype, f.signature)
        return g_fluent

    def add_facts(self, problem, new_problem):
        # Add copy for each fact
        for f in problem.ma_environment.fluents:
            g_fluent = self.create_fluent(f)
            self.env_fluent_map[f.name] = g_fluent   
            if self.default_value is None:
                default_val = problem.ma_environment.fluents_defaults[f]    
            else:
                default_val = self.default_value
            new_problem.add_fluent(g_fluent, default_initial_value=default_val)            

        for agent in problem.agents:
            for f in agent.fluents:
                g_fluent = self.create_fluent(f, agent)
                self.agent_fluent_map[agent.name, f.name] = g_fluent      
                if self.default_value is None:
                    default_val = agent.fluents_defaults[f]          
                else:
                    default_val = self.default_value
                new_problem.add_fluent(g_fluent, default_initial_value=default_val)                

    

class RobustnessVerifier(engines.engine.Engine, CompilerMixin):
    '''Robustness verifier (abstract) class:
    this class requires a (multi agent) problem, and creates a classical planning problem which is unsolvable iff the multi agent problem is not robust.'''
    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.MA_SL_ROBUSTNESS_VERIFICATION)
        self.act_pred = None        
        
    @staticmethod
    def get_credits(**kwargs) -> Optional['Credits']:
        return credits

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

    def get_agent_obj(self, agent : Agent):
        return Object(agent.name, self.agent_type)

    def get_agent_goal(self, problem : MultiAgentProblem, agent : Agent):
        """ Returns the individual goal of the given agent"""
        #TODO: update when new API is available
        l = []
        for goal in problem.goals:
            if goal.is_dot() and goal.agent() == agent:
                l.append(goal)
        return l

    def get_action_preconditions(self, problem : MultiAgentProblemWithWaitfor, agent : Agent, action : Action, fail : bool, wait: bool) -> List[FNode]:
        """ Get the preconditions for the given action of the given agent. fail/wait specify which preconditions we want (True to return, False to omit) """
        assert fail or wait
        if wait and not fail:
            return problem.waitfor.get_preconditions_wait(agent.name, action.name)
        else:
            preconds = []
            for fact in action.preconditions:
                if fact.is_and():
                    if wait or not fact.args in problem.waitfor.get_preconditions_wait(agent.name, action.name):
                        preconds += fact.args
                else:
                    if wait or not fact in problem.waitfor.get_preconditions_wait(agent.name, action.name):
                        preconds.append(fact)
        return preconds


    def initialize_problem(self, problem):
        assert isinstance(problem, MultiAgentProblemWithWaitfor)
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

    def create_action_copy(self, problem: MultiAgentProblemWithWaitfor, agent : Agent , action : InstantaneousAction, prefix : str):
        """Create a new copy of an action, with name prefix_action_name, and duplicates the local preconditions/effects
        """
        d = {}
        for p in action.parameters:
            d[p.name] = p.type

        new_action = InstantaneousAction(prefix + "_" + agent.name + "_" + action.name, _parameters=d)        
        for fact in self.get_action_preconditions(problem, agent, action, True, True):
            new_action.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, fact))
        for effect in action.effects:
            new_action.add_effect(self.local_fluent_map[agent].get_correct_version(agent, effect.fluent), effect.value)

        return new_action

class SimpleInstantaneousActionRobustnessVerifier(InstantaneousActionRobustnessVerifier):
    '''Robustness verifier class for instanteanous actions using alternative formulation:
    this class requires a (multi agent) problem, and creates a classical planning problem which is unsolvable iff the multi agent problem is not robust.
    Implements the robustness verification compilation from Nir, Shleyfman, Karpas limited to propositions with the bugs fixed
    '''
    def __init__(self):
        InstantaneousActionRobustnessVerifier.__init__(self)

    @property
    def name(self):
        return "srbv"

    def _compile(self, problem: "up.model.AbstractProblem", compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        '''Creates a the robustness verification problem.'''

        #Represents the map from the new action to the old action
        new_to_old: Dict[Action, Action] = {}
        
        new_problem = self.initialize_problem(problem)

        self.waiting_fluent_map = FluentMap("w", default_value=False)
        self.waiting_fluent_map.add_facts(problem, new_problem)

        # Add fluents
        failure = Fluent("failure")
        crash = Fluent("crash")
        act = Fluent("act")
        fin = Fluent("fin", _signature=[Parameter("a", self.agent_type)])
        waiting = Fluent("waiting", _signature=[Parameter("a", self.agent_type)])

        act_pred = act

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
                a_s = self.create_action_copy(problem, agent, action, "s")
                a_s.add_precondition(Not(waiting(self.get_agent_obj(agent))))
                a_s.add_precondition(Not(crash))
                for effect in action.effects:
                    if effect.value.is_true():
                        a_s.add_precondition(Not(self.waiting_fluent_map.get_correct_version(agent, effect.fluent)))
                for fact in self.get_action_preconditions(problem, agent, action, True, True):
                    a_s.add_precondition(self.global_fluent_map.get_correct_version(agent, fact))
                for effect in action.effects:
                    a_s.add_effect(self.global_fluent_map.get_correct_version(agent, effect.fluent), effect.value)
                new_problem.add_action(a_s)

                real_preconds = self.get_action_preconditions(problem, agent, action, fail=True, wait=False)
                
                # Fail version
                for i, fact in enumerate(real_preconds):
                    a_f = self.create_action_copy(problem, agent, action, "f_" + str(i))
                    a_f.add_precondition(act_pred)
                    a_f.add_precondition(Not(waiting(self.get_agent_obj(agent))))
                    a_f.add_precondition(Not(crash))                    
                    for pre in self.get_action_preconditions(problem, agent, action, False, True):
                        a_f.add_precondition(self.global_fluent_map.get_correct_version(agent, pre))
                    a_f.add_precondition(Not(self.global_fluent_map.get_correct_version(agent, fact)))
                    a_f.add_effect(failure, True)
                    a_f.add_effect(crash, True)
                    new_problem.add_action(a_f)

                # Wait version                
                for i, fact in enumerate(self.get_action_preconditions(problem, agent, action, False, True)): 
                    a_w = self.create_action_copy(problem, agent, action, "w_" + str(i))
                    a_w.add_precondition(act_pred)
                    a_w.add_precondition(Not(crash))
                    a_w.add_precondition(Not(waiting(self.get_agent_obj(agent))))
                    a_w.add_precondition(Not(self.global_fluent_map.get_correct_version(agent,fact)))
                    assert not fact.is_not()
                    a_w.add_effect(self.waiting_fluent_map.get_correct_version(agent,fact), True)  # , action.agent.obj), True)
                    a_w.add_effect(waiting(self.get_agent_obj(agent)), True)
                    a_w.add_effect(failure, True)
                    new_problem.add_action(a_w)

                # Phantom version            
                a_pc = self.create_action_copy(problem, agent, action, "pc")
                a_pc.add_precondition(act_pred)
                a_pc.add_precondition(crash)
                new_problem.add_action(a_pc)

                # Phantom version            
                a_pw = self.create_action_copy(problem, agent, action, "pw")
                a_pw.add_precondition(act_pred)
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


class WaitingActionRobustnessVerifier(InstantaneousActionRobustnessVerifier):
    '''Robustness verifier class for instanteanous actions using alternative formulation:
    this class requires a (multi agent) problem, and creates a classical planning problem which is unsolvable iff the multi agent problem is not robust.
    Implements the robustness verification compilation from Tuisov, Shleyfman, Karpas with the bugs fixed
    '''
    def __init__(self):
        InstantaneousActionRobustnessVerifier.__init__(self)

    
    @property
    def name(self):
        return "wrbv"

    def _compile(self, problem: "up.model.AbstractProblem", compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        '''Creates a the robustness verification problem.'''

        #Represents the map from the new action to the old action
        new_to_old: Dict[Action, Action] = {}
        
        new_problem = self.initialize_problem(problem)

        self.waiting_fluent_map = FluentMap("w", default_value=False)
        self.waiting_fluent_map.add_facts(problem, new_problem)

        # Add fluents
        stage_1 = Fluent("stage 1")
        stage_2 = Fluent("stage 2")
        precondition_violation = Fluent("precondition violation")
        possible_deadlock = Fluent("possible deadlock")
        conflict = Fluent("conflict")
        fin = Fluent("fin", _signature=[Parameter("a", self.agent_type)])

        new_problem.add_fluent(stage_1, default_initial_value=False)
        new_problem.add_fluent(stage_2, default_initial_value=False)
        new_problem.add_fluent(precondition_violation, default_initial_value=False)
        new_problem.add_fluent(possible_deadlock, default_initial_value=False)
        new_problem.add_fluent(conflict, default_initial_value=False)
        new_problem.add_fluent(fin, default_initial_value=False)

        allow_action_map = {}
        for agent in problem.agents:
            for action in agent.actions:
                action_fluent = Fluent("allow-" + agent.name + "-" + action.name)
                # allow_action_map.setdefault(action.agent, {}).update(action=action_fluent)
                if agent.name not in allow_action_map.keys():
                    allow_action_map[agent.name] = {action.name: action_fluent}
                else:
                    allow_action_map[agent.name][action.name] = action_fluent
                new_problem.add_fluent(action_fluent, default_initial_value=True)

        # Add actions
        for agent in problem.agents:
            for action in agent.actions:            
                # Success version - affects globals same way as original
                a_s = self.create_action_copy(problem, agent, action, "_s_" + agent.name)
                a_s.add_precondition(stage_1)
                a_s.add_precondition(allow_action_map[agent.name][action.name])
                for fact in self.get_action_preconditions(problem, agent, action, True, True):
                    a_s.add_precondition(self.global_fluent_map.get_correct_version(agent, fact))
                for effect in action.effects:
                    a_s.add_effect(self.global_fluent_map.get_correct_version(agent, effect.fluent), effect.value)
                new_problem.add_action(a_s)

                # Fail version
                for i, fact in enumerate(self.get_action_preconditions(problem, agent, action, True, False)):
                    a_f = self.create_action_copy(problem, agent, action, "_f_" + agent.name + "_" + str(i))
                    a_f.add_precondition(stage_1)
                    a_f.add_precondition(allow_action_map[agent.name][action.name])
                    for pre in self.get_action_preconditions(problem, agent, action, False, True):
                        a_f.add_precondition(self.global_fluent_map.get_correct_version(agent,pre))
                    a_f.add_precondition(Not(self.global_fluent_map.get_correct_version(agent,fact)))
                    a_f.add_effect(precondition_violation, True)
                    a_f.add_effect(stage_2, True)
                    a_f.add_effect(stage_1, False)

                    new_problem.add_action(a_f)

                for i, fact in enumerate(self.get_action_preconditions(problem, agent, action, False, True)):
                    # Wait version
                    a_w = self.create_action_copy(problem, agent, action, "_w_" + agent.name + "_" + str(i))
                    a_s.add_precondition(stage_1)
                    a_s.add_precondition(allow_action_map[agent.name][action.name])
                    a_w.add_precondition(Not(self.global_fluent_map.get_correct_version(agent,fact)))
                    assert not fact.is_not()
                    a_w.add_effect(self.waiting_fluent_map.get_correct_version(agent,fact), True)  # , action.agent.obj), True)
                    new_problem.add_action(a_w)

                    # deadlock version
                    a_deadlock = self.create_action_copy(problem, agent, action, "_deadlock_" + agent.name + "_" + str(i))
                    a_deadlock.add_precondition(Not(self.global_fluent_map.get_correct_version(agent,fact)))
                    for another_action in allow_action_map[agent.name].keys():
                        a_deadlock.add_precondition(Not(allow_action_map[agent.name][another_action]))
                    a_deadlock.add_effect(fin(self.get_agent_obj(agent)), True)
                    a_deadlock.add_effect(possible_deadlock, True)
                    new_problem.add_action(a_deadlock)
                
                # local version
                a_local = self.create_action_copy(problem, agent, action, "_local_" + agent.name)
                a_local.add_precondition(stage_2)
                a_local.add_precondition(allow_action_map[agent.name][action.name])
                for fluent in allow_action_map[agent.name].values():                    
                    a_local.add_effect(fluent, True)
                new_problem.add_action(a_local)

            #end-success        
            end_s = InstantaneousAction("end_s_" + agent.name)
            for goal in self.get_agent_goal(problem, agent):
                end_s.add_precondition(self.global_fluent_map.get_correct_version(agent, goal.args[0]))
                end_s.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, goal.args[0]))
            end_s.add_effect(fin(self.get_agent_obj(agent)), True)
            end_s.add_effect(stage_1, False)
            new_problem.add_action(end_s)

        # start-stage-2
        start_stage_2 = InstantaneousAction("start_stage_2")
        for agent in problem.agents:
            start_stage_2.add_precondition(fin(self.get_agent_obj(agent)))
        start_stage_2.add_effect(stage_2, True)
        start_stage_2.add_effect(stage_1, False)
        new_problem.add_action(start_stage_2)

        # goals_not_achieved
        goals_not_achieved = InstantaneousAction("goals_not_achieved")
        goals_not_achieved.add_precondition(stage_2)
        for agent in problem.agents:
            for i, goal in enumerate(self.get_agent_goal(problem, agent)):
                goals_not_achieved.add_precondition(Not(self.global_fluent_map.get_correct_version(agent, goal.args[0])))
                for g in self.get_agent_goal(problem, agent):
                    goals_not_achieved.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, g.args[0]))
        goals_not_achieved.add_effect(conflict, True)
        new_problem.add_action(goals_not_achieved)

        # declare_deadlock
        declare_deadlock = InstantaneousAction("declare_deadlock")
        declare_deadlock.add_precondition(stage_2)
        declare_deadlock.add_precondition(possible_deadlock)
        for agent in problem.agents:
            for i, goal in enumerate(self.get_agent_goal(problem, agent)):
                for g in self.get_agent_goal(problem, agent):
                    declare_deadlock.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, g.args[0]))
        declare_deadlock.add_effect(conflict, True)
        new_problem.add_action(declare_deadlock)

        # declare_fail
        declare_fail = InstantaneousAction("declare_fail")
        declare_fail.add_precondition(stage_2)
        declare_fail.add_precondition(precondition_violation)
        for agent in problem.agents:
            for i, goal in enumerate(self.get_agent_goal(problem, agent)):
                for g in self.get_agent_goal(problem, agent):
                    declare_fail.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, g.args[0]))
        declare_fail.add_effect(conflict, True)
        new_problem.add_action(declare_fail)
                

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
        new_problem.add_goal(conflict)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )        

class DurativeActionRobustnessVerifier(RobustnessVerifier):
    '''Robustness verifier class for durative actions:
    this class requires a (multi agent) problem, and creates a temporal planning problem which is unsolvable iff the multi agent problem is not robust.'''
    def __init__(self):
        RobustnessVerifier.__init__(self)
    
    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED_MULTI_AGENT")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_time("CONTINUOUS_TIME")        
        supported_kind.set_time("DURATION_INEQUALITIES")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= DurativeActionRobustnessVerifier.supported_kind()

    def get_action_conditions(self, problem : MultiAgentProblemWithWaitfor, agent : Agent, action : Action, fail : bool, wait: bool):
        c_start = []
        c_overall = []
        c_end = []
        assert fail or wait
        if wait and not fail:
            # Can only wait for start conditions
            return (problem.waitfor.get_preconditions_wait(agent.name, action.name), [], [])
        else:
            for interval, cl in action.conditions.items():
                for c in cl:
                    if interval.lower == interval.upper:
                        if interval.lower.is_from_start():
                            if wait or not c in problem.waitfor.get_preconditions_wait(agent.name, action.name):
                                c_start.append(c)
                        else:
                            if fail:
                                c_end.append(c)
                    else:
                        if not interval.is_left_open():
                            if wait or not c in problem.waitfor.get_preconditions_wait(agent.name, action.name):
                                c_start.append(c)
                        if fail:
                            c_overall.append(c)
                        if not interval.is_right_open():
                            if fail:
                                c_end.append(c)
        return (c_start, c_overall, c_end)


    def create_action_copy(self, problem: MultiAgentProblemWithWaitfor, agent : Agent , action : DurativeAction, prefix : str):
        """Create a new copy of an action, with name prefix_action_name, and duplicates the local preconditions/effects
        """
        d = {}
        for p in action.parameters:
            d[p.name] = p.type

        new_action = DurativeAction(prefix + "_" + agent.name + "_" + action.name, _parameters=d)     
        new_action.set_duration_constraint(action.duration)
        #new_action.add_condition(ClosedDurationInterval(StartTiming(), EndTiming()), self.act_pred)   

        # TODO: can probably do this better with a substitution walker
        for timing in action.conditions.keys():
            for fact in action.conditions[timing]:
                assert not fact.is_and()                
                new_action.add_condition(timing, self.local_fluent_map[agent].get_correct_version(agent, fact))

        for timing in action.effects.keys():
            for effect in action.effects[timing]:
                new_action.add_effect(timing, self.local_fluent_map[agent].get_correct_version(agent, effect.fluent), effect.value)

        return new_action

    def _compile(self, problem: "up.model.AbstractProblem", compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        '''Creates a the robustness verification problem.'''

        #Represents the map from the new action to the old action
        new_to_old: Dict[Action, Action] = {}
        
        new_problem = self.initialize_problem(problem)

        # Fluents
        waiting_fluent_map = {}
        for agent in problem.agents:
            waiting_fluent_map[agent] = FluentMap("w-" + agent.name, default_value=False)
            waiting_fluent_map[agent].add_facts(problem, new_problem)

        inv_count_map = FluentMap("i", default_value=0, override_type=IntType(0))
        inv_count_map.add_facts(problem, new_problem)



        failure = Fluent("failure")
        act = Fluent("act")
        fin = Fluent("fin", _signature=[Parameter("a", self.agent_type)])
        waiting = Fluent("waiting", _signature=[Parameter("a", self.agent_type)])

        new_problem.add_fluent(failure, default_initial_value=False)
        new_problem.add_fluent(act, default_initial_value=True)
        new_problem.add_fluent(fin, default_initial_value=False)
        new_problem.add_fluent(waiting, default_initial_value=False)

        # Actions

        for agent in problem.agents:
            # Create end_s_i action
            end_s_action = InstantaneousAction("end_s_" + agent.name)
            end_s_action.add_precondition(Not(fin(self.get_agent_obj(agent))))
            for g in self.get_agent_goal(problem, agent):
                end_s_action.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, g.args[0]))
                end_s_action.add_precondition(self.global_fluent_map.get_correct_version(agent, g.args[0]))
            end_s_action.add_effect(fin(self.get_agent_obj(agent)), True)
            end_s_action.add_effect(act, False)
            new_problem.add_action(end_s_action)

            # Create end_f_i action
            for j, gf in enumerate(self.get_agent_goal(problem, agent)):
                end_f_action = InstantaneousAction("end_f_" + agent.name + "_" + str(j))
                end_f_action.add_precondition(Not(self.global_fluent_map.get_correct_version(agent, gf.args[0])))
                end_f_action.add_precondition(Not(fin(self.get_agent_obj(agent))))
                for g in self.get_agent_goal(problem, agent):
                    end_f_action.add_precondition(self.local_fluent_map[agent].get_correct_version(agent, g.args[0]))
                end_f_action.add_effect(fin(self.get_agent_obj(agent)), True)
                end_f_action.add_effect(failure, True)
                end_f_action.add_effect(act, False)
                new_problem.add_action(end_f_action)


            for action in agent.actions:                
                c_start, c_overall, c_end = self.get_action_conditions(problem, agent, action, fail=True, wait=True)
                w_start, w_overall, w_end = self.get_action_conditions(problem, agent, action, fail=False, wait=True)
                f_start, f_overall, f_end = self.get_action_conditions(problem, agent, action, fail=True, wait=False)
                assert(c_overall == f_overall and c_end == f_end and w_overall == [] and w_end == [])


                a_s = self.create_action_copy(problem, agent, action, "s")
                a_s.add_condition(StartTiming(), Not(waiting(self.get_agent_obj(agent))))
                a_s.add_condition(ClosedDurationInterval(StartTiming(), EndTiming()), act())   
                # Conditions/Effects on global copy
                for timing in action.conditions.keys():
                    for fact in action.conditions[timing]:
                        assert not fact.is_and()                
                        a_s.add_condition(timing, self.global_fluent_map.get_correct_version(agent, fact))
                for timing in action.effects.keys():
                    for effect in action.effects[timing]:
                        a_s.add_effect(timing, self.global_fluent_map.get_correct_version(agent, effect.fluent), effect.value)
                # accounting for invariant count
                for c in c_overall:
                    a_s.add_increase_effect(StartTiming(), inv_count_map.get_correct_version(agent, c), 1)
                    a_s.add_decrease_effect(EndTiming(), inv_count_map.get_correct_version(agent, c), 1)
                for effect in action.effects.get(StartTiming(), []):
                    if effect.value.is_false():
                        a_s.add_condition(StartTiming(), Equals(inv_count_map.get_correct_version(agent, effect.fluent), 0))
                for effect in action.effects.get(EndTiming(), []):
                    if effect.value.is_false():
                        a_s.add_condition(EndTiming(), Equals(inv_count_map.get_correct_version(agent, effect.fluent), 1))
                # accouting for other agents waiting
                for effect in action.effects.get(StartTiming(), []):
                    if effect.value.is_true():
                        for ag in problem.agents:
                            a_s.add_condition(StartTiming(), Not(waiting_fluent_map[ag].get_correct_version(agent, effect.fluent)))
                for effect in action.effects.get(EndTiming(), []):
                    if effect.value.is_true():
                        for ag in problem.agents:
                            a_s.add_condition(ClosedDurationInterval(StartTiming(), EndTiming()), Not(waiting_fluent_map[ag].get_correct_version(agent, effect.fluent)))
                new_problem.add_action(a_s)

                # Fail start version            
                for i, fact in enumerate(f_start):
                    a_fstart = self.create_action_copy(problem, agent, action, "f_start_" + str(i))
                    for c in w_start:
                        a_fstart.add_condition(StartTiming(), self.global_fluent_map.get_correct_version(agent, c))
                    a_fstart.add_condition(StartTiming(), Not(self.global_fluent_map.get_correct_version(agent, fact)))
                    a_fstart.add_condition(StartTiming(), Not(waiting(self.get_agent_obj(agent))))
                    a_fstart.add_effect(StartTiming(), failure, True)
                    new_problem.add_action(a_fstart)

                # Fail inv version            
                for i, fact in enumerate(f_overall):
                    overall_condition_added_by_start_effect = False
                    for effect in action.effects.get(StartTiming(), []):
                        if effect.fluent == fact and effect.value.is_true():
                            overall_condition_added_by_start_effect = True
                            break
                    if not overall_condition_added_by_start_effect:
                        a_finv = self.create_action_copy(problem, agent, action, "f_inv_" + str(i))
                        for c in c_start:
                            a_fstart.add_condition(StartTiming(), self.global_fluent_map.get_correct_version(agent, c))
                        a_finv.add_condition(StartTiming(), Not(self.global_fluent_map.get_correct_version(agent, fact)))
                        for effect in action.effects.get(StartTiming(), []):
                            a_finv.add_effect(StartTiming(), self.global_fluent_map.get_correct_version(agent, effect.fluent), effect.value)
                            if effect.value.is_false():
                                a_finv.add_condition(StartTiming(), Equals(inv_count_map.get_correct_version(agent, effect.fluent), 0))
                            if effect.value.is_true():
                                for ag in problem.agents:
                                    a_finv.add_condition(StartTiming(),
                                                        Not(waiting_fluent_map[ag].get_correct_version(agent, effect.fluent)))
                        a_finv.add_condition(StartTiming(), Not(waiting(self.get_agent_obj(agent))))
                        a_finv.add_effect(StartTiming(), failure, True)
                        new_problem.add_action(a_finv)

                # Fail end version            
                for i, fact in enumerate(f_end):
                    a_fend = self.create_action_copy(problem, agent, action, "f_end_" + str(i))
                    for c in c_start:
                        a_fend.add_condition(StartTiming(), self.global_fluent_map.get_correct_version(agent, c))
                    for c in c_overall:
                        a_fend.add_condition(OpenDurationInterval(StartTiming(), EndTiming()), self.global_fluent_map.get_correct_version(agent, c))
                    a_fend.add_condition(StartTiming(), Not(waiting(self.get_agent_obj(agent))))
                    a_fend.add_condition(EndTiming(), Not(self.global_fluent_map.get_correct_version(agent, fact)))
                    for effect in action.effects.get(StartTiming(), []):
                        a_fend.add_effect(StartTiming(), self.global_fluent_map.get_correct_version(agent, effect.fluent), effect.value)
                        if effect.value.is_false():
                            a_fend.add_condition(StartTiming(), Equals(inv_count_map.get_correct_version(agent, effect.fluent), 0))                                
                        if effect.value.is_true():
                            for agent in problem.agents:
                                for ag in problem.agents:
                                    a_fend.add_condition(StartTiming(),
                                                    Not(waiting_fluent_map[ag].get_correct_version(agent, effect.fluent)))
                    for effect in action.effects.get(EndTiming(), []):
                        # if effect.value.is_false():
                        #    a_fend.add_condition(StartTiming(), Equals(self.get_inv_count_version(effect.fluent), 0))
                        if effect.value.is_true():
                            for ag in problem.agents:
                                # Changed timing from start to end
                                a_fend.add_condition(EndTiming(), Not(waiting_fluent_map[ag].get_correct_version(agent, effect.fluent)))
                    a_fend.add_condition(StartTiming(), Not(waiting(self.get_agent_obj(agent))))
                    a_fend.add_effect(EndTiming(), failure, True)
                    for c in c_overall:
                        a_s.add_increase_effect(StartTiming(), inv_count_map.get_correct_version(agent, c), 1)
                    new_problem.add_action(a_fend)

                # Del inv start version            
                for i, effect in enumerate(action.effects.get(StartTiming(), [])):
                    if effect.value.is_false():
                        a_finvstart = self.create_action_copy(problem, agent, action, "f_inv_start_" + str(i))
                        a_finvstart.add_condition(StartTiming(), Not(waiting(self.get_agent_obj(agent))))
                        for c in c_start:
                            a_finvstart.add_condition(StartTiming(), self.global_fluent_map.get_correct_version(agent, c))
                        a_finvstart.add_condition(StartTiming(), GT(inv_count_map.get_correct_version(agent, effect.fluent), 0))
                        a_finvstart.add_effect(StartTiming(), failure, True)
                        new_problem.add_action(a_finvstart)

                # Del inv end version            
                for i, effect in enumerate(action.effects.get(EndTiming(), [])):
                    if effect.value.is_false():
                        a_finvend = self.create_action_copy(problem, agent, action, "f_inv_end_" + str(i))
                        a_finvend.add_condition(StartTiming(), Not(waiting(self.get_agent_obj(agent))))
                        for c in c_start:
                            a_finvend.add_condition(StartTiming(), self.global_fluent_map.get_correct_version(agent, c))
                        for c in c_overall:
                            a_finvend.add_condition(OpenDurationInterval(StartTiming(), EndTiming()),
                                                    self.global_fluent_map.get_correct_version(agent, c))
                        for c in c_end:
                            a_finvend.add_condition(EndTiming(), self.global_fluent_map.get_correct_version(agent, c))
                        a_finvend.add_condition(EndTiming(), GT(inv_count_map.get_correct_version(agent, effect.fluent), 0))
                        
                        for seffect in action.effects.get(StartTiming(), []):
                            a_finvend.add_effect(StartTiming(), self.global_fluent_map.get_correct_version(agent, seffect.fluent), seffect.value)
                            if seffect.value.is_false():
                                a_finvstart.add_condition(StartTiming(), Equals(inv_count_map.get_correct_version(agent, seffect.fluent), 0))                                
                            if seffect.value.is_true():
                                for ag in problem.agents:
                                    a_finvend.add_condition(StartTiming(),
                                                            Not(waiting_fluent_map[ag].get_correct_version(agent, seffect.fluent)))
                                    a_finvend.add_condition(OpenDurationInterval(StartTiming(), EndTiming()),
                                                            Not(waiting_fluent_map[ag].get_correct_version(agent, seffect.fluent)))
                        for seffect in action.effects.get(EndTiming(), []):
                            a_finvend.add_effect(EndTiming(), self.global_fluent_map.get_correct_version(agent, seffect.fluent), seffect.value)

                        # self.add_condition_inv_count_zero(effect.fluent, a_finvend, StartTiming(), True, 0)
                        a_finvend.add_effect(StartTiming(), failure, True)
                        for interval, condition in action.conditions.items():
                            if interval.lower != interval.upper:
                                for fact in condition:
                                    a_finvend.add_increase_effect(StartTiming(), inv_count_map.get_correct_version(agent, fact), 1)                                    
                        new_problem.add_action(a_finvend)
                
                # a^w_x version - wait forever for x to be true
                for i, w_fact in enumerate(w_start):
                    a_wx = self.create_action_copy(problem, agent, action, "w_" + str(i))
                    a_wx.add_condition(StartTiming(), Not(waiting(self.get_agent_obj(agent))))

                    a_wx.add_effect(StartTiming(), failure, True)
                    a_wx.add_effect(StartTiming(), waiting(self.get_agent_obj(agent)), True)
                    a_wx.add_condition(StartTiming(), Not(self.global_fluent_map.get_correct_version(agent, w_fact)))
                    a_wx.add_effect(StartTiming(), waiting_fluent_map[agent].get_correct_version(agent, w_fact), True)
                    new_problem.add_action(a_wx)

                # a_waiting version - dummy version while agent is waiting
                a_waiting = self.create_action_copy(problem, agent, action, "waiting")
                a_waiting.add_condition(StartTiming(), waiting(self.get_agent_obj(agent)))
                new_problem.add_action(a_waiting)





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


        w = PDDLWriter(new_problem)
        w.write_domain("domain.pddl")
        w.write_problem("problem.pddl")

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )  


env = up.environment.get_env()
env.factory.add_engine('SimpleInstantaneousActionRobustnessVerifier', __name__, 'SimpleInstantaneousActionRobustnessVerifier')
env.factory.add_engine('WaitingActionRobustnessVerifier', __name__, 'WaitingActionRobustnessVerifier')
env.factory.add_engine('DurativeActionRobustnessVerifier', __name__, 'DurativeActionRobustnessVerifier')