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
"""This module defines the multi agent problem with waitfor specification."""

import unified_planning.environment
from unified_planning.model.expression import ConstantExpression
from unified_planning.social_law.waitfor_specification import WaitforSpecification
from unified_planning.model.multi_agent.ma_problem import MultiAgentProblem
from unified_planning.model.multi_agent.agent import Agent
from typing import Dict


class MultiAgentProblemWithWaitfor(MultiAgentProblem):
    """ Represents a multi-agent problem with waitfor conditions"""

    def __init__(
        self,        
        name: str = None,
        env: "unified_planning.environment.Environment" = None,
        *,
        initial_defaults: Dict["unified_planning.model.types.Type", "ConstantExpression"] = {},
        waitfor : WaitforSpecification = None
    ):
        MultiAgentProblem.__init__(self, name=name, env=env, initial_defaults=initial_defaults)
        if waitfor is None:
            waitfor = WaitforSpecification()
        self._waitfor = waitfor

    @property
    def waitfor(self) -> WaitforSpecification:
        return self._waitfor

    def __repr__(self) -> str:
        return MultiAgentProblem.__repr__(self) + "\n" + "waitfor: " + str(self.waitfor)

    def __eq__(self, oth: object) -> bool:
        if not (isinstance(oth, MultiAgentProblem)) or self._env != oth._env:
            return False
        return MultiAgentProblem.__eq__(self, oth) and self.waitfor == oth.waitfor

    def __hash__(self) -> int:
        return MultiAgentProblem.__hash__(self) + hash(self.waitfor)     

    def clone(self):
        new_p = MultiAgentProblemWithWaitfor(self._name, self._env)
        for f in self.ma_environment.fluents:
            new_p.ma_environment.add_fluent(f)
        for ag in self.agents:
            new_ag = Agent(ag.name, self)
            for f in ag.fluents:
                new_ag.add_fluent(f)
            for a in ag.actions:
                new_ag.add_action(a.clone())
        new_p._user_types = self._user_types[:]
        new_p._user_types_hierarchy = self._user_types_hierarchy.copy()
        new_p._objects = self._objects[:]
        new_p._initial_value = self._initial_value.copy()
        new_p._goals = self._goals[:]
        new_p._initial_defaults = self._initial_defaults.copy()
        new_p._waitfor = self.waitfor.copy()
        return new_p   