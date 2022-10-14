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

from unified_planning.social_law.waitfor_specification import WaitforSpecification
from unified_planning.model.multi_agent.ma_problem import MultiAgentProblem
from typing import Dict


class MultiAgentProblemWithWaitfor(MultiAgentProblem):
    """ Represents a multi-agent problem with waitfor conditions"""

    def __init__(
        self,        
        name: str = None,
        env: "up.environment.Environment" = None,
        *,
        initial_defaults: Dict["up.model.types.Type", "ConstantExpression"] = {},
        waitfor : WaitforSpecification = None
    ):
        MultiAgentProblem.__init__(self, name=name, env=env, initial_defaults=initial_defaults)
        if waitfor is None:
            waitfor = WaitforSpecification()
        self._waitfor = waitfor


    @property
    def waitfor(self) -> WaitforSpecification:
        return self._waitfor