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
"""This module defines the waitfor specification class, which enables managing/reading/writing the specification of the waitfor precondtions."""

from unified_planning.model.action import Action
from unified_planning.model.fnode import FNode
from unified_planning.model.multi_agent.agent import Agent
from typing import Dict, List
import json
import copy

class WaitforSpecification:
    """ This is the waitfor specification class, which enables managing/reading/writing the specification of the waitfor precondtions"""

    def __init__(self):
        self.waitfor_map = {}

    def annotate_as_waitfor(self, agent_name: str, action_name : str, precondition : FNode):
        if not (agent_name, action_name) in self.waitfor_map:
            self.waitfor_map[(agent_name, action_name)] = []    
        self.waitfor_map[(agent_name, action_name)].append(precondition)

    def get_preconditions_wait(self,  agent_name: str, action_name : str) -> List[FNode]:
        if (agent_name, action_name) in self.waitfor_map:
            return self.waitfor_map[(agent_name, action_name)]
        else:
            return []

    def __repr__(self):
        str_dict = {}
        for x, y in self.waitfor_map.items():
            str_dict[x] = list(map(str, y))
        return str(str_dict)

    def __eq__(self, oth: object) -> bool:
        if not isinstance(oth, WaitforSpecification):
            return False
        return self.waitfor_map == oth.waitfor_map

    def __hash__(self) -> int:
        return hash(self.waitfor_map)

    def clone(self):
        new_w = WaitforSpecification()
        for agent_name, action_name in self.waitfor_map:
            new_w.waitfor_map[(agent_name, action_name)] = self.waitfor_map[(agent_name, action_name)][:]
        return new_w



