# Copyright 2022 Technion project
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


import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.test import TestCase, main
from unified_planning.test.examples.multi_agent import get_example_problems, get_intersection_problem
from unified_planning.social_law.single_agent_projection import SingleAgentProjection
from unified_planning.social_law.robustness_verification import RobustnessVerifier, SimpleInstantaneousActionRobustnessVerifier, WaitingActionRobustnessVerifier
from unified_planning.social_law.waitfor_specification import WaitforSpecification
from unified_planning.model.multi_agent import *
from unified_planning.io import PDDLWriter
from unified_planning.engines import PlanGenerationResultStatus
from collections import namedtuple

POSITIVE_OUTCOMES = frozenset(
    [
        PlanGenerationResultStatus.SOLVED_SATISFICING,
        PlanGenerationResultStatus.SOLVED_OPTIMALLY,
    ]
)

UNSOLVABLE_OUTCOMES = frozenset(
    [
        PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
        PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
    ]
)

RobustnessTestCase = namedtuple("RobustnessTestCase", ["name", "cars", "yields_list", "expected_outcome_set"])
class RobustnessTestCase:
    def __init__(self, name, cars = ["car-north", "car-south", "car-east", "car-west"], yields_list = [], wait_drive = True, expected_outcome_set = []):
        self.name = name
        self.cars = cars
        self.yields_list = yields_list
        self.expected_outcome_set = expected_outcome_set
        self.wait_drive = wait_drive

    def run_test(self, rbv : RobustnessVerifier, test : TestCase):
        problem = get_intersection_problem(self.cars, self.yields_list).problem

        wfr = WaitforSpecification()
        for agent in problem.agents:
            drive = agent.action("drive")
            if self.wait_drive:
                wfr.annotate_as_waitfor(agent.name, drive.name, drive.preconditions[1])
            if len(self.yields_list) > 0:
                wfr.annotate_as_waitfor(agent.name, drive.name, drive.preconditions[5])        
        
        rbv.waitfor_specification = wfr
        rbv_result = rbv.compile(problem)

        w = PDDLWriter(rbv_result.problem)
        w.write_domain(rbv.name + "_" + self.name + "_domain.pddl")
        w.write_problem(rbv.name + "_" + self.name + "_problem.pddl")

        f = open(rbv.name + "_" + self.name + "_waitfor.json", "w")
        f.write(str(rbv.waitfor_specification))
        f.close()

        with OneshotPlanner(name='fast-downward') as planner:
            result = planner.solve(rbv_result.problem)
            test.assertTrue(result.status in self.expected_outcome_set, self.name)


class TestProblem(TestCase):
    def setUp(self):
        TestCase.setUp(self)        
        self.test_cases = [         
            RobustnessTestCase("4cars_crash", yields_list=[], wait_drive=False, expected_outcome_set=POSITIVE_OUTCOMES),   
            RobustnessTestCase("4cars_deadlock", yields_list=[], expected_outcome_set=POSITIVE_OUTCOMES),
            RobustnessTestCase("4cars_yield_deadlock", yields_list=[("south-ent", "east-ent"),("east-ent", "north-ent"),("north-ent", "west-ent"),("west-ent", "south-ent")], expected_outcome_set=POSITIVE_OUTCOMES),
            RobustnessTestCase("4cars_robust", yields_list=[("south-ent", "cross-ne"),("north-ent", "cross-sw"),("east-ent", "cross-nw"),("west-ent", "cross-se")], expected_outcome_set=UNSOLVABLE_OUTCOMES),
            RobustnessTestCase("2cars_crash", cars=["car-north", "car-east"], yields_list=[], wait_drive=False, expected_outcome_set=POSITIVE_OUTCOMES),   
            RobustnessTestCase("2cars_robust", cars=["car-north", "car-south"], yields_list=[], wait_drive=False, expected_outcome_set=UNSOLVABLE_OUTCOMES)            
        ]

    def test_all_cases(self):
        for t in self.test_cases:
            rbv = SimpleInstantaneousActionRobustnessVerifier()
            t.run_test(rbv, self)
            # wrbv = WaitingActionRobustnessVerifier()
            # t.run_test(wrbv, self)


    def test_intersection_single_agent_projection(self):
        problem = get_intersection_problem().problem

        for agent in problem.agents:
            sap = SingleAgentProjection(agent)        
            result = sap.compile(problem)

            with OneshotPlanner(name='fast-downward') as planner:
                result = planner.solve(result.problem)
                self.assertTrue(result.status in POSITIVE_OUTCOMES)

