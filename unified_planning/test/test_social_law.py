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
from unified_planning.social_law.robustness_verification import InstantaneousActionRobustnessVerifier
from unified_planning.social_law.waitfor_specification import WaitforSpecification
from unified_planning.model.multi_agent import *
from unified_planning.io import PDDLWriter
from unified_planning.engines import PlanGenerationResultStatus

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



class TestProblem(TestCase):
    def setUp(self):
        TestCase.setUp(self)        

    def test_intersection_single_agent_projects(self):
        problem = get_intersection_problem().problem

        sap = SingleAgentProjection(problem.agents[0])        
        result = sap.compile(problem)

        with OneshotPlanner(name='fast-downward') as planner:
            result = planner.solve(result.problem)
            self.assertTrue(result.status in POSITIVE_OUTCOMES)
        
    def test_intersection_robustness_verification_4cars_robust(self):
        problem = get_intersection_problem(yields_list=[("south-ent", "cross-ne"),("north-ent", "cross-sw"),("east-ent", "cross-nw"),("west-ent", "cross-se")]).problem

        wfr = WaitforSpecification()
        for agent in problem.agents:
            drive = agent.action("drive")    
            wfr.annotate_as_waitfor(agent.name, drive.name, drive.preconditions[1])        

        rbv = InstantaneousActionRobustnessVerifier()
        rbv.waitfor_specification = wfr
        rbv_result = rbv.compile(problem)

        with OneshotPlanner(name='fast-downward') as planner:
            result = planner.solve(rbv_result.problem)
            self.assertTrue(result.status in UNSOLVABLE_OUTCOMES)        

    def test_intersection_robustness_verification_4cars_yield_deadlock(self):
        problem = get_intersection_problem(yields_list=[("south-ent", "east-ent"),("east-ent", "north-ent"),("north-ent", "west-ent"),("west-ent", "south-ent")]).problem

        wfr = WaitforSpecification()
        for agent in problem.agents:
            drive = agent.action("drive")    
            wfr.annotate_as_waitfor(agent.name, drive.name, drive.preconditions[1])        

        rbv = InstantaneousActionRobustnessVerifier()
        rbv.waitfor_specification = wfr
        rbv_result = rbv.compile(problem)

        with OneshotPlanner(name='fast-downward') as planner:
            result = planner.solve(rbv_result.problem)
            self.assertTrue(result.status in POSITIVE_OUTCOMES)       


    def test_intersection_robustness_verification_4cars_deadlock(self):
        problem = get_intersection_problem().problem

        wfr = WaitforSpecification()
        for agent in problem.agents:
            drive = agent.action("drive")    
            wfr.annotate_as_waitfor(agent.name, drive.name, drive.preconditions[1])        

        rbv = InstantaneousActionRobustnessVerifier()
        rbv.waitfor_specification = wfr
        rbv_result = rbv.compile(problem)

        with OneshotPlanner(name='fast-downward') as planner:
            result = planner.solve(rbv_result.problem)
            self.assertTrue(result.status in POSITIVE_OUTCOMES)
        
    def test_intersection_robustness_verification_4cars_crash(self):
        problem = get_intersection_problem().problem

        rbv = InstantaneousActionRobustnessVerifier()
        rbv_result = rbv.compile(problem)

        with OneshotPlanner(name='fast-downward') as planner:
            result = planner.solve(rbv_result.problem)
            self.assertTrue(result.status in POSITIVE_OUTCOMES)

    def test_intersection_robustness_verification_2cars_crash(self):
        problem = get_intersection_problem(["car-north", "car-east"]).problem
  
        rbv = InstantaneousActionRobustnessVerifier()
        rbv_result = rbv.compile(problem)

        with OneshotPlanner(name='fast-downward') as planner:
            result = planner.solve(rbv_result.problem)
            self.assertTrue(result.status in POSITIVE_OUTCOMES)            
        
    def test_intersection_robustness_verification_2cars_robust(self):
        problem = get_intersection_problem(["car-north", "car-south"]).problem
  
        rbv = InstantaneousActionRobustnessVerifier()
        rbv_result = rbv.compile(problem)

        with OneshotPlanner(name='fast-downward') as planner:
            result = planner.solve(rbv_result.problem)
            self.assertTrue(result.status in UNSOLVABLE_OUTCOMES)
