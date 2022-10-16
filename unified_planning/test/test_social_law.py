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
from unified_planning.social_law.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from unified_planning.model.multi_agent.ma_centralizer import MultiAgentProblemCentralizer
from unified_planning.social_law.social_law import SocialLaw, SocialLawRobustnessChecker, SocialLawRobustnessStatus
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

class RobustnessTestCase:
    def __init__(self, name, 
                    expected_outcome : SocialLawRobustnessStatus, 
                    cars = ["car-north", "car-south", "car-east", "car-west"], 
                    yields_list = [], 
                    wait_drive = True):
        self.name = name
        self.cars = cars
        self.yields_list = yields_list
        self.expected_outcome = expected_outcome
        self.wait_drive = wait_drive        


class TestProblem(TestCase):
    def setUp(self):
        TestCase.setUp(self)        
        self.test_cases = [         
            RobustnessTestCase("4cars_crash", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL, yields_list=[], wait_drive=False),   
            RobustnessTestCase("4cars_deadlock", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK, yields_list=[]),
            RobustnessTestCase("4cars_yield_deadlock", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK, yields_list=[("south-ent", "east-ent"),("east-ent", "north-ent"),("north-ent", "west-ent"),("west-ent", "south-ent")]),
            RobustnessTestCase("4cars_robust", SocialLawRobustnessStatus.ROBUST_RATIONAL, yields_list=[("south-ent", "cross-ne"),("north-ent", "cross-sw"),("east-ent", "cross-nw"),("west-ent", "cross-se")]),
            RobustnessTestCase("2cars_crash", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL, cars=["car-north", "car-east"], yields_list=[], wait_drive=False),   
            RobustnessTestCase("2cars_robust", SocialLawRobustnessStatus.ROBUST_RATIONAL, cars=["car-north", "car-south"], yields_list=[], wait_drive=False)            
        ]

    def test_social_law(self):
        slrc = SocialLawRobustnessChecker(
            planner_name="fast-downward",
            robustness_verifier_name="SimpleInstantaneousActionRobustnessVerifier"
            )
        p_4cars_crash = get_intersection_problem(wait_drive=False).problem
        l = SocialLaw()
        for agent in p_4cars_crash.agents:
            l.add_waitfor_annotation(agent.name, "drive", "free", ["l2"])
        
        res = l.compile(p_4cars_crash)
        p_4cars_deadlock = res.problem
        self.assertEqual(len(p_4cars_crash.waitfor.waitfor_map), 0)
        self.assertEqual(len(p_4cars_deadlock.waitfor.waitfor_map), 4)

        r_result = slrc.is_robust(p_4cars_deadlock)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK)
        r_result = slrc.is_robust(p_4cars_crash)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL)        

        l2 = SocialLaw()
        l2.disallow_action("car-north", "drive", ["south-ent", "cross-se", "north"])
        res = l2.compile(p_4cars_crash)
        p_nosap = res.problem
        r_result = slrc.is_robust(p_nosap)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT)

    def test_all_cases(self):
        for t in self.test_cases:
            problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=False).problem
            slrc = SocialLawRobustnessChecker(
                planner_name="fast-downward",
                robustness_verifier_name="SimpleInstantaneousActionRobustnessVerifier"
                )
            r_result = slrc.is_robust(problem)
            self.assertEqual(r_result.status, t.expected_outcome, t.name)
            if t.expected_outcome == SocialLawRobustnessStatus.ROBUST_RATIONAL:
                presult = slrc.solve(problem)
                self.assertIn(presult.status, POSITIVE_OUTCOMES, t.name)

    def test_all_cases_waiting(self):
        for t in self.test_cases:
            problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=False).problem
            slrc = SocialLawRobustnessChecker(
                planner_name="fast-downward",
                robustness_verifier_name="WaitingActionRobustnessVerifier"
                )
            r_result = slrc.is_robust(problem)
            self.assertEqual(r_result.status, t.expected_outcome, t.name)
            if t.expected_outcome == SocialLawRobustnessStatus.ROBUST_RATIONAL:
                presult = slrc.solve(problem)
                self.assertIn(presult.status, POSITIVE_OUTCOMES, t.name)


    def test_centralizer(self):
        for t in self.test_cases:
            for durative in [False]:# True]:
                problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=durative).problem
                mac = MultiAgentProblemCentralizer()
                cresult = mac.compile(problem)
                with OneshotPlanner(problem_kind=cresult.problem.kind) as planner:
                    presult = planner.solve(cresult.problem)
                    self.assertIn(presult.status, POSITIVE_OUTCOMES, t.name)

        
    def test_all_cases_durative(self):
        for t in self.test_cases:
            problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=True).problem
            with open("waitfor.json", "w") as f:
                f.write(str(problem.waitfor))

            slrc = SocialLawRobustnessChecker(                                
                save_pddl=True,                
                )
            self.assertEqual(slrc.is_robust(problem).status, t.expected_outcome, t.name)

            
