# Copyright 2021 AIPlan4EU project
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
from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
from unified_planning.social_law.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from collections import namedtuple

Example = namedtuple("Example", ["problem", "plan"])


def get_example_problems():
    problems = {}

    # basic multi agent
    problem = MultiAgentProblem("ma-basic")

    Location = UserType("Location")

    is_connected = Fluent("is_connected", BoolType(), l1=Location, l2=Location)
    problem.ma_environment.add_fluent(is_connected, default_initial_value=False)

    r = Agent("robot", problem)
    pos = Fluent("pos", Location)
    r.add_fluent(pos)
    move = InstantaneousAction("move", l_from=Location, l_to=Location)
    l_from = move.parameter("l_from")
    l_to = move.parameter("l_to")
    move.add_precondition(Equals(pos, l_from))
    move.add_precondition(is_connected(l_from, l_to))
    move.add_effect(pos, l_to)
    r.add_action(move)
    problem.add_agent(r)

    l1 = Object("l1", Location)
    l2 = Object("l2", Location)
    problem.add_objects([l1, l2])

    problem.set_initial_value(is_connected(l1, l2), True)
    problem.set_initial_value(Dot(r, pos), l1)
    problem.add_goal(Equals(Dot(r, pos), l2))

    plan = up.plans.SequentialPlan(
        [up.plans.ActionInstance(move, (ObjectExp(l1), ObjectExp(l2)), r)]
    )

    basic = Example(problem=problem, plan=plan)
    problems["ma-basic"] = basic

    # Loader multi agent
    problem = MultiAgentProblem("ma-loader")

    Location = UserType("Location")

    is_connected = Fluent("is_connected", BoolType(), l1=Location, l2=Location)
    cargo_at = Fluent("cargo_at", BoolType(), position=Location)
    problem.ma_environment.add_fluent(is_connected, default_initial_value=False)
    problem.ma_environment.add_fluent(cargo_at, default_initial_value=False)

    robot1 = Agent("robot1", problem)
    robot2 = Agent("robot2", problem)
    pos = Fluent("pos", Location)

    cargo_mounted = Fluent("cargo_mounted")
    robot1.add_fluent(pos)
    robot1.add_fluent(cargo_mounted)
    robot2.add_fluent(pos)
    robot2.add_fluent(cargo_mounted)

    move = InstantaneousAction("move", l_from=Location, l_to=Location)
    l_from = move.parameter("l_from")
    l_to = move.parameter("l_to")
    move.add_precondition(Equals(pos, l_from))
    move.add_precondition(is_connected(l_from, l_to))
    move.add_effect(pos, l_to)

    load = InstantaneousAction("load", loc=Location)
    loc = load.parameter("loc")
    load.add_precondition(cargo_at(loc))
    load.add_precondition(Equals(pos, loc))
    load.add_precondition(Not(cargo_mounted))
    load.add_effect(cargo_at(loc), False)
    load.add_effect(cargo_mounted, True)

    unload = InstantaneousAction("unload", loc=Location)
    loc = unload.parameter("loc")
    unload.add_precondition(Not(cargo_at(loc)))
    unload.add_precondition(Equals(pos, loc))
    unload.add_precondition(cargo_mounted)
    unload.add_effect(cargo_at(loc), True)
    unload.add_effect(cargo_mounted, False)

    robot1.add_action(move)
    robot2.add_action(move)
    robot1.add_action(load)
    robot2.add_action(load)
    robot1.add_action(unload)
    robot2.add_action(unload)
    problem.add_agent(robot1)
    problem.add_agent(robot2)

    l1 = Object("l1", Location)
    l2 = Object("l2", Location)
    l3 = Object("l3", Location)
    problem.add_objects([l1, l2, l3])

    problem.set_initial_value(is_connected(l1, l2), True)
    problem.set_initial_value(is_connected(l2, l1), True)
    problem.set_initial_value(is_connected(l2, l3), True)
    problem.set_initial_value(Dot(robot1, pos), l2)
    problem.set_initial_value(Dot(robot2, pos), l2)
    problem.set_initial_value(cargo_at(l1), True)
    problem.set_initial_value(cargo_at(l2), False)
    problem.set_initial_value(cargo_at(l3), False)
    problem.set_initial_value(Dot(robot1, cargo_mounted), False)
    problem.set_initial_value(Dot(robot2, cargo_mounted), False)

    problem.add_goal(cargo_at(l3))

    plan = up.plans.SequentialPlan(
        [
            up.plans.ActionInstance(move, (ObjectExp(l2), ObjectExp(l1)), robot1),
            up.plans.ActionInstance(load, (ObjectExp(l1),), robot1),
            up.plans.ActionInstance(move, (ObjectExp(l1), ObjectExp(l2)), robot1),
            up.plans.ActionInstance(unload, (ObjectExp(l2),), robot1),
            up.plans.ActionInstance(load, (ObjectExp(l2),), robot2),
            up.plans.ActionInstance(move, (ObjectExp(l2), ObjectExp(l3)), robot2),
            up.plans.ActionInstance(unload, (ObjectExp(l3),), robot2),
        ]
    )
    ma_loader = Example(problem=problem, plan=plan)
    problems["ma-loader"] = ma_loader    

    return problems

def get_intersection_problem(
    cars = ["car-north", "car-south", "car-east", "car-west"], 
    yields_list = [], 
    wait_drive = True,
    durative = False) -> MultiAgentProblemWithWaitfor:
    # intersection multi agent
    problem = MultiAgentProblemWithWaitfor("intersection")

    loc = UserType("loc")
    direction = UserType("direction")
    car = UserType("car")

    # Environment     
    connected = Fluent('connected', BoolType(), l1=loc, l2=loc, d=direction)
    free = Fluent('free', BoolType(), l=loc)
    if len(yields_list) > 0:
        yieldsto = Fluent('yieldsto', BoolType(), l1=loc, l2=loc)
        problem.ma_environment.add_fluent(yieldsto, default_initial_value=False)
        dummy_loc = unified_planning.model.Object("dummy", loc)
        problem.add_object(dummy_loc)
    
    problem.ma_environment.add_fluent(connected, default_initial_value=False)
    problem.ma_environment.add_fluent(free, default_initial_value=True)

    intersection_map = {
        "north": ["south-ent", "cross-se", "cross-ne", "north-ex"],
        "south": ["north-ent", "cross-nw", "cross-sw", "south-ex"],
        "west": ["east-ent", "cross-ne", "cross-nw", "west-ex"],
        "east": ["west-ent", "cross-sw", "cross-se", "east-ex"]
    }

    location_names = set()
    for l in intersection_map.values():
        location_names = location_names.union(l)
    locations = list(map(lambda l: unified_planning.model.Object(l, loc), location_names))
    problem.add_objects(locations)

    direction_names = intersection_map.keys()
    directions = list(map(lambda d: unified_planning.model.Object(d, direction), direction_names))
    problem.add_objects(directions)

    for d, l in intersection_map.items():
        for i in range(len(l)-1):            
            problem.set_initial_value(connected(unified_planning.model.Object(l[i], loc), unified_planning.model.Object(l[i+1], loc), unified_planning.model.Object(d, direction)), True)

    # Agents
    at = Fluent('at', BoolType(), l1=loc)    
    arrived = Fluent('arrived', BoolType())
    not_arrived = Fluent('not-arrived', BoolType())
    start = Fluent('start', BoolType(), l=loc)        
    traveldirection = Fluent('traveldirection', BoolType(), d=direction)
    
    #  (:action arrive
        #     :agent    ?a - car 
        #     :parameters  (?l - loc)
        #     :precondition  (and  
        #     	(start ?a ?l)
        #     	(not (arrived ?a))
        #     	(free ?l)      
        #       )
        #     :effect    (and     	
        #     	(at ?a ?l)
        #     	(not (free ?l))
        #     	(arrived ?a)
        #       )
        #   )
    if durative:
        arrive = DurativeAction('arrive', l=loc)
        arrive.set_fixed_duration(1)        
        l = arrive.parameter('l')
        
        arrive.add_condition(StartTiming(),start(l))
        arrive.add_condition(StartTiming(),not_arrived())
        arrive.add_condition(OpenDurationInterval(StartTiming(), EndTiming()),free(l))
        arrive.add_effect(EndTiming(), at(l), True)
        arrive.add_effect(EndTiming(), free(l), False)
        arrive.add_effect(EndTiming(), arrived(), True)        
        arrive.add_effect(EndTiming(), not_arrived(), False)        
    else:
        arrive = InstantaneousAction('arrive', l=loc)    
        l = arrive.parameter('l')
        arrive.add_precondition(start(l))
        arrive.add_precondition(not_arrived())
        arrive.add_precondition(free(l))
        arrive.add_effect(at(l), True)
        arrive.add_effect(free(l), False)
        arrive.add_effect(arrived(), True)   
        arrive.add_effect(not_arrived(), False)   



    #   (:action drive
    #     :agent    ?a - car 
    #     :parameters  (?l1 - loc ?l2 - loc ?d - direction ?ly - loc)
    #     :precondition  (and      	
    #     	(at ?a ?l1)
    #     	(free ?l2)     
    #     	(travel-direction ?a ?d)
    #     	(connected ?l1 ?l2 ?d)
    #     	(yields-to ?l1 ?ly)
    #     	(free ?ly)
    #       )
    #     :effect    (and     	
    #     	(at ?a ?l2)
    #     	(not (free ?l2))
    #     	(not (at ?a ?l1))
    #     	(free ?l1)
    #       )
    #    )    
    # )
    if durative:
        if len(yields_list) > 0:
            drive = DurativeAction('drive', l1=loc, l2=loc, d=direction, ly=loc)
        else:
            drive = DurativeAction('drive', l1=loc, l2=loc, d=direction)
        drive.set_fixed_duration(1)        
        l1 = drive.parameter('l1')
        l2 = drive.parameter('l2')
        d = drive.parameter('d')        
        drive.add_condition(StartTiming(), at(l1))
        if wait_drive:        
            drive.add_condition(ClosedDurationInterval(StartTiming(), EndTiming()), free(l2))
        drive.add_condition(StartTiming(), traveldirection(d))
        drive.add_condition(EndTiming(), connected(l1,l2,d))        
        drive.add_effect(EndTiming(), at(l2),True)
        drive.add_effect(EndTiming(), free(l2), False)
        drive.add_effect(StartTiming(), at(l1), False)
        drive.add_effect(EndTiming(), free(l1), True)
        if len(yields_list) > 0:
            ly = drive.parameter('ly')
            drive.add_condition(StartTiming(), yieldsto(l1,ly))
            drive.add_condition(ClosedDurationInterval(StartTiming(), EndTiming()), free(ly))

    else:
        if len(yields_list) > 0:
            drive = InstantaneousAction('drive', l1=loc, l2=loc, d=direction, ly=loc)    
        else:
            drive = InstantaneousAction('drive', l1=loc, l2=loc, d=direction)
        l1 = drive.parameter('l1')
        l2 = drive.parameter('l2')
        d = drive.parameter('d')
        #ly = drive.parameter('ly')
        drive.add_precondition(at(l1))
        drive.add_precondition(free(l2))  # Remove for yield/wait
        drive.add_precondition(traveldirection(d))
        drive.add_precondition(connected(l1,l2,d))
        if len(yields_list) > 0:
            ly = drive.parameter('ly')
            drive.add_precondition(yieldsto(l1,ly))
            drive.add_precondition(free(ly))
        drive.add_effect(at(l2),True)
        drive.add_effect(free(l2), False)
        drive.add_effect(at(l1), False)
        drive.add_effect(free(l1), True)    



    plan = up.plans.SequentialPlan([])

    for d, l in intersection_map.items():
        carname = "car-" + d
        if carname in cars:
            car = Agent(carname, problem)
        
            problem.add_agent(car)
            car.add_fluent(at, default_initial_value=False)
            car.add_fluent(arrived, default_initial_value=False)
            car.add_fluent(not_arrived, default_initial_value=True)
            car.add_fluent(start, default_initial_value=False)
            car.add_fluent(traveldirection, default_initial_value=False)
            car.add_action(arrive)
            car.add_action(drive)

            slname = l[0]
            slobj = unified_planning.model.Object(slname, loc)

            glname = l[-1]
            globj = unified_planning.model.Object(glname, loc)
            
            dobj = unified_planning.model.Object(d, direction)

            problem.set_initial_value(Dot(car, car.fluent("start")(slobj)), True)
            problem.set_initial_value(Dot(car, car.fluent("traveldirection")(dobj)), True)        
            problem.add_goal(Dot(car, car.fluent("at")(globj)))

            if len(yields_list) > 0:
                yields = set()
                for l1_name, ly_name in yields_list:
                    problem.set_initial_value(yieldsto(problem.object(l1_name), problem.object(ly_name)), True)     
                    yields.add(problem.object(l1_name))
                for l1 in problem.objects(loc):
                    if l1 not in yields:
                        problem.set_initial_value(yieldsto(l1, dummy_loc), True)        

            # slobjexp1 = (ObjectExp(slobj)),        
            # plan.actions.append(up.plans.ActionInstance(arrive, slobjexp1, car))

            # for i in range(1,len(l)):
            #     flname = l[i-1]
            #     tlname = l[i]
            #     flobj = unified_planning.model.Object(flname, loc)
            #     tlobj = unified_planning.model.Object(tlname, loc)
            #     plan.actions.append(up.plans.ActionInstance(drive, (ObjectExp(flobj), ObjectExp(tlobj), ObjectExp(dobj) ), car))

    # Add waitfor annotations
    for agent in problem.agents:
        drive = agent.action("drive")
        l2 = drive.parameter("l2")        
        if wait_drive:
            problem.waitfor.annotate_as_waitfor(agent.name, drive.name, free(l2))
        if len(yields_list) > 0:
            ly = drive.parameter("ly")
            problem.waitfor.annotate_as_waitfor(agent.name, drive.name, free(ly)) 



    intersection = Example(problem=problem, plan=plan)
    return intersection