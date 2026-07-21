""" Scenario Description
This is a lane following simulaion only used for collecting data. 
Ego car is controlled by an IDM model to calculate the acc.
The ego car follows the leader car maintaining a certain time headway while lane keeping.
The leader car usually starts before a turining point and takes a turn at the intersection.

IDM driving profiles (param ego_idm), ordered from most to least aggressive:
    "sport"        - shortest headway/gap, strongest accel & braking, fastest actuator
                     response (lowest ACTUATOR_TAU) -> snappy, tightly-following.
    "aggressive"   - closes the gap to the leader as much as possible; slightly less
                     extreme than "sport".
    "dynamic"      - brisk but not extreme; noticeably quicker than "balanced".
    "balanced"     - middle of the spectrum on every axis.
    "comfort"      - gentle accel/braking, slower actuator response for a soft ride.
    "conservative" - long headway and large gap from the leader to minimize risk,
                     still holds real braking authority (MAX_BRAKE) in reserve.
    "defensive"    - longest headway/gap, gentlest accel/braking, slowest actuator
                     response -> maximally cautious.

cmd:
scenic gen_sim.scenic --2d -S --count 1 --time 800  --param result_path "test_data" --param car_dist -30 --param leader_speed 8 --param weather "ClearNoon" --param ego_idm "smooth"
"""

import random
import carla

from scenic.domains.driving.controllers import (
    PIDLateralController,
    PIDLongitudinalController,
)

param timeout = 30
param map = localPath('carla_map/Town01.xodr')
param carla_map = 'Town01'
param render = 0
param timeBound = 300
param weather = globalParameters.weather

model scenic.simulators.carla.model

#Passing parameters
RESULT_PATH = globalParameters.result_path
CAR_DISTANCE = globalParameters.car_dist
LEADER_SPEED = globalParameters.leader_speed
EGO_IDM = globalParameters.ego_idm


#CONSTANTS
EGO_MODEL = "vehicle.tesla.model3"
EGO_SPEED = 18
EGO_TO_LEADER = CAR_DISTANCE



IDM_PROFILES = {

    "sport": {
        "IDM_T": 0.7,
        "IDM_S0": 1.0,
        "IDM_A": 3.5,
        "IDM_B": 4.0,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.10
    },

    "aggressive": {
        "IDM_T": 0.9,
        "IDM_S0": 1.5,
        "IDM_A": 3.0,
        "IDM_B": 3.5,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.15
    },

    "dynamic": {
        "IDM_T": 1.1,
        "IDM_S0": 2.0,
        "IDM_A": 2.5,
        "IDM_B": 3.0,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.25
    },

    "balanced": {
        "IDM_T": 1.4,
        "IDM_S0": 2.5,
        "IDM_A": 2.0,
        "IDM_B": 2.5,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.35
    },

    "comfort": {
        "IDM_T": 1.7,
        "IDM_S0": 3.0,
        "IDM_A": 1.5,
        "IDM_B": 2.0,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.50
    },

    "conservative": {
        "IDM_T": 2.2,
        "IDM_S0": 4.0,
        "IDM_A": 1.2,
        "IDM_B": 1.8,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 0.70
    },

    "defensive": {
        "IDM_T": 2.8,
        "IDM_S0": 5.0,
        "IDM_A": 0.8,
        "IDM_B": 1.5,
        "IDM_DELTA": 4.0,
        "MAX_BRAKE": 6.0,
        "ACTUATOR_TAU": 1.0
    },
}


behavior FollowLaneBehaviorModified(target_speed = 12, laneToFollow=None, is_oppositeTraffic=False, leaderCar=None, no_leader=False, idm_profile="smooth"):
    """ 
    Follow's the lane on which the vehicle is at, unless the laneToFollow is specified.
    Once the vehicle reaches an intersection, by default, the vehicle will take the straight route.
    If straight route is not available, then any availble turn route will be taken, uniformly randomly. 
    If turning at the intersection, the vehicle will slow down to make the turn, safely. 

    This behavior does not terminate. A recommended use of the behavior is to accompany it with condition,
    e.g. do FollowLaneBehavior() until ...

    :param target_speed: Its unit is in m/s. By default, it is set to 10 m/s
    :param laneToFollow: If the lane to follow is different from the lane that the vehicle is on, this parameter can be used to specify that lane. By default, this variable will be set to None, which means that the vehicle will follow the lane that it is currently on.
    """

    past_steer_angle = 0
    past_speed = 0 # making an assumption here that the agent starts from zero speed
    filtered_accel = 0.0 # actuator-lag state for the IDM low-pass filter; must persist across steps
    if laneToFollow is None:
        current_lane = self.lane
    else:
        current_lane = laneToFollow

    current_centerline = current_lane.centerline
    in_turning_lane = False # assumption that the agent is not instantiated within a connecting lane
    intersection_passed = False
    entering_intersection = False # assumption that the agent is not instantiated within an intersection
    end_lane = None
    original_target_speed = target_speed
    TARGET_SPEED_FOR_TURNING = 3 # KM/H
    TRIGGER_DISTANCE_TO_SLOWDOWN = 10 # FOR TURNING AT INTERSECTIONS

    if current_lane.maneuvers != ():
        nearby_intersection = current_lane.maneuvers[0].intersection
        if nearby_intersection == None:
            nearby_intersection = current_lane.centerline[-1]
    else:
        nearby_intersection = current_lane.centerline[-1]
    
    # instantiate longitudinal and lateral controllers
    _lon_controller, _lat_controller = simulation().getLaneFollowingControllers(self)

    while True:

        if self.speed is not None:
            current_speed = self.speed
        else:
            current_speed = past_speed

        if not entering_intersection and (distance from self.position to nearby_intersection) < TRIGGER_DISTANCE_TO_SLOWDOWN:
            entering_intersection = True
            intersection_passed = False

            if distance from self to intersection < TRIGGER_DISTANCE_TO_SLOWDOWN:
                if leaderCar is None:
                    maneuvers = current_lane.maneuvers
                    turn_maneuvers = filter(lambda i : i.type != ManeuverType.STRAIGHT, maneuvers)
                    if len(turn_maneuvers) > 0:
                        select_maneuver = Uniform(*turn_maneuvers)
                        self.turn_maneuver = select_maneuver
                    else:
                        select_maneuver = Uniform(*maneuvers) 
                    self.select_maneuver = select_maneuver
                else:
                    select_maneuver = leaderCar.turn_maneuver
                    #print(select_maneuver.type)
            
            

            elif len(current_lane.maneuvers) > 0:
                select_maneuver = Uniform(*current_lane.maneuvers)
                #self.selected_maneuver = select_maneuver.type.value
                #print(select_maneuver)
            else:
                take SetBrakeAction(1.0)
                break

            # print(select_maneuver.type)

            # assumption: there always will be a maneuver
            if select_maneuver.connectingLane != None:
                current_centerline = concatenateCenterlines([current_centerline, select_maneuver.connectingLane.centerline, \
                    select_maneuver.endLane.centerline])
            else:
                current_centerline = concatenateCenterlines([current_centerline, select_maneuver.endLane.centerline])

            current_lane = select_maneuver.endLane
            end_lane = current_lane

            if current_lane.maneuvers != ():
                nearby_intersection = current_lane.maneuvers[0].intersection
                if nearby_intersection == None:
                    nearby_intersection = current_lane.centerline[-1]
            else:
                nearby_intersection = current_lane.centerline[-1]

            if select_maneuver.type != ManeuverType.STRAIGHT:
                self.selected_maneuver = select_maneuver.type.value
                in_turning_lane = True
                target_speed = TARGET_SPEED_FOR_TURNING

                # do TurnBehavior(trajectory = current_centerline, target_speed=target_speed)
                trajectory = current_centerline
                target_speed = target_speed
                if isinstance(trajectory, PolylineRegion):
                    trajectory_centerline = trajectory
                else:
                    trajectory_centerline = concatenateCenterlines([traj.centerline for traj in trajectory])

                # instantiate longitudinal and lateral controllers
                # _lon_controller, _lat_controller = simulation().getTurningControllers(self)
                dt = simulation().timestep
                _lon_controller = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
                _lat_controller = PIDLateralController(K_P=0.8, K_D=0.6, K_I=0.0, dt=dt)

                past_steer_angle = 0

                while self in network.intersectionRegion:
                    if self.speed is not None:
                        current_speed = self.speed
                    else:
                        current_speed = 0

                    self.cte = trajectory_centerline.signedDistanceTo(self.position)
                    speed_error = target_speed - current_speed

                    # compute throttle : Longitudinal Control
                    throttle = _lon_controller.run_step(speed_error)

                    # compute steering : Latitudinal Control
                    current_steer_angle = _lat_controller.run_step(self.cte)

                    # Keep record_acc live during turns instead of freezing at its
                    # pre-turn value: the turning controller only outputs a single
                    # throttle signal (no separate brake command), so clamp it to
                    # [-1, 1] to match the sign convention used in the IDM branch
                    # (positive = throttle fraction, negative = braking fraction).
                    self.record_acc = max(min(throttle, 1.0), -1.0)

                    take RegulatedControlAction(throttle, current_steer_angle, past_steer_angle)
                    past_steer_angle = current_steer_angle


        if (end_lane is not None) and (self.position in end_lane) and not intersection_passed:
            self.selected_maneuver = 1 # out of intersection and straight road again
            intersection_passed = True
            in_turning_lane = False
            entering_intersection = False 
            target_speed = original_target_speed
            # _lon_controller, _lat_controller = simulation().getLaneFollowingControllers(self)
            dt = simulation().timestep
            _lon_controller = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
            _lat_controller = PIDLateralController(K_P=0.2, K_D=0.5, K_I=0.0, dt=dt)

        nearest_line_points = current_centerline.nearestSegmentTo(self.position)
        nearest_line_segment = PolylineRegion(nearest_line_points)
        self.cte = nearest_line_segment.signedDistanceTo(self.position)
        if is_oppositeTraffic:
            self.cte = -self.cte

        speed_error = target_speed - current_speed

        throttle = _lon_controller.run_step(speed_error)


        current_steer_angle = _lat_controller.run_step(self.cte) 

        if leaderCar or no_leader:
            _idm = IDM_PROFILES[idm_profile]
            # --- IDM Parameters ---
            IDM_V0      = target_speed   # desired speed (m/s)
            IDM_T     = _idm["IDM_T"]
            IDM_S0    = _idm["IDM_S0"]
            IDM_A     = _idm["IDM_A"]
            IDM_B     = _idm["IDM_B"]
            IDM_DELTA = _idm["IDM_DELTA"]
            MAX_BRAKE     = _idm["MAX_BRAKE"]
            ACTUATOR_TAU  = _idm["ACTUATOR_TAU"]
            v = current_speed

            if no_leader:
                # Free-road driving: no vehicle ahead, so drop the gap/interaction
                # term entirely and just accelerate smoothly toward target_speed.
                # (Using a fake far-away "virtual leader" here instead can cause
                # premature braking at high speed / large IDM_T, since the desired
                # gap s_star can exceed the fake gap and trigger unwanted braking.)
                idm_accel = IDM_A * (1 - (v / IDM_V0) ** IDM_DELTA)
            else:
                gap         = (distance from self to leaderCar) - 4.5  # 4.5m = approx car length
                gap         = max(gap, 0.1)                             # avoid division by zero
                delta_v     = v - (leaderCar.speed if leaderCar.speed else 0)
                s_star      = IDM_S0 + max(0, v * IDM_T + (v * delta_v) / (2 * (IDM_A * IDM_B) ** 0.5))
                idm_accel   = IDM_A * (1 - (v / IDM_V0) ** IDM_DELTA - (s_star / gap) ** 2)

            idm_accel = max(min(idm_accel, IDM_A), -MAX_BRAKE)  # clamp to physical accel/brake limits

            dt = simulation().timestep

            # First-order actuator lag: smooths the raw IDM command toward a more
            # realistic, gradually-responding acceleration instead of an instant jump.
            filtered_accel += (dt / ACTUATOR_TAU) * (idm_accel - filtered_accel)

            idm_accel = filtered_accel

            if idm_accel >= 0:
                throttle = min(idm_accel / IDM_A, 1.0)
                brake_cmd = 0.0
                self.record_acc = throttle
            else:
                throttle = 0.0
                brake_cmd = min(abs(idm_accel) / MAX_BRAKE, 1.0)
                take SetBrakeAction(brake_cmd)
                self.record_acc = -brake_cmd
            
            #print(throttle, brake_cmd, self.record_acc)
        take RegulatedControlAction(throttle, current_steer_angle, past_steer_angle)
        past_steer_angle = current_steer_angle
        past_speed = current_speed

intersec = Uniform(*network.intersections)
turn_maneuvers = filter(lambda i: i.type == ManeuverType.RIGHT_TURN, intersec.maneuvers)
turn_maneuver = Uniform(*turn_maneuvers)
startLane = turn_maneuver.startLane 


start = startLane.centerline[-1]

attrs = {"image_size_x": 640,
         "image_size_y": 320}

if EGO_TO_LEADER < 50:

    leader = new Car following roadDirection from start for -5,
                with blueprint EGO_MODEL,
                with behavior FollowLaneBehaviorModified(target_speed=LEADER_SPEED),
                with color Color(0,0,0) 

    ego = new Car following roadDirection from leader.position for -1*EGO_TO_LEADER,
                with blueprint EGO_MODEL,
                with behavior FollowLaneBehaviorModified(target_speed=EGO_SPEED, leaderCar=leader, idm_profile=EGO_IDM),
                with cte 0,
                with selected_maneuver 1,
                with record_acc 0.0,
                with sensors {"front_rgb": RGBSensor(offset=(0, 2, 1), attributes=attrs)
                            } 
else:
    ego = new Car following roadDirection from start for -5,
                with blueprint EGO_MODEL,
                with behavior FollowLaneBehaviorModified(target_speed=EGO_SPEED, no_leader=True, idm_profile=EGO_IDM),
                with cte 0,
                with selected_maneuver 1,
                with record_acc 0.0,
                with leader_speed EGO_SPEED,
                with sensors {"front_rgb": RGBSensor(offset=(0, 2, 1), attributes=attrs)
                            } 
    


time_step = 0.1

if EGO_TO_LEADER < 50:
    require ego can see leader
    record leader.speed every time_step seconds after 3 seconds to RESULT_PATH+"/leader_speed.npz"
else:
    record ego.leader_speed every time_step seconds after 3 seconds to RESULT_PATH+"/leader_speed.npz"


#record ego.position.x every time_step seconds after 3 seconds to RESULT_PATH+"/ego_pos.npz"
record ego.distanceToClosest(Car) every time_step seconds after 3 seconds to RESULT_PATH+"/dist.npz"
record ego.cte every time_step seconds after 3 seconds to RESULT_PATH+"/cte.npz"
record ego.record_acc every time_step seconds after 3 seconds to RESULT_PATH+"/acc.npz"
record ego.selected_maneuver every time_step seconds after 3 seconds to RESULT_PATH+"/maneuver.npz"
record ego.speed every time_step seconds after 3 seconds to RESULT_PATH+"/speed.npz"
record ego.observations["front_rgb"] every time_step seconds after 3 seconds to RESULT_PATH+"/img/front_rgb_{time:.1f}.jpg"