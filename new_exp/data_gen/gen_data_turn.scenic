""" Scenario Description
This is a lane following simulaion only used for collecting data. 
Ego car is controlled by an IDM model to calculate the acc.
The ego car follows the leader car maintaining a certain time headway while lane keeping.
The leader car usually starts before a turining point and takes a turn at the intersection.
cmd:
scenic gen_data_turn.scenic --2d -S --count 1 --time 800  --param result_path "test_data" --param car_dist -30 --param leader_speed 8 --param weather "ClearNoon" --param ego_idm "smooth"
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
param render = 1
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
EGO_TO_LEADER = Range(CAR_DISTANCE, CAR_DISTANCE+10) # -30, -20

IDM_PROFILES = {
    "aggressive": {
        "IDM_T":     0.5,   # short headway — tailgating
        "IDM_S0":    0.1,   # tiny standstill gap
        "IDM_A":     4.5,   # punchy acceleration
        "IDM_B":     5.0,   # hard braking
        "IDM_DELTA": 10,
    },
    "moderate": {
        "IDM_T":     1.5,   # long headway — anticipatory
        "IDM_S0":    2.5,   # generous standstill gap
        "IDM_A":     2.0,   # gentle acceleration
        "IDM_B":     2.5,   # soft braking, decelerates early
        "IDM_DELTA": 4.0,
    },
    "smooth": {
        "IDM_T":     5.0,   # long headway — anticipatory
        "IDM_S0":    6.0,   # generous standstill gap
        "IDM_A":     0.8,   # gentle acceleration
        "IDM_B":     1.0,   # soft braking, decelerates early
        "IDM_DELTA": 4.0,
    },
}



behavior FollowLaneBehaviorModified(target_speed = 12, laneToFollow=None, is_oppositeTraffic=False, leaderCar=None, idm_profile="smooth"):
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

        if leaderCar:
            _idm = IDM_PROFILES[idm_profile]
            # --- IDM Parameters ---
            IDM_V0      = target_speed   # desired speed (m/s)
            IDM_T     = _idm["IDM_T"]
            IDM_S0    = _idm["IDM_S0"]
            IDM_A     = _idm["IDM_A"]
            IDM_B     = _idm["IDM_B"]
            IDM_DELTA = _idm["IDM_DELTA"]

            gap         = (distance from self to leaderCar) - 4.5  # 4.5m = approx car length
            gap         = max(gap, 0.1)                             # avoid division by zero
            v           = current_speed
            delta_v     = v - (leaderCar.speed if leaderCar.speed else 0)

            s_star      = IDM_S0 + max(0, v * IDM_T + (v * delta_v) / (2 * (IDM_A * IDM_B) ** 0.5))
            idm_accel   = IDM_A * (1 - (v / IDM_V0) ** IDM_DELTA - (s_star / gap) ** 2)

            idm_accel   = max(min(idm_accel, IDM_A), -IDM_B * 1.5)  # clamp

            if idm_accel >= 0:
                throttle = min(idm_accel / IDM_A, 1.0)
                brake_cmd = 0.0
                self.record_acc = throttle
            else:
                throttle = 0.0
                brake_cmd = min(abs(idm_accel) / (IDM_B * 1.5), 1.0)
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


leader = new Car following roadDirection from start for -5,
            with blueprint EGO_MODEL,
            with behavior FollowLaneBehaviorModified(target_speed=LEADER_SPEED),
            with color Color(0,0,0) 

ego = new Car following roadDirection from leader.position for EGO_TO_LEADER,
            with blueprint EGO_MODEL,
            with behavior FollowLaneBehaviorModified(target_speed=EGO_SPEED, leaderCar=leader, idm_profile=EGO_IDM),
            with cte 0,
            with selected_maneuver 1,
            with record_acc 0.0,
            with sensors {"front_rgb": RGBSensor(offset=(0, 2, 1), attributes=attrs)
                        } 


time_step = 0.1

require ego can see leader


#record ego.position.x every time_step seconds after 3 seconds to RESULT_PATH+"/ego_pos.npz"
record ego.distanceToClosest(Car) every time_step seconds after 3 seconds to RESULT_PATH+"/dist.npz"
record ego.cte every time_step seconds after 3 seconds to RESULT_PATH+"/cte.npz"
record ego.record_acc every time_step seconds after 3 seconds to RESULT_PATH+"/acc.npz"
record ego.selected_maneuver every time_step seconds after 3 seconds to RESULT_PATH+"/maneuver.npz"
record ego.speed every time_step seconds after 3 seconds to RESULT_PATH+"/ego.npz"
record leader.speed every time_step seconds after 3 seconds to RESULT_PATH+"/leader_speed.npz"
record ego.observations["front_rgb"] every time_step seconds after 3 seconds to RESULT_PATH+"/img/front_rgb_{time:.1f}.jpg"


#terminate when ((distance from ego to leader > ((-1*CAR_DISTANCE)+10)) or ((distance from ego to leader < 4.7) ))
terminate when (distance from ego to leader < 4.7)
