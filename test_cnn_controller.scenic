# os.system(f"scenic -S test_cnn_controller.scenic --count 1 --time 500 --2d --param controller_dir "/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/turn_resnet18_c/resnet18_coarse_best.pt" --param weather 'ClearNoon' --param results_path "test_controller_sim_results/0" --param dist_car 30 --param intersec 0")

import warnings
warnings.filterwarnings("ignore") 

import numpy as np 
import random
import os
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from model_utils import (
    build_midpoint_table,
    class_to_distance,
    get_test_transform,
    predict_single,
    load_model,
)

from scenic.domains.driving.controllers import (
    PIDLateralController,
    PIDLongitudinalController,
)


torch.set_default_device('cuda')

param timeout = 180
param map = localPath('carla_map/Town01.xodr')
param carla_map = 'Town01'
param timeBound = 300

param weather = globalParameters.weather

model scenic.simulators.carla.model

os.makedirs(globalParameters.results_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#CONSTANTS
EGO_MODEL = "vehicle.tesla.model3"
DIST_CAR1 = int(globalParameters.dist_car)
INTERSEC = int(globalParameters.intersec)
if 5 <= DIST_CAR1 and DIST_CAR1 < 10:
    OUT_REACH_DIST = 10 
elif 10 <= DIST_CAR1 and DIST_CAR1 < 30:
    OUT_REACH_DIST = DIST_CAR1 + 15
else:
    OUT_REACH_DIST = DIST_CAR1 + 25


EGO_SPEED = 5
THROTTLE_ACTION = 0.5
BRAKE_ACTION = 1.0
EGO_TO_LEADER = Range(-15, -10)
EGO_BRAKING_THRESHOLD = 6
EGO_ACCELERATION_THRESHOLD = 10

CONTROLLER_PATH = globalParameters.controller_dir

RESULTS_PATH = globalParameters.results_path


behavior FollowLaneBehaviorModified(target_speed = 10, laneToFollow=None, is_oppositeTraffic=False, leaderCar=None):
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
        self.select_maneuver = Uniform(*maneuvers)
        nearby_intersection = current_lane.maneuvers[0].intersection
        if nearby_intersection == None:
            nearby_intersection = current_lane.centerline[-1]
    else:
        nearby_intersection = current_lane.centerline[-1]
        self.select_maneuver = None
    
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
                    select_maneuver = Uniform(*maneuvers)
                    self.select_maneuver = select_maneuver
                    #self.selected_maneuver = select_maneuver.type.value
                    #print(self.select_maneuver)
                else:
                    select_maneuver = leaderCar.select_maneuver
                    #self.selected_maneuver = select_maneuver.type.value

            elif len(current_lane.maneuvers) > 0:
                select_maneuver = Uniform(*current_lane.maneuvers)
                #self.select_maneuver = select_maneuver
                #self.selected_maneuver = select_maneuver.type.value
                #print(select_maneuver)
            else:
                take SetBrakeAction(1.0)
                break
            self.select_maneuver = select_maneuver
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
                #self.selected_maneuver = select_maneuver.type.value
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
        

        take RegulatedControlAction(throttle, current_steer_angle, past_steer_angle)
        past_steer_angle = current_steer_angle
        past_speed = current_speed





behavior ControllerBehavior(target_speed = 10, controller_path = CONTROLLER_PATH, leaderCar = None):
    past_steer_angle = 0
    past_speed = 0 # making an assumption here that the agent starts from zero speed

    original_target_speed = target_speed

    current_lane = self.lane

    TRIGGER_DISTANCE_TO_SLOWDOWN = 10 
    if current_lane.maneuvers != ():
        nearby_intersection = current_lane.maneuvers[0].intersection
        if nearby_intersection == None:
            nearby_intersection = current_lane.centerline[-1]
    else:
        nearby_intersection = current_lane.centerline[-1]
    
    # instantiate longitudinal and lateral controllers
    # _lon_controller, _lat_controller = simulation().getLaneFollowingControllers(self)
    dt = simulation().timestep
    _lon_controller = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
    _lat_controller_turn = PIDLateralController(K_P=0.8, K_D=0.2, K_I=0.0, dt=dt)
    _lat_controller_straight = PIDLateralController(K_P=0.2, K_D=0.1, K_I=0.0, dt=dt)


    controller, _ = load_model(controller_path, device)
    transform = get_test_transform(112, 224)

    self.selected_maneuver = 1

    while True:

        """
        if distance from self to intersection < TRIGGER_DISTANCE_TO_SLOWDOWN:
            if leaderCar is None:
                maneuvers = current_lane.maneuvers
                select_maneuver = Uniform(*maneuvers)
            else:
                select_maneuver = leaderCar.select_maneuver
            self.selected_maneuver = select_maneuver.type.value
        else:
            self.selected_maneuver = 1
        """
        #if distance from self to intersection < TRIGGER_DISTANCE_TO_SLOWDOWN:
        if leaderCar is not None:
            select_maneuver = leaderCar.select_maneuver
            if select_maneuver is not None:
                self.selected_maneuver = select_maneuver.type.value
            else:
                self.selected_maneuver = 1

        front_img = self.sensors["front_rgb"]._lastObservation
        if isinstance(front_img, np.ndarray):

            input_img = Image.fromarray(front_img)
            result = predict_single(controller, input_img, self.selected_maneuver, device, transform, 100.0)
            cte_pred = result['cte']
            dist_pred = result['distance_m']

        else:
            cte_pred, dist_pred = 0, 0

        if self.speed is not None:
            current_speed = self.speed
        else:
            current_speed = past_speed

        speed_error = target_speed - current_speed


        # compute throttle : Longitudinal Control
        throttle = _lon_controller.run_step(speed_error)
        self.acc = throttle

        # compute steering : Lateral Control
        if abs(cte_pred) > 0.5: 
            _lat_controller = _lat_controller_turn
        else:
            _lat_controller = _lat_controller_straight
        current_steer_angle = _lat_controller.run_step(cte_pred) 

        if dist_pred > EGO_ACCELERATION_THRESHOLD:
            throttle = THROTTLE_ACTION
            self.acc = throttle
        if dist_pred < EGO_BRAKING_THRESHOLD:
            self.acc = -1
            take SetBrakeAction(1.0)
        else:
            take RegulatedControlAction(throttle, current_steer_angle, past_steer_angle)
        past_steer_angle = current_steer_angle
        past_speed = current_speed


# so the context does not change
behavior KeepDistance(speed=10):
    try: 
        do FollowLaneBehavior(speed)
    interrupt when self.distanceToClosest(Object) > OUT_REACH_DIST:
        take SetBrakeAction(0.8), SetThrottleAction(0)


## DEFINING SPATIAL RELATIONS

if INTERSEC:
    intersec = Uniform(*network.intersections)
    lane = Uniform(*intersec.incomingLanes)
    end = lane.centerline[-1]
    start = new OrientedPoint following roadDirection from end for -Range(5,10)
else:
    lane = Uniform(*network.lanes)
    start = new OrientedPoint on lane.centerline

if DIST_CAR1 <= 50:
    
    car1 = new Car at start,
            with behavior FollowLaneBehaviorModified(8),
            with select_maneuver 1

    ego = new Car following roadDirection from car1.position for -1*DIST_CAR1,
        with blueprint EGO_MODEL,
        with behavior ControllerBehavior(target_speed=EGO_SPEED, leaderCar=car1),
        with cte 0,
        with acc 0,
        with selected_maneuver 1,
        with sensors {"front_rgb": RGBSensor(offset=(0, 2, 1), width=640, height=320),
                    "aerial_rgb": RGBSensor(offset=(0, -10, 4), width=1280, height=640)
                    },    
else:
    ego = new Car at start,
        with blueprint EGO_MODEL,
        with behavior ControllerBehavior(EGO_SPEED),
        with cte 0,
        with acc 0,
        with selected_maneuver 1,
        with sensors {"front_rgb": RGBSensor(offset=(0, 2, 1), width=640, height=320),
                    "aerial_rgb": RGBSensor(offset=(0, -10, 4), width=1280, height=640)
                    },  




"""
if DIST_CAR1 == 5:
    require distance to car1 > DIST_CAR1-1 and distance to car1 < DIST_CAR1+3
elif DIST_CAR1 == 10 or DIST_CAR1 == 20:
    require distance to car1 > DIST_CAR1 and distance to car1 < DIST_CAR1+10 
elif DIST_CAR1 == 30:
    require distance to car1 > DIST_CAR1 and distance to car1 < DIST_CAR1+20 
"""
if DIST_CAR1 <= 50:
    require ego can see car1

if INTERSEC:
    require (distance from start to intersection) < 10 and (distance from start to intersection) > 3 
else:
    require (distance from start to intersection) > 10  

"""
if DIST_CAR1 == 5:
    terminate when ego.collision > 0 and (ego.speed < 0.1 and (distance to start) > 1) # or (distance to car1 > (DIST_CAR1+10))
elif DIST_CAR1 == 10 or DIST_CAR1 == 20: 
    terminate when ego.collision > 0 and (ego.speed < 0.1 and (distance to start) > 1) # or (distance to car1 > (DIST_CAR1+15))
elif DIST_CAR1 == 30: 
    terminate when ego.collision > 0 and (ego.speed < 0.1 and (distance to start) > 1) # or (distance to car1 > (DIST_CAR1+25))
else:
    terminate when ego.collision > 0 and (ego.speed < 0.1 and (distance to start) > 1)
"""

record ego.speed every 0.1 seconds after 3 seconds to RESULTS_PATH+"/speed.npz"
record ego.acc every 0.1 seconds after 3 seconds to RESULTS_PATH+"/acc.npz"
record ego.selected_maneuver every 0.1 seconds after 3 seconds to RESULTS_PATH+"/maneuver.npz"
#record ego.distanceToClosest(Car) every time_step seconds after 3 seconds to RESULTS_PATH+"/dist.npz"

"""
if INTERSEC:
    terminate when distance from ego to intersection > 10 
else:
    terminate when distance from ego to intersection < 3
"""
