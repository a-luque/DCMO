#scenic -S gen_sim.scenic --count 1 --time 200 --2d --param results_path "sim_results/0" --param controller_dir "../controllers/resnet18_coarse_best.pt" --param weather "ClearNoon" --param intersect 0 --param dist_car 30 --param speed_car 8


import warnings
warnings.filterwarnings("ignore") 

import sys
sys.path.append("..")

import numpy as np 
import random
import os
from PIL import Image

import torch
import carla
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
param map = localPath('../carla_map/Town01.xodr')
param carla_map = 'Town01'
param timeBound = 300
param render = 0

param weather = globalParameters.weather

model scenic.simulators.carla.model

os.makedirs(globalParameters.results_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
EGO_MODEL = "vehicle.tesla.model3"
dist_to_leader = int(globalParameters.dist_car)
if dist_to_leader == 5:
    DIST_CAR1 = Range(4.6, 9.6)
    OUT_REACH_DIST = 10
elif dist_to_leader == 10:
    DIST_CAR1 = Range(9.6, 14.6)
    OUT_REACH_DIST = 15
elif dist_to_leader == 15:
    DIST_CAR1 = Range(14.6, 24.6)
    OUT_REACH_DIST = 25
else:
    DIST_CAR1 = 100
    OUT_REACH_DIST = 100


input_leader_speed = int(globalParameters.speed_car)
if input_leader_speed == 4:
    LEADER_SPEED = Range(3,5)
if input_leader_speed == 6:
    LEADER_SPEED = Range(5,7)
if input_leader_speed == 8:
    LEADER_SPEED = Range(7,9)


EGO_SPEED = 5
THROTTLE_ACTION = 0.5
MAX_SPEED = 8
BRAKE_ACTION = 1.0
EGO_BRAKING_THRESHOLD = 10
EGO_ACCELERATION_THRESHOLD = OUT_REACH_DIST
TARGET_SPEED_FOR_TURNING = 3 # KM/H
TRIGGER_DISTANCE_TO_SLOWDOWN = 6 # FOR TURNING AT INTERSECTIONS
CONTROLLER_PATH = globalParameters.controller_path

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
    current_lane = None

    if laneToFollow is None:
        current_lane = self._lane
    else:
        current_lane = laneToFollow

    current_centerline = current_lane.centerline
    in_turning_lane = False # assumption that the agent is not instantiated within a connecting lane
    intersection_passed = False
    entering_intersection = False # assumption that the agent is not instantiated within an intersection
    end_lane = None
    original_target_speed = target_speed

    if current_lane.maneuvers != ():
        nearby_intersection = current_lane.maneuvers[0].intersection
        if nearby_intersection == None:
            nearby_intersection = current_lane.centerline[-1]
    else:
        nearby_intersection = current_lane.centerline[-1]
    
    # instantiate longitudinal and lateral controllers
    _lon_controller, _lat_controller = simulation().getLaneFollowingControllers(self)    


    if leaderCar is None:
        # only leading car registers its initial position
        self.initialPos = self.position

    egoFollow = False


    while True:

        if self.speed is not None:
            current_speed = self.speed
        else:
            current_speed = past_speed

        if not entering_intersection and (distance from self.position to nearby_intersection) < TRIGGER_DISTANCE_TO_SLOWDOWN:
            entering_intersection = True
            intersection_passed = False
            
            if distance from self to intersection < TRIGGER_DISTANCE_TO_SLOWDOWN:

                maneuvers = current_lane.maneuvers
                straight_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, maneuvers)
                if len(straight_maneuvers) > 0:
                    select_maneuver = Uniform(*straight_maneuvers)
                else:
                    right_turn_maneuvers = filter(lambda i: i.type == ManeuverType.RIGHT_TURN, maneuvers)
                    select_maneuver = Uniform(*right_turn_maneuvers)


            elif len(current_lane.maneuvers) > 0:
                select_maneuver = Uniform(*current_lane.maneuvers)
            if not select_maneuver.connectingLane is None:
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
                in_turning_lane = True
                target_speed = TARGET_SPEED_FOR_TURNING

                trajectory = current_centerline
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

                while distance from self to intersection < TRIGGER_DISTANCE_TO_SLOWDOWN:
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
            # current_steer_angle += 1/3 * random.randrange(-2,2) * random.random()
            if distance from self to leaderCar > OUT_REACH_DIST:
                throttle = 0.6
            if distance from self to leaderCar < EGO_BRAKING_THRESHOLD+4:
                throttle = 0
                take SetBrakeAction(1.0)
        

        take RegulatedControlAction(throttle, current_steer_angle, past_steer_angle)
        past_steer_angle = current_steer_angle
        past_speed = current_speed

        if leaderCar and (distance from self to intersection > TRIGGER_DISTANCE_TO_SLOWDOWN+3):
            self.closeIntersection = 0






behavior ControllerBehavior(target_speed = 10, controller_path = CONTROLLER_PATH, leaderCar = None):

    collision_bp = self.carlaActor.get_world().get_blueprint_library().find('sensor.other.collision')
    collision_sensor = self.carlaActor.get_world().spawn_actor(collision_bp, carla.Transform(),
        attach_to=self.carlaActor,
        attachment_type=carla.AttachmentType.Rigid)
    collision_sensor.listen(lambda col: _collision_callback(self,col))

    self.collision_sensor = collision_sensor


    past_steer_angle = 0
    past_speed = 0 # making an assumption here that the agent starts from zero speed

    original_target_speed = target_speed
    TARGET_SPEED_FOR_TURNING = 3 # KM/H


    TRIGGER_DISTANCE_TO_SLOWDOWN = 10 
    current_lane = self._lane

    if len(current_lane.maneuvers) > 0:
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

    if self.controller is None:
        self.controller, _ = load_model(controller_path, device)
    transform = get_test_transform(112, 224)


    while True:
        current_lane = self._lane
        if current_lane:
            current_centerline = current_lane.centerline


        self.select_maneuver = 1
        front_img = self.sensors["front_rgb"]._lastObservation
        if isinstance(front_img, np.ndarray):

            input_img = Image.fromarray(front_img)
            result = predict_single(self.controller, input_img, self.select_maneuver, device, transform, 100.0)
            cte_pred = result['cte']
            if distance from self to intersection < TRIGGER_DISTANCE_TO_SLOWDOWN: 
                self.closeIntersection = 1
            dist_pred = result['distance_m']

        else:
            cte_pred, dist_pred = 0, 0

        if self.speed is not None:
            current_speed = self.speed
        else:
            current_speed = past_speed

        
        if current_lane:
            self.cte = current_centerline.signedDistanceTo(self.position)

        # compute steering : Lateral Control
        if abs(cte_pred) > 0.5: 
            if current_speed > TARGET_SPEED_FOR_TURNING:
                take SetBrakeAction(0.4)
            target_speed = TARGET_SPEED_FOR_TURNING
            _lat_controller = _lat_controller_turn
            # compute throttle : Longitudinal Control
            speed_error = target_speed - current_speed
            throttle = _lon_controller.run_step(speed_error)
            self.acc = throttle
        else:
            target_speed = original_target_speed
            _lat_controller = _lat_controller_straight
            if dist_pred > EGO_ACCELERATION_THRESHOLD and current_speed < MAX_SPEED:
                throttle = THROTTLE_ACTION
                self.acc = throttle
            else:
                # compute throttle : Longitudinal Control
                speed_error = target_speed - current_speed
                throttle = _lon_controller.run_step(speed_error)
                self.acc = throttle
        current_steer_angle = _lat_controller.run_step(cte_pred) 




        
        if dist_pred < EGO_BRAKING_THRESHOLD:
            take SetBrakeAction(1.0)
            self.acc = (self.speed - past_speed) / 0.1
        else:
            take RegulatedControlAction(throttle, current_steer_angle, past_steer_angle)

        past_steer_angle = current_steer_angle
        past_speed = self.speed



behavior EgoBehavior(target_speed = 10, leaderCar=None):
    
    if self.sensor_created_flag:
        collision_bp = self.carlaActor.get_world().get_blueprint_library().find('sensor.other.collision')
        collision_sensor = self.carlaActor.get_world().spawn_actor(collision_bp, carla.Transform(),
            attach_to=self.carlaActor,
            attachment_type=carla.AttachmentType.Rigid)
        collision_sensor.listen(lambda col: _collision_callback(self,col))

        self.collision_sensor = collision_sensor
        self.sensor_created_flag = False

    try: 
        do ControllerBehavior(target_speed=target_speed, leaderCar=leaderCar)
    interrupt when self.closeIntersection:
        do EgoBehavior2(target_speed = target_speed, leaderCar=leaderCar)

behavior EgoBehavior2(target_speed = 10, leaderCar=None):
    try:
        do FollowLaneBehaviorModified(target_speed=target_speed, leaderCar=leaderCar)
    interrupt when not self.closeIntersection:
        do EgoBehavior(target_speed=target_speed, leaderCar=leaderCar)


## DEFINING SPATIAL RELATIONS

lane = Uniform(*network.lanes)
start = new OrientedPoint on lane.centerline



if dist_to_leader <= 15:
    
    car1 = new Car at start,
            with select_maneuver 1,
            with behavior FollowLaneBehaviorModified(target_speed=LEADER_SPEED)

    ego = new Car following roadDirection from car1.position for -1*DIST_CAR1,
        with blueprint EGO_MODEL,
        with behavior EgoBehavior(target_speed=EGO_SPEED, leaderCar=car1),
        with cte 0,
        with acc 0,
        with collision 0,
        with select_maneuver 1,
        with sensor_created_flag 1,
        with closeIntersection 0,
        with controller None,
        with sensors {"front_rgb": RGBSensor(offset=(0, 2, 1), width=640, height=320),
                    "aerial_rgb": RGBSensor(offset=(0, -10, 4), width=1280, height=640),
                    },    
else:
    ego = new Car at start,
        with blueprint EGO_MODEL,
        with behavior EgoBehavior(target_speed=EGO_SPEED),
        with cte 0,
        with acc 0,
        with collision 0,
        with select_maneuver 1,
        with sensor_created_flag 1,
        with closeIntersection 0,
        with controller None,
        with sensors {"front_rgb": RGBSensor(offset=(0, 2, 1), width=640, height=320),
                    "aerial_rgb": RGBSensor(offset=(0, -10, 4), width=1280, height=640),
                    },  
    

def _collision_callback(_car,collision):
    _car.collision = 1




if dist_to_leader <= 15:
    require ego can see car1

require (distance from start to intersection) < 10 and (distance from start to intersection) > 3 


record ego.speed every 0.1 seconds after 3 seconds to RESULTS_PATH+"/speed.npz"
record ego.acc every 0.1 seconds after 3 seconds to RESULTS_PATH+"/acc.npz"
record ego.select_maneuver every 0.1 seconds after 3 seconds to RESULTS_PATH+"/maneuver.npz"
record ego.distanceToClosest(Car) every 0.1 seconds after 3 seconds to RESULTS_PATH+"/dist.npz"
record ego.cte every 0.1 seconds after 3 seconds to RESULTS_PATH+"/true_cte.npz"
record ego.collision every 0.1 seconds after 3 seconds to RESULTS_PATH+"/collision.npz"