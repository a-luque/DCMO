# os.system(f"scenic -S safety_program_testing.scenic --count 1 --time 500 --2d --param controllers_dir "/mimer/NOBACKUP/groups/naiss2025-22-1298/CMO/experiments/checkpoints/train_faster/" --param weather 'ClearNoon' --param results_path "test_controller_sim_results/0" --param dist_car 30 --param intersec 0 --param speed_car 4 --param monitor /mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/safety_monitor_1/weights_1000.npy")

import warnings
warnings.filterwarnings("ignore") 

import sys
sys.path.append("..")

import numpy as np 
import os
from PIL import Image

import torch
import carla
import torch.nn.functional as F
from torchvision import transforms
from scipy.special import expit

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
    OUT_REACH_DIST = DIST_CAR1 + 10
else:
    OUT_REACH_DIST = DIST_CAR1 + 25


EGO_SPEED = 5
LEADER_SPEED = int(globalParameters.speed_car)
THROTTLE_ACTION = 0.5
MAX_SPEED = 8
BRAKE_ACTION = 1.0
EGO_TO_LEADER = Range(-15, -10)
EGO_BRAKING_THRESHOLD = 7
EGO_ACCELERATION_THRESHOLD = OUT_REACH_DIST
TARGET_SPEED_FOR_TURNING = 3 # KM/H
TRIGGER_DISTANCE_TO_SLOWDOWN = 6 # FOR TURNING AT INTERSECTIONS
# CONTROLLER_PATH = globalParameters.controller_dir
CONTROLLERS_FOLDER = globalParameters.controllers_dir

controllers_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(f"{CONTROLLERS_FOLDER}")) for f in fn]
controllers_paths.sort()

RESULTS_PATH = globalParameters.results_path

monitor_model = None
if globalParameters.monitor != "":
    with open(globalParameters.monitor, 'rb') as f:
        monitor_model = np.load(f) 



    
    
        
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
        if leaderCar is not None:
            if distance from self.position to leaderCar.initialPos < 20:
                egoFollow = True

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
                    #print(self.select_maneuver)
                if egoFollow:
                    select_maneuver = leaderCar.select_maneuver

                elif leaderCar is not None:
                    # before ego car follows leading car maneuver, it just goes straight
                    all_maneuvers = current_lane.maneuvers
                    straight_maneuver = filter(lambda i: i.type == ManeuverType.STRAIGHT, all_maneuvers)

                    if straight_maneuver is not None:
                        select_maneuver = Uniform(*straight_maneuver)
                    else:
                        select_maneuver = Uniform(*maneuvers)

            elif len(current_lane.maneuvers) > 0:
                select_maneuver = Uniform(*current_lane.maneuvers)
                # print(select_maneuver)
            # else:
            #     take SetBrakeAction(1.0)
            #     break
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
                #self.select_maneuver = select_maneuver
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
            # self.select_maneuver = 1 # out of intersection and straight road again
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
            if distance from self to leaderCar < EGO_BRAKING_THRESHOLD:
                take SetBrakeAction(1.0)
                throttle = 0
            else:
                take RegulatedControlAction(throttle, current_steer_angle, past_steer_angle)
        
        else:
            take RegulatedControlAction(throttle, current_steer_angle, past_steer_angle)
        past_steer_angle = current_steer_angle
        past_speed = current_speed






behavior ControllerBehavior(target_speed = 10, controller_path = None, leaderCar = None):
    past_steer_angle = 0
    past_speed = 0 # making an assumption here that the agent starts from zero speed
    
    original_target_speed = target_speed
    TARGET_SPEED_FOR_TURNING = 3 # KM/H


    TRIGGER_DISTANCE_TO_SLOWDOWN = 10 
    current_lane = self._lane
    if current_lane.maneuvers != ():
        nearby_intersection = current_lane.maneuvers[0].intersection
        if nearby_intersection == None:
            nearby_intersection = current_lane.centerline[-1]
    else:
        nearby_intersection = current_lane.centerline[-1]

    # Monitor trigger
    if not monitor_model is None:
        boolIntersection = int(distance from self to nearby_intersection < 10)
        # boolIntersection = int(distIntersection < 10)
        distLeader = distance from self to leaderCar
        if leaderCar is None:
            speedLeader = 0
        else: 
            speedLeader = leaderCar.speed

        input_features = np.concatenate((self.weather, np.array([boolIntersection, distLeader, speedLeader, 1.])))
        input_features = np.expand_dims(input_features, axis=0)
        index_cont = np.argmax(expit(np.dot(input_features, monitor_model.T))[0])
        self.p = expit(np.dot(input_features, monitor_model.T))[0][index_cont]
        print(f"Safe probability: {self.p} (for index {index_cont})")
        # self.isSafe = (self.p >= 0.5)
        # print(f"Is it safe? {self.isSafe}")


    
    # instantiate longitudinal and lateral controllers
    # _lon_controller, _lat_controller = simulation().getLaneFollowingControllers(self)
    dt = simulation().timestep
    _lon_controller = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
    _lat_controller_turn = PIDLateralController(K_P=0.8, K_D=0.2, K_I=0.0, dt=dt)
    _lat_controller_straight = PIDLateralController(K_P=0.2, K_D=0.1, K_I=0.0, dt=dt)


    controllers = []
    for controller_path in controllers_paths:
        controller, _ = load_model(controller_path, device)
        controllers += [controller]
    transform = get_test_transform(112, 224)

    # self.select_maneuver = 1
    # Features for monitor
    if not monitor_model is None:
        self.weather = np.array([
            self.carlaActor.get_world().get_weather().cloudiness,
            self.carlaActor.get_world().get_weather().precipitation,
            self.carlaActor.get_world().get_weather().precipitation_deposits,
            self.carlaActor.get_world().get_weather().wind_intensity,
            self.carlaActor.get_world().get_weather().sun_azimuth_angle,
            self.carlaActor.get_world().get_weather().sun_altitude_angle,
            self.carlaActor.get_world().get_weather().fog_density,
            self.carlaActor.get_world().get_weather().fog_distance,
            self.carlaActor.get_world().get_weather().wetness,
            self.carlaActor.get_world().get_weather().fog_falloff,
            self.carlaActor.get_world().get_weather().scattering_intensity,
            self.carlaActor.get_world().get_weather().mie_scattering_scale,
            self.carlaActor.get_world().get_weather().rayleigh_scattering_scale,
            self.carlaActor.get_world().get_weather().dust_storm  
        ])

    while True:
        # Monitor prediction
        if distance from self to intersection < TRIGGER_DISTANCE_TO_SLOWDOWN:
            intersec = 1
        else: 
            intersec = 0
        if not leaderCar is None:
            dist = distance to leaderCar
            speed = leaderCar.speed
        else:
            dist = 100
            speed = 0
            
        cont_index_prev = self.cont_index
        x = np.concatenate([self.weather, np.array([intersec]), np.array([dist]), np.array([speed]), np.array([1.])])
        probs = expit(np.dot(x, monitor_model.T))
        self.cont_index = np.argmax(probs)
        self.p = probs[self.cont_index]
        # if cont_index != cont_index_prev:
        #     print(f"Using controller {cont_index}")

        # Controller behaviour
        current_lane = self._lane
        if current_lane:
            current_centerline = current_lane.centerline

        """
        if self in network.intersectionRegion:
            if leaderCar is None:
                maneuvers = current_lane.maneuvers
                select_maneuver = Uniform(*maneuvers)
            else:
                select_maneuver = leaderCar.select_maneuver
            self.select_maneuver = select_maneuver
        else:
            self.select_maneuver = 1
        """
        intersection_flag = True
        if distance from self to intersection < TRIGGER_DISTANCE_TO_SLOWDOWN and intersection_flag:
            intersection_flag = False
            if leaderCar is not None:
                select_maneuver = leaderCar.select_maneuver
                if not select_maneuver is None:
                    self.select_maneuver = select_maneuver
                else:
                    self.select_maneuver = 1
            else: 
                if not current_lane is None:
                    maneuvers = current_lane.maneuvers
                    self.select_maneuver = Uniform(*maneuvers)
                else:
                    self.select_maneuver = 1
        else:
            if intersection_flag: 
                self.select_maneuver = 1
        if distance from self to intersection > TRIGGER_DISTANCE_TO_SLOWDOWN:
            intersection_flag = True
        
        if not isinstance(self.select_maneuver, int):
            self.select_maneuver = self.select_maneuver.type.value


        front_img = self.sensors["front_rgb"]._lastObservation
        if isinstance(front_img, np.ndarray):

            input_img = Image.fromarray(front_img)
            result = predict_single(controllers[self.cont_index], input_img, self.select_maneuver, device, transform, 100.0)
            cte_pred = result['cte']
            if distance from self to intersection < TRIGGER_DISTANCE_TO_SLOWDOWN:
                if self.select_maneuver == 2:
                    cte_pred -= 0.25
                if self.select_maneuver == 3:
                    cte_pred += 0.25
            dist_pred = result['distance_m']
            # print(self.select_maneuver, cte_pred, dist_pred)

        else:
            cte_pred, dist_pred = 0, 0

        if self.speed is not None:
            current_speed = self.speed
        else:
            current_speed = past_speed

        
        if current_lane:
            self.cte = current_centerline.signedDistanceTo(self.position)
        # print(self.select_maneuver, self.cte)

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

        



# so the context does not change
behavior KeepDistance(speed=10):
    try: 
        do FollowLaneBehavior(speed)
    interrupt when self.distanceToClosest(Object) > OUT_REACH_DIST:
        take SetBrakeAction(0.8), SetThrottleAction(0)




behavior EgoBehavior(target_speed = 10, leaderCar=None):
    
    if self.sensor_created_flag:
        collision_bp = self.carlaActor.get_world().get_blueprint_library().find('sensor.other.collision')
        collision_sensor = self.carlaActor.get_world().spawn_actor(collision_bp, carla.Transform(),
            attach_to=self.carlaActor,
            attachment_type=carla.AttachmentType.Rigid)
        collision_sensor.listen(lambda col: _collision_callback(self,col))

        self.collision_sensor = collision_sensor
        self.sensor_created_flag = False

    if monitor_model is None:
        do ControllerBehavior(target_speed, leaderCar)
    else:
        try:
            do ControllerBehavior(target_speed=target_speed, leaderCar=leaderCar)
        interrupt when not self.isSafe:
            print("-- Monitor: Not safe! Switching to SafeBehavior.")
            take SetBrakeAction(1.0)
            do EgoBehavior2(target_speed = target_speed, leaderCar=leaderCar)

behavior EgoBehavior2(target_speed = 10, leaderCar=None):
    try:
        do FollowLaneBehaviorModified(target_speed=target_speed, leaderCar=leaderCar)
    interrupt when self.isSafe:
        print("-- Monitor: Safe! Switching back to CNN Controller.")
        do EgoBehavior(target_speed=target_speed, leaderCar=leaderCar)


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
            with select_maneuver 1,
            with behavior FollowLaneBehaviorModified(target_speed=LEADER_SPEED)

    ego = new Car following roadDirection from car1.position for -1*DIST_CAR1,
        with blueprint EGO_MODEL,
        with behavior EgoBehavior(target_speed=EGO_SPEED, leaderCar=car1),
        with cte 0,
        with acc 0,
        with collision 0,
        with select_maneuver 1,
        with isSafe 1,
        with sensor_created_flag 1,
        with p 0,
        with cont_index 0,
        with weather np.zeros(14),
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
        with isSafe 1,
        with sensor_created_flag 1,
        with p 0,
        with cont_index 0,
        with weather np.zeros(14),
        with sensors {"front_rgb": RGBSensor(offset=(0, 2, 1), width=640, height=320),
                    "aerial_rgb": RGBSensor(offset=(0, -10, 4), width=1280, height=640),
                    },  
    

def _collision_callback(_car,collision):
    _car.collision = 1

if DIST_CAR1 <= 50:
    require ego can see car1

if INTERSEC:
    require (distance from start to intersection) < 10 and (distance from start to intersection) > 3 
else:
    require (distance from start to intersection) > 10  


record ego.speed every 0.1 seconds after 3 seconds to RESULTS_PATH+"/speed.npz"
record ego.acc every 0.1 seconds after 3 seconds to RESULTS_PATH+"/acc.npz"
record ego.select_maneuver every 0.1 seconds after 3 seconds to RESULTS_PATH+"/maneuver.npz"
record ego.distanceToClosest(Car) every 0.1 seconds after 3 seconds to RESULTS_PATH+"/dist.npz"
record ego.cte every 0.1 seconds after 3 seconds to RESULTS_PATH+"/true_cte.npz"
record ego.p every 0.1 seconds after 3 seconds to RESULTS_PATH+"/prob_monitor.npz"
record ego.cont_index  every 0.1 seconds after 3 seconds to RESULTS_PATH+"/cont_selected.npz"
record ego.collision every 0.1 seconds after 3 seconds to RESULTS_PATH+"/collision.npz"
# record ego.speed
