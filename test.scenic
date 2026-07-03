param map = localPath('carla_map/Town01.xodr')
param carla_map = 'Town01'
model scenic.simulators.carla.model

param render = 1

ego = new Car in intersection

ego = new Car on ego.lane.predecessor

new Pedestrian on visible sidewalk

third = new Car on visible ego.road
require abs((apparent heading of third) - 180 deg) <= 30 deg