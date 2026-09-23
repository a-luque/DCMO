#!/bin/sh
until timeout 1800 bash /home/luque/Documents/Carla0916/CarlaUE4.sh -RenderOffScreen; do #  -RenderOffScreen -quality-level=low -benchmark -fps=10; do
    echo "Server 'CarlaUE4' crashed with exit code $?.  Respawning.." >&2
    sleep 15
    pkill -f CarlaUE4-
    sleep 3
    pkill -f CarlaUE4-
    sleep 3
    pkill -f CarlaUE4.sh
    sleep 3
    pkill -f CarlaUE4.sh
    sleep 5
done
