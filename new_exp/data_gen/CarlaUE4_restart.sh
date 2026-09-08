#!/bin/sh
until bash /proj/berzelius-2026-227/users/x_alluq/Carla_0_9_16/CarlaUE4.sh -RenderOffScreen; do #  -RenderOffScreen -quality-level=low -benchmark -fps=10; do
    echo "Server 'CarlaUE4' crashed with exit code $?.  Respawning.." >&2
    sleep 1
done