#!/bin/sh
until bash /mimer/NOBACKUP/groups/naiss2025-22-1298/carla0916/Carla_0_9_16/CarlaUE4.sh -RenderOffScreen; do #  -RenderOffScreen -quality-level=low -benchmark -fps=10; do
    echo "Server 'CarlaUE4' crashed with exit code $?.  Respawning.." >&2
    sleep 1
done