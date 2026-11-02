#!/bin/bash
export PYTHONPATH=$PYTHONPATH:src:src/envs/:../safety-gym
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so.2.2:/usr/lib/x86_64-linux-gnu/libGL.so.1
export PATH=$HOME/.pyenv/versions/reward/bin:$PATH
