#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 21:14:35 2018

@author: Young
"""



#==============================================================================
# Import Modules
#==============================================================================
import os
import sys
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt

main_dir = '/Users/Young/Documents/Capgemini/Learning/Machine Learning/Reinforcement Learning/Civilisation-Simulation'
sys.path.append(os.path.join(main_dir, 'Scripts/Simulation Functions'))
sys.path.append(os.path.join(main_dir, 'Scripts'))
from Blob import Blob
from GridWorld import GridWorld
from simulation_utility import sample_coords
from simulation_utility import find_possible_actions


# =============================================================================
# Unit Test
# =============================================================================
# Random Initialiser
rnd.seed(42)

# Create GridWorld
grid = np.zeros((5,7))
grid[1][5] = 1
grid[0][1] = np.nan
print (grid)
g = GridWorld.create(grid, 10)

# Place blobs
g.populate()
g.show_board(figsize=(8,5))#, reward_overlay = True)

# Play multiple times
%time g.play(duration=1)
g.show_board(figsize=(8,5))#, reward_overlay = True)

# Save video
g.save_video(save_path = os.path.join(main_dir, 'Temp/temp.gif'),
             start_time = 0, end_time = 20, 
             fps = 5, dpi = 60, figsize=(5,8))
g.save_video(save_path = os.path.join(main_dir, 'Temp/temp.mp4'),
             start_time = 0, end_time = 20, 
             fps = 10, dpi = 60, figsize=(5,8))


# =============================================================================
# Example 1: 
# =============================================================================
rnd.seed(43)
grid = np.zeros((3,4))
grid[0][3] = 1
grid[2][3] = -10
grid[1][2] = np.nan
grid[1][1] = np.nan
print (grid)

# Populate gridworld
g2 = GridWorld.create(grid=grid, population_count=1, 
                      initial_state = [[[0, (2,0)]]])
g2.populate()
g2.show_board(figsize=(8,5), reward_overlay = True)

# Iterate
itertion_number = 100
rewards_list = []
for play_number in range(itertion_number):
    g2.play(duration=30, beta=0.8)
    rewards_list.append(g2.total_rewards)
    g2.reset(population_count=1, inherit=True)
    #print (g2.blobs[0].value_grid)

plt.plot(rewards_list)

g2.play(duration=30, beta=0.8)
g2.save_video(save_path = os.path.join(main_dir, 'Temp/example_1.mp4'),
             start_time = 0, end_time = 30, 
             fps = 3, dpi = 60, figsize=(5,8))


# =============================================================================
# Example 2: Pitfall Road
# =============================================================================
rnd.seed(43)
grid = np.zeros((3,5))
grid[1][4] = 1
grid[0] = [-10, -10, -10, -10, -10]
grid[2] = [-10, -10, -10, -10, -10]
print(grid)

# Populate gridworld
g2 = GridWorld.create(grid=grid, 
                      population_count=1, 
                      initial_state = [[[0, (1,0)]]], 
                      verbose=False)
g2.populate()
g2.show_board(figsize=(8,5), reward_overlay = True)

# Iterate
itertion_number = 10
rewards_list = []
for play_number in range(itertion_number):
    g2.play(duration=30, beta=0.8)
    rewards_list.append(g2.total_rewards)
    g2.reset(population_count=1, inherit=True)
    #print (g2.blobs[0].value_grid)

plt.plot(rewards_list)

# Time=0 error
g2.update_board(time=0)


g2 = GridWorld.create(grid=grid, 
                      population_count=1, 
                      initial_state = [[[0, (2,0)]]], 
                      verbose=False)
g2.populate()
g2.show_board(figsize=(8,5), reward_overlay = True)

g2.play(duration=30, beta=0.99)
g2.show_board(figsize=(8,5), reward_overlay = True)
g2.reset(population_count=1, inherit=True)

g2.show_board(time=0,figsize=(8,5), reward_overlay = True)
g2.positions


# =============================================================================
# Example 3: Valley Treasure
# =============================================================================
rnd.seed(43)
grid = np.zeros((5,5))
grid[2][4] = 1
grid[0][4] = 100
grid[1] = [-10, -10, 0, -10, -10]
grid[3] = [-10, -10, -10, -10, -10]

# Populate gridworld
g1 = GridWorld.create(grid=grid, population_count=1, 
                      initial_state = [[[0, (2,0)]]])
g1.populate()
g1.show_board(figsize=(8,5), reward_overlay = True)

# Iterate
g1.play(duration=30, beta=0.99)
g1.show_board(figsize=(8,5), reward_overlay = True)
print ('Total Reward:\t', g1.total_rewards)
g1.reset(population_count=1, inherit=True)
print (g1.blobs[0].value_grid)

# Iterate
itertion_number = 100
rewards_list = []
for play_number in range(itertion_number):
    g1.play(duration=30, beta=0.9)
    rewards_list.append(g1.total_rewards)
    g1.reset(population_count=1, inherit=True)
    #print (g1.blobs[0].value_grid)

plt.plot(rewards_list)
plt.show()
plt.imshow(g1.blobs[0].value_grid)
plt.show()