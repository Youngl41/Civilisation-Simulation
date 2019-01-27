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

main_dir = '/Users/young/Documents/projects/genesis_ai/blobs'
sys.path.append(os.path.join(main_dir, 'scripts/utility'))
sys.path.append(os.path.join(main_dir, 'scripts'))
from blob import Blob
from gridworld import GridWorld
from util_simulation import sample_coords
from util_simulation import find_possible_actions


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
                      initial_state = [[[0, (2,0)]]],
                      verbose=True)
g2.populate()
g2.show_board(figsize=(8,5), reward_overlay = True)

# Iterative training
g2.play_iteratively(iteration_count=100, beta=0.5, 
                         plan_flag=False, population_count=1, verbose=1)
print('Number of iterations:\t', g2.iteration_count)

# Play one game with plan
g2.play(duration=30, beta=0.8, plan_flag=True)
print('Total reward:\t\t', g2.total_rewards)
print('\nValue grid for first blob\n')
print(np.round(g2.blobs[0].value_grid, 2))

# Save plan in action
g2.save_video(save_path = os.path.join(main_dir, 'results/example_1.mp4'),
             start_time = 0, end_time = 30, 
             fps = 3, dpi = 20, figsize=(5,8))


# =============================================================================
# Example 2: Pitfall Road
# =============================================================================
rnd.seed(43)
grid = np.zeros((3,5))#-0.5
grid[1][4] = 5
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

# Iterative training
g2.play_iteratively(iteration_count=500, beta=0.5, 
                         plan_flag=False, population_count=1, verbose=1)
print('Number of iterations:\t', g2.iteration_count)

# Play one game with plan
g2.play(duration=30, beta=0.8, plan_flag=True)
print('Total reward:\t\t', g2.total_rewards)
print('\nValue grid for first blob\n')
print(np.round(g2.blobs[0].value_grid, 2))

# Save plan in action
g2.save_video(save_path = os.path.join(main_dir, 'results/example_2.mp4'),
             start_time = 0, end_time = 30, 
             fps = 3, dpi = 60, figsize=(5,8))

# Iterate manually
g2.play(duration=1, beta=0.5, plan_flag=False)
g2.show_board(figsize=(8,5), reward_overlay = True)
print (g2.blobs[0].value_grid)
print ('Total Reward:\t', g2.total_rewards)
g2.reset(population_count=1, inherit=True)
plt.plot(g2.iteration_rewards)


# =============================================================================
# Example 3: Valley Treasure
# =============================================================================
rnd.seed(43)
grid = np.zeros((5,5))#-0.5
grid[2][4] = 1
grid[0][4] = 10
grid[1] = [-10, -10, 0, -10, -10]
grid[3] = [-10, -10, -10, -10, -10]

# Populate gridworld
g3 = GridWorld.create(grid=grid, population_count=1, 
                      initial_state = [[[0, (2,0)]]],
                      verbose=False)
g3.populate()
g3.show_board(figsize=(8,5), reward_overlay = True)

# Iterative training
g3.play_iteratively(iteration_count=100, beta=0.9, alpha=0.1, reroll_chance=0.3,
                         plan_flag=False, population_count=1, verbose=1)
print('\nNumber of iterations:\t', g3.iteration_count)
print('\nValue grid for first blob\n')
print(np.round(g3.blobs[0].value_grid, 2).astype(int))

# Play one game with plan
g3.play(duration=30, beta=0.9, plan_flag=True)
print('Total reward:\t\t', g3.total_rewards)
print('\nValue grid for first blob\n')
print(np.round(g3.blobs[0].value_grid, 0))

# Save plan in action
g3.save_video(save_path = os.path.join(main_dir, 'results/example_3.mp4'),
             start_time = 0, end_time = 30, 
             fps = 5, dpi = 60, figsize=(5,8))


# =============================================================================
# Example 4: Valley Hidden Treasure
# =============================================================================
rnd.seed(43)
grid = np.zeros((5,5))-0.5
grid[2][4] = 1
grid[0][4] = 100
grid[1] = [-10, -10, -10, -10, -10]
grid[3] = [-10, -10, -10, -10, -10]

# Populate gridworld
g3 = GridWorld.create(grid=grid, population_count=1, 
                      initial_state = [[[0, (2,0)]]],
                      verbose=False)
g3.populate()
g3.show_board(figsize=(8,5), reward_overlay = True)

# Iterative training
g3.play_iteratively(iteration_count=10000, beta=0.9, alpha=0.1
                         plan_flag=False, population_count=1, verbose=1)
print('\nNumber of iterations:\t', g3.iteration_count)

# Play one game with plan
g3.play(duration=30, beta=0.8, plan_flag=True)
print('Total reward:\t\t', g3.total_rewards)
print('\nValue grid for first blob\n')
print(np.round(g3.blobs[0].value_grid, 0))

# Save plan in action
g3.save_video(save_path = os.path.join(main_dir, 'results/example_3.mp4'),
             start_time = 0, end_time = 30, 
             fps = 3, dpi = 60, figsize=(5,8))

# Plot reward iterations in groups
df = pd.DataFrame(g3.iteration_rewards)
df.columns = ['total_reward']
group_size = 100
df.loc[:,'grouped'] = (df.index / group_size).astype(int)
df.groupby('grouped')['total_reward'].agg('mean').plot(figsize = (9,5))
plt.title('Total reward after each iteration of play')
plt.xlabel('Iteration count (in ' + str(group_size) + "'s)")
plt.ylabel('Total reward')
plt.show()