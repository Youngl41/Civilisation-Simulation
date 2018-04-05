#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:55:53 2018

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

from copy import deepcopy
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegFileWriter

main_dir = '/Users/Young/Documents/Capgemini/Learning/Machine Learning/Reinforcement Learning/Civilisation-Simulation'
sys.path.append(os.path.join(main_dir, 'Scripts/Simulation Functions'))
sys.path.append(os.path.join(main_dir, 'Scripts'))
from Blobs import Blob
from simulation_utility import sample_coords
from simulation_utility import find_possible_actions


#==============================================================================
# Main
#==============================================================================
class GridWorld:
    def __init__(self, grid, population_count, initial_state = None):
        self.grid = grid
        self.population_count = population_count
        self.beginning_population_count = population_count
        self.time = -1
        self.status = 'Not Started'
        self.total_rewards = 0
        if initial_state:
            self.positions = initial_state
            self.beginning_positions = deepcopy(initial_state)
        print ('\nGridWorld created!\n')

    @classmethod
    def create(cls, grid, population_count, initial_state=None):
        if not initial_state:
            return cls(grid, population_count)
        else:
            return cls(grid, population_count, initial_state)
    
    def populate(self):
        # Check if grid has already been populated
        if self.status == 'In progress':
            raise Exception('ERROR: grid has already been populated.')
        elif self.status == 'Finished':
            raise Exception('ERROR: world has ended. Please restart by using\
                            the .reset() command')

        try:
            self.positions
        except AttributeError:
            # Allocate variable
            self.positions = []
            self.beginning_positions = []
            
            # Generate random states
            positions = []
            rand_coords = sample_coords(self.grid, self.population_count)
            for ix, rand_coord in enumerate(rand_coords):
                positions.append([ix, rand_coord])
            
            # Update parameters
            self.positions.append(positions)
            self.beginning_positions.append(positions)
        
        # Initialise blobs
        self.blobs = []
        for blob_name, coords in self.positions[0]:
            self.blobs.append(Blob(name=blob_name, 
                                   time=0, max_age=30, 
                                   coords=coords, 
                                   value_grid=np.zeros(self.grid.shape)))
        
        # Update parameters
        self.time = 0
        self.status = 'In progress'
        print ('\nGridWorld populated.\n')
        
    def reset(self, population_count, inherit=True):
        '''
        Reset has two options:
            (1) Reset with inheritance populates the gridworld to time 0 
                with blobs in original positions and updates value grid.
            (2) Reset with inheritance creates a fresh grid from previous grid 
                and population count must be inserted.
        '''
        
        # If inerhitance is true check if population_count is same as before
        if (inherit == True) and (population_count != self.beginning_population_count):
            exception_string = 'ERROR: Original population_count is, ' + \
                                str(self.beginning_population_count) + \
                                ', does not match the input population_count, '\
                                + str(population_count) + '.'
            raise Exception(exception_string)
        
        # Inherit reset
        self.time = -1
        self.status = 'Not Started'
        self.population_count = population_count
        self.beginning_population_count = population_count
        self.total_rewards = 0
        if inherit == True:
            self.positions = deepcopy(self.beginning_positions)
            
            # Initialise blobs
            blobs = []
            for index, original_positions in enumerate(self.positions[0]):
                blob_name, coords = original_positions
                blobs.append(Blob(name=blob_name, 
                                       time=0, max_age=30, 
                                       coords=coords, 
                                       value_grid=self.blobs[index].value_grid))
            
            # Update parameters
            self.blobs = blobs
            self.time = 0
            self.status = 'In progress'            
            print ('\nGridWorld reset and populated with inheritance.\n')
                    
        # Don't inherit
        else:
            del self.positions   
            del self.beginning_positions
            del self.blobs
            print ('\nGridWorld reset.\n')
    
    def play(self, duration=1, beta=0.5):
        current_time = deepcopy(self.time)
        
        if self.status == 'Finished':
            raise Exception('ERROR: game has finished. Please restart by using\
                            the .reset() command')
        
        while (self.time < current_time + duration) and (self.status != 'Finished'):
#            # Move each position randomly
#            positions = []
#            for index, position in self.positions[-1]:
#                new_pos = rnd.sample(find_possible_actions(self.grid, position), 1)[0]
#                positions.append([index, new_pos])
            # Move each blob using iterative q-learning
            positions = []
            blobs = []
            for blob in self.blobs:
                new_pos = blob.check_actions_and_update_values(grid=self.grid, beta=beta)
                positions.append([blob.name, new_pos])
                blob.coords = new_pos
                blobs.append(blob)
            
            # Update parameters
            self.positions.append(positions)
            self.blobs = blobs
            self.time = self.time+1
            
            # Update total rewards
            for index, position in self.positions[-1]:
                self.total_rewards = self.total_rewards + self.grid[position[0]][position[1]]
            
            # Check new positions
            for blob in self.blobs:
                if self.time >= blob.max_age:
                    blob.status = 'Died of old age'
                    print ('Blob name',str(blob.name), 'dead. \t-', blob.status)
                    self.population_count = self.population_count - 1

            # Check world end
            if self.population_count <= 0:
                print ('\nGame Finished.\n')
                self.status = 'Finished'
    
    def update_board(self, time = None):
        '''
        Updates and returns board with every blob same colour, except for when 
        they stack.
        '''
        if not time:
            time = self.time
        
        # Make clean board
        valid_grid = 1-np.isnan(self.grid)
        board = valid_grid-1
            
        # Update board
        for index, position in self.positions[time]:
            pos_h = position[0]
            pos_w = position[1]
            board[pos_h][pos_w] = board[pos_h][pos_w] + 1
        
        return board
    
    def show_board(self, time=None, figsize=None, reward_overlay=False):
        '''
        Displays and updates board with every blob same colour, except for when 
        they stack.
        '''
        board = self.update_board(time=time)
        
        # Print board
        if figsize:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
        plt.imshow(board, interpolation='nearest', cmap='terrain')
        title_string = 'Board at time=' + str(time)
        plt.title(title_string)
        #plt.grid(color = 'white')
        plt.colorbar()
        
        # Add reward labels
        if reward_overlay:
            for (j,i),label in np.ndenumerate(self.grid):
                plt.text(i,j,label,ha='center',va='center', color='orange')
                plt.text(i,j,label,ha='center',va='center', color='orange')
        plt.show()
        
    def save_video(self, save_path, start_time = 0, end_time = 50, 
                   fps = 5, dpi = 60, figsize=(5,8)):
        '''
        Saves video of frames from start_time to end_time
        they stack.
        '''
        def update_frame(t):
            ax.imshow(self.update_board(time=t), interpolation='nearest')
            ax.grid(color='white')
            ax.set_axis_off()
        
        fig, ax = plt.subplots(figsize=figsize)
        animation = FuncAnimation(fig, update_frame, repeat=False, 
                                  frames=np.arange(start_time, end_time))
        
        # Gif
        if save_path[-4:].lower() == '.gif':
            animation.save(save_path, dpi=60, fps=fps, writer='imagemagick')
        elif save_path[-4:].lower() == '.mp4':
            writer = FFMpegFileWriter(fps=fps)
            animation.save(save_path, dpi=60, writer = writer)
        else:
            error_msg = 'ERROR: unknown file type "'\
                        + save_path.split('.')[-1] + '".'
            raise Exception(error_msg)


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
print (grid)

# Populate gridworld
g2 = GridWorld.create(grid=grid, population_count=1, 
                      initial_state = [[[0, (1,0)]]])
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

# Verbose
# Time=0 error

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


# =============================================================================
# 
# =============================================================================

rnd.seed(42)

# Create GridWorld
grid = np.zeros((5,7))
grid[1][5] = 1
grid[0][1] = np.nan
grid[1][1] = np.nan
grid[2][1] = np.nan
grid[3][1] = np.nan
grid[4][1] = np.nan
print (grid)
g = GridWorld.create(grid, 3)

# Place blobs
g.populate()
g.show_board(reward_overlay=True)
