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
from simulation_utility import sample_coords
from simulation_utility import find_possible_actions


#==============================================================================
# Main
#==============================================================================
class GridWorld:
    def __init__(self, grid, population_count, initial_state = None):
        self.grid = grid
        self.population_count = population_count
        self.time = -1
        self.status = 'Not Started'
        if initial_state:
            self.positions = initial_state
            self.time = 0
            self.status = 'In progress'
        print ('\nGridWorld created!\n')

    @classmethod
    def create(cls, grid, population_count):
        return cls(grid, population_count)
    
    @classmethod
    def create_from_initial_state(cls, grid, population_count, initial_state):
        return cls(grid, population_count, initial_state)
    
    def populate(self):
        # Check if grid has already been populated
        if self.status == 'In progress':
            raise Exception('ERROR: grid has already been populated.')
        elif self.status == 'Finished':
            raise Exception('ERROR: world has ended. Please restart by using\
                            the .reset() command')
            
        # Allocate variable
        self.positions = []
        
        # Generate random states
        positions = []
        rand_coords = sample_coords(self.grid, self.population_count)
        for ix, rand_coord in enumerate(rand_coords):
            positions.append([ix, rand_coord])
        
        # Update parameters
        self.positions.append(positions)
        self.time = 0
        self.status = 'In progress'
        print ('\nGridWorld populated.\n')
        
    def reset(self):
        self.time = -1
        self.positions = []
    
    def play(self, duration = 1):
        current_time = deepcopy(self.time)
        while (self.time < current_time + duration) and (self.status != 'Finished'):
            # Move each position randomly
            positions = []
            for index, position in self.positions[-1]:
                new_pos = rnd.sample(find_possible_actions(self.grid, position), 1)[0]
                positions.append([index, new_pos])
                
            # Update parameters
            self.positions.append(positions)
            self.time = self.time+1
            
            # Update rewards
            
            # Check world end
            if self.population_count <= 0:
                print ('\nGame Finished.\n')
                self.status = 'Finished'
        
    def show_board(self, time=None, figsize=None, reward_overlay=False):
        '''
        Displays and updates board with every blob same colour, except for when 
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
g.show_board()

# Play multiple times
%time g.play(duration=100)
g.show_board()

# Save video
g.save_video(save_path = os.path.join(main_dir, 'Temp/temp.gif'),
             start_time = 0, end_time = 50, 
             fps = 5, dpi = 60, figsize=(5,8))
g.save_video(save_path = os.path.join(main_dir, 'Temp/temp.mp4'),
             start_time = 0, end_time = 50, 
             fps = 10, dpi = 60, figsize=(5,8))


# =============================================================================
# Example 1: 
# =============================================================================


# =============================================================================
# Example 2: Pitfall Road
# =============================================================================
rnd.seed(43)
grid = np.zeros((3,5))
grid[1][4] = 1
grid[0] = [-10, -10, -10, -10, -10]
grid[2] = [-10, -10, -10, -10, -10]
print (grid)
g1 = GridWorld.create_from_initial_state(grid, 
                                         population_count=1, 
                                         initial_state = [[[0, (1,0)]]])
g1.show_board(figsize=(8,5), reward_overlay = True)

g1.play(duration=1)
g1.show_board(figsize=(8,5), reward_overlay = True)



g.show_board(figsize=(8,5), reward_overlay = True)





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

# Play multiple times
%time g.play(duration=50)
g.show_board()

# Save video
g.save_video(save_path = os.path.join(main_dir, 'Temp/temp.mp4'),
             start_time = 0, end_time = 50, 
             fps = 10, dpi = 60, figsize=(5,8))



# Valid grid
valid_grid = 1-np.isnan(grid)

# Valid oordinates
h, w = np.where(valid_grid > 0)
valid_coords = []
for h_index, w_index in zip(h,w):
    valid_coords.append((h_index, w_index))

