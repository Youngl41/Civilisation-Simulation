#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:55:53 2018

@author: Young
"""



#==============================================================================
# Import Modules
#==============================================================================
import sys
import numpy as np
import pandas as pd
import random as rnd
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append('/Users/Young/Documents/Capgemini/Learning/Machine Learning/Reinforcement Learning/Civilisation-Simulation/Scripts/Simulation Functions')
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
        if initial_state:
            self.positions = initial_state
            self.time = 0
        print ('\nGridWorld created!\n')

    @classmethod
    def create(cls, grid, population_count):
        return cls(grid, population_count)
    
    @classmethod
    def create_from_initial_state(cls, grid, population_count, initial_state):
        return cls(grid, population_count, initial_state)
    
    def populate(self):        
        # Check if grid has already been populated
        if self.time >= 0:
            raise Exception('ERROR: grid has already been populated.')
        elif self.time < -1:
            raise Exception('ERROR: unknown error.')
            
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
        print ('\nGridWorld populated.\n')
        
    def reset(self):
        self.time = -1
        self.positions = []
        
    def play(self, duration = 1):
        current_time = deepcopy(self.time)
        while self.time < current_time + duration:
            # Move each position randomly
            positions = []
            for index, position in self.positions[-1]:
                new_pos = rnd.sample(find_possible_actions(self.grid, position), 1)[0]
                positions.append([index, new_pos])
            
            # Update parameters
            self.positions.append(positions)
            self.time = self.time+1
        
    def show_board(self, time = None):
        '''
        Displays and updates board with every blob same colour, except for when 
        they stack.
        '''
        if not time:
            time = self.time
        
        # Make clean board
        valid_grid = 1-np.isnan(grid)
        board = valid_grid-1
            
        # Update board
        for index, position in self.positions[time]:
            pos_h = position[0]
            pos_w = position[1]
            board[pos_h][pos_w] = board[pos_h][pos_w] + 1
        
        # Print board
        plt.imshow(board, interpolation='nearest')
        title_string = 'Board at time=' + str(time)
        plt.title(title_string)
        #plt.grid(color = 'white')
        plt.colorbar()
        plt.show()
    
    def update_board(self, time = None):
        '''
        Updates and returns board with every blob same colour, except for when 
        they stack.
        '''
        if not time:
            time = self.time
        
        # Make clean board
        valid_grid = 1-np.isnan(grid)
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
            ax.imshow(g.update_board(time=t), interpolation='nearest')
            ax.grid(color='white')
            ax.set_axis_off()
        
        fig, ax = plt.subplots(figsize=figsize)
        animation = FuncAnimation(fig, update_frame, repeat=False, 
                                  frames=np.arange(start_time, end_time))
        
        # Gif
        if save_path[-4:].lower() == '.gif':
            animation.save(save_path, dpi=60, fps=5, writer='imagemagick')
        elif save_path[-4:].lower() == '.mp4':
            writer = animation.FFMpegFileWriter(fps=fps)
            animation.save(save_path, dpi=60, writer = writer)
        else:
            error_msg = 'ERROR: unknown file type "' + save_path.split('.')[-1] + '".'
            raise Exception(error_msg)


# Test
rnd.seed(42)
grid = np.zeros((5,7))
grid[1][5] = 1
grid[0][1] = np.nan
print (grid)

g = GridWorld.create(grid, 1)
g.populate()
g.show_board()
%time g.play(duration=10000)
g.show_board()



g.reset()
g.populate()
g.show_board()

plt.imshow(g.board, interpolation='nearest')
plt.colorbar()
plt.show()


def update_frame(i):
    g.play(duration=1)
    ax.imshow(g.update_board(), interpolation='nearest')
    ax.grid(color='white')
    ax.set_axis_off()


fig, ax = plt.subplots(figsize=(5, 8))
anim = FuncAnimation(fig, update_frame, repeat=False, frames=np.arange(0, 50))

%time anim.save('/Users/Young/Documents/Capgemini/Learning/Machine Learning/Reinforcement Learning/Civilisation-Simulation/Temp/temp.gif', dpi=60, fps=5, writer='imagemagick')

# Save
fps = 5
#dpi = 10
save_name = '/Users/Young/Documents/Capgemini/Learning/Machine Learning/Reinforcement Learning/Civilisation-Simulation/Temp/'
writer=animation.FFMpegFileWriter(fps=fps)
 



grid = np.zeros((5,7))
grid[1][5] = 1
grid[0][1] = np.nan

# Valid grid
valid_grid = 1-np.isnan(grid)

# Valid oordinates
h, w = np.where(valid_grid > 0)
valid_coords = []
for h_index, w_index in zip(h,w):
    valid_coords.append((h_index, w_index))

