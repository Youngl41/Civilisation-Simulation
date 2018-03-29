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

sys.path.append('/Users/Young/Documents/Capgemini/Learning/Machine Learning/Reinforcement Learning/Civilisation Simulation/Scripts/Simulation Functions')
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
        
    def show_grid(self, time = None):
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
        



        
# Test
rnd.seed(42)
grid = np.zeros((5,7))
grid[1][5] = 1
grid[0][1] = np.nan
print (grid)

g = GridWorld.create(grid, 1)
g.populate()
g.show_grid()
%time g.play(duration=10000)
g.show_grid()



g.reset()
g.populate()
g.show_grid()

plt.imshow(g.board, interpolation='nearest')
plt.colorbar()
plt.show()









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

