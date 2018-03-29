#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:38:07 2018

@author: Young
"""

#==============================================================================
# Import Modules
#==============================================================================
import numpy as np
import pandas as pd
import random as rnd
from copy import deepcopy
import matplotlib.pyplot as plt


#==============================================================================
# Utility Functions
#==============================================================================
def sample_coords(grid, sample_num):
    '''
    Samples N=sample_num of valid coordinates from grid.
    
    Args:
        grid: numpy array n x m
        sample_num: integer
        
    Returns:
        list of N 2-tuple valid courdinates from grid
        
    Example:
        >>> grid = np.zeros((5,7))
        >>> grid[1][5] = np.nan
        >>> sample_num = 3
        >>> sample_coords(grid, sample_num)
    '''
    # Valid grid
    valid_grid = 1-np.isnan(grid)
    
    # Valid oordinates
    h, w = np.where(valid_grid > 0)
    valid_coords = []
    for h_index, w_index in zip(h,w):
        valid_coords.append((h_index, w_index))
    
    # Sample
    rand_coords = rnd.sample(valid_coords, sample_num)
    return rand_coords

def find_possible_actions(grid, coord):
    '''
    Finds valid actions from coordinate.
    
    Args:
        grid: numpy array n x m
        coord: 2-tuple
        
    Returns:
        list of 2-tuples
        
    Example:
        >>> grid = np.zeros((5,7))
        >>> grid[1][5] = np.nan
        >>> coord = (1,4)
        >>> find_possible_actions(grid, coord)
    '''
    # Get all actions coords
    no_move_coord = coord
    move_up_coord = (coord[0]-1, coord[1])
    move_down_coord = (coord[0]+1, coord[1])
    move_left_coord = (coord[0], coord[1]-1)
    move_right_coord = (coord[0], coord[1]+1)
    all_action_coords = [no_move_coord,
                         move_up_coord, 
                         move_down_coord,
                         move_left_coord,
                         move_right_coord]
    
    # Remove out of grid coords
    h_max = grid.shape[0]
    w_max = grid.shape[1]
    cleaned_action_coords = []
    for action_coord in all_action_coords:
        if (action_coord[0] < 0) or (action_coord[1] < 0) or \
        (action_coord[0] >= h_max) or (action_coord[1] >= w_max):
            pass
        else: 
            cleaned_action_coords.append(action_coord)
    
    # Remove invalid coords
    valid_grid = 1-np.isnan(grid)
    invalid_h, invalid_w = np.where(valid_grid == 0)
    invalid_coords = []
    for h_index, w_index in zip(invalid_h, invalid_w):
        invalid_coords.append((h_index, w_index))
    
    valid_action_coords = []
    for action_coord in cleaned_action_coords:
        if action_coord in invalid_coords:
            pass
        else:
            valid_action_coords.append(action_coord)

    return valid_action_coords
