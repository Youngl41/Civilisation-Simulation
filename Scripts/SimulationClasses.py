

#==============================================================================
# Import Modules
#==============================================================================
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt


#==============================================================================
# Utility Functions
#==============================================================================
def gen_ran_coords(shape, num):
    rand_ixs = rnd.sample(range(np.prod(shape)), num)
    rand_coords = []
    for rand_ix in rand_ixs:
        rand_y = rand_ix // shape[1]
        rand_x = np.mod(rand_ix, shape[1])
        rand_coords.append([rand_x, rand_y])
    return rand_coords


#==============================================================================
# Main
#==============================================================================
class grid():
    # Initialise
    def __init__(self):
        self.status = 0
        self.positions = []        
        
    # Make grid
    def make(self, shape):
        self.shape = shape
        self.positions = []
        self.board = np.zeros(self.shape)
        
    # Generate population
    def gen_pop(self, population_num=1):
        try: 
            self.positions
        except AttributeError:
            raise Exception('ERROR: run grid.make first.')
        if self.status != 0:
            raise Exception('ERROR: grid has already been populated.')
        
        # Generate random states
        self.population_num=population_num
        positions = []
        rand_coords = gen_ran_coords(self.shape, self.population_num)
        for ix, rand_coord in enumerate(rand_coords):
            positions.append([ix, rand_coord])
            
        # Update board
        #self.board[rand_coord[1]][rand_coord[0]] = ix+1
        
        # Update parameters
        self.positions.append(positions)
        self.status = 1

    # Run grid
    def run(self, ):
        if self.status == 0:
            raise Exception('ERROR: grid needs to be populated first.\nrun grid.gen_pop.')
        
        # Run
        for 
        
        # Update parameters
        self.positions.append()
        self.status = self.status + 1
        
    # Show board
    def show(self):
        try: 
            self.board
        except AttributeError:
            raise Exception('ERROR: run grid.make first.')
        plt.imshow(self.board, interpolation='nearest')
        plt.colorbar()
        plt.show()


class GridWorld():
    def __init__(self, grid, population_count, initial_state = None):
        self.grid = grid
        self.population_count = population_count
        self.time = 0
        if initial_state:
            self.positions = initial_state
            self.time = 1

    @classmethod
    def create(cls, grid, population_count):
        return cls(grid, population_count)
    
    @classmethod
    def create_from_initial_state(cls, grid, population_count, initial_state):
        return cls(grid, population_count, initial_state)
    
    def populate(self):
        # Check if grid has already been populated
        if self.time != 0:
            raise Exception('ERROR: grid has already been populated.')
        
        # Generate random states
        positions = []
        rand_coords = gen_ran_coords(grid.shape, self.population_count)
        for ix, rand_coord in enumerate(rand_coords):
            positions.append([ix, rand_coord])
            
        # Update board
        #self.board[rand_coord[1]][rand_coord[0]] = ix+1
        
        # Update parameters
        self.positions.append(positions)
        self.time = 1
        
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


def sample_coords(grid, sample_num):
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

        
# Test
rnd.seed(42)
g = grid()
g.make(shape=(5,7))
g.gen_pop(population_num=3)
g.board

plt.imshow(g.board, interpolation='nearest')
plt.colorbar()
plt.show()



board = g.board
board
g.positions