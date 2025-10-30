import numpy as np
import matplotlib.pyplot as plt
from simulated_rsrp import SimulatedRSRP
from problem_formulation import CCORasterBlanketFormulation
import json
import glob
import re

def plot_rsrp_map(rsrp_powermap, interference_powermap, x_coords, y_coords, title):
    """Plot RSRP map in dBm"""
    plt.figure(figsize=(10, 8))
    
    # Convert to dBm if in linear scale
    if np.any(rsrp_powermap > 0):
        rsrp_dbm = 10 * np.log10(rsrp_powermap) + 30
    else:
        rsrp_dbm = rsrp_powermap

    # Create heatmap
    plt.imshow(rsrp_dbm, 
              extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
              aspect='equal',
              cmap='jet',
              origin='lower')
    
    plt.colorbar(label='RSRP (dBm)')
    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.grid(True)
    plt.show()

def main():
    # Load configuration
    with open('cco_noop.json') as f:
        config = json.load(f)
    
    # Load power maps
    power_maps_path = config['simulated_rsrp']['path']
    power_range = config['simulated_rsrp']['power_range']
    
    # Initialize RSRP simulator
    simulated_rsrp = SimulatedRSRP.construct_from_npz_files(
        power_maps_path,
        (power_range[0], power_range[1])
    )
    
    # Initialize problem formulation
    problem_formulation = CCORasterBlanketFormulation(**config['problem_formulation']['parameters'])
    
    # Get configuration ranges
    downtilt_range, power_range = simulated_rsrp.get_configuration_range()
    _, num_sectors = simulated_rsrp.get_configuration_shape()
    
    # Create random configuration
    downtilts = np.random.uniform(downtilt_range[0], downtilt_range[1], num_sectors)
    powers = np.random.uniform(power_range[0], power_range[1], num_sectors)
    configuration = (downtilts, powers)
    
    # Get RSRP and interference maps
    rsrp_powermap, interference_powermap, _ = simulated_rsrp.get_RSRP_and_interference_powermap(configuration)
    
    # Get x, y coordinates range
    xy_min, xy_max = simulated_rsrp.get_locations_range()
    x_coords = [xy_min.x, xy_max.x]
    y_coords = [xy_min.y, xy_max.y]
    
    # Plot RSRP map
    plot_rsrp_map(rsrp_powermap, interference_powermap, x_coords, y_coords, 
                 f'RSRP Map\nDowntilt range: {downtilt_range}, Power range: {power_range} dBm')

if __name__ == "__main__":
    main()