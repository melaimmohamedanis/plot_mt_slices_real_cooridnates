import numpy as np
import pandas as pd
import math
from pyproj import CRS, Transformer
import os
import sys
import traceback

# ====================================================================
# A. UTILITY FUNCTIONS
# ====================================================================

def get_utm_zone(longitude):
    """Calculates the UTM zone number for a given longitude."""
    return int(math.floor((longitude + 180) / 6)) + 1

def parse_rho_file_with_pandas(rho_file_path):
    """
    Parse the ModEM .rho file and extract grid dimensions and cell widths.
    """
    if not os.path.exists(rho_file_path):
        raise FileNotFoundError(f"Error: .rho file not found at {rho_file_path}")
        
    with open(rho_file_path, 'r') as f:
        lines = f.readlines()

    non_comment_lines = []
    for line in lines[1:]:
        stripped_line = line.strip()
        if stripped_line.startswith('#') or not stripped_line:
            continue
        non_comment_lines.append(stripped_line)

    header = non_comment_lines[0].split()
    n_north, n_east, nz = int(header[0]), int(header[1]), int(header[2])

    # ModEM .rho: Lines 2, 3, 4 are widths in X, Y, Z
    north_widths = list(map(float, non_comment_lines[1].split()))
    east_widths = list(map(float, non_comment_lines[2].split()))
    z_widths = list(map(float, non_comment_lines[3].split()))

    resistivity_start_index = 4
    resistivity_blocks = []
    current_block = []
    
    # Collect all data values
    for line in non_comment_lines[resistivity_start_index:]:
        current_block.extend(map(float, line.split()))

    # Split data into NZ blocks of size NX*NY
    block_size = n_north * n_east
    for k in range(nz):
        start = k * block_size
        end = start + block_size
        slice_data = np.array(current_block[start:end]).reshape((n_east, n_north))
        slice_data_north_flipped = slice_data[:, ::-1]  # column 0 = north
        resistivity_blocks.append(pd.DataFrame(slice_data_north_flipped))

    print(f"✅ RHO Parse Success: {n_north}x{n_east}x{nz}")
    return (n_north, n_east, nz), (north_widths, east_widths, z_widths), resistivity_blocks

def replace_invalid_resistivity_values(resistivity_values):
    """Replaces <=0 values with the last valid value or a tiny positive floor."""
    resistivity_values = np.array(resistivity_values, dtype=np.float64)
    valid_values = []
    for i, value in enumerate(resistivity_values):
        if value <= 0:
            resistivity_values[i] = valid_values[-1] if valid_values else 1e-6
        else:
            valid_values.append(value)
    return resistivity_values

def convert_xy_to_latlon_axes(center_lat, center_lon, north_widths, east_widths):
    """Converts local grid meters to WGS84 Lat/Lon."""
    utm_zone = get_utm_zone(center_lon)
    utm_epsg = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
    transformer_to_utm = Transformer.from_crs("epsg:4326", f"epsg:{utm_epsg}", always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(f"epsg:{utm_epsg}", "epsg:4326", always_xy=True)
    
    center_east_utm, center_north_utm = transformer_to_utm.transform(center_lon, center_lat)
    
    def get_cell_centers(widths, center_coord):
        N = len(widths)
        N_half = N // 2
        edges = np.zeros(N + 1)
        # Reconstruct edges relative to center point
        left_side = widths[:N_half][::-1]
        edges[:N_half+1] = (center_coord - np.cumsum(np.insert(left_side, 0, 0)))[::-1]
        right_side = widths[N_half:]
        edges[N_half:] = center_coord + np.cumsum(np.insert(right_side, 0, 0))
        return (edges[:-1] + edges[1:]) / 2

    ax_east = get_cell_centers(east_widths, center_east_utm)
    ax_north = get_cell_centers(north_widths, center_north_utm)
    
    _, unique_north_lats = transformer_to_wgs84.transform(ax_north, np.full(len(ax_north), center_north_utm))
    unique_east_lons, _ = transformer_to_wgs84.transform(np.full(len(ax_east), center_east_utm), ax_east)
    
    return unique_north_lats, unique_east_lons

def generate_surfer_ascii_grd_file(output_dir, z_km_str, lat_axis, lon_axis, resistivity_df):
    """Writes the .grd file in Surfer GS ASCII format."""
    N_NORTH, N_EAST= lat_axis.size, lon_axis.size
    # Flip E-W for Surfer coordinate system
    resistivity_array = resistivity_df.values 
 
    output_lines = [
        "DSAA",
        f"{N_EAST} {N_NORTH}",
        f"{np.min(lon_axis):.6f} {np.max(lon_axis):.6f}",
        f"{np.min(lat_axis):.6f} {np.max(lat_axis):.6f}",
        f"{np.min(resistivity_array):.4e} {np.max(resistivity_array):.4e}"
    ]
    
    for row in resistivity_array:
        output_lines.append(" ".join([f"{val:.4e}" for val in row]))
    
    filename = f"depth_{z_km_str}km.grd"
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, 'w') as f:
        f.write("\n".join(output_lines))
    return file_path

# ====================================================================
# B. MAIN EXECUTION BLOCK
# ====================================================================

if __name__ == "__main__":
    # USER PATHS
    RHO_FILE_PATH = r'C:\Users\moham\Music\mtest20a\Modular_NLCG_079.rho'
    CENTER_LAT = 35.947777
    CENTER_LON = 4.102361
    H_ELEV_Z_COORD = 1149.00  # Reference elevation in meters
    
    # FOLDER CREATION LOGIC
    # This gets the folder 'C:\Users\moham\Music\mtest20a'
    BASE_DIR = os.path.dirname(RHO_FILE_PATH)
    # This creates 'C:\Users\moham\Music\mtest20a\new_depth'
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'new_depth')
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 Created output directory: {OUTPUT_FOLDER}")

    try:
        # Step 1: Parse
        (N_NORTH, N_EAST, NZ), (north_w, east_w, z_widths), resistivity_blocks = parse_rho_file_with_pandas(RHO_FILE_PATH)

        # Step 2: Calculate Vertical Medians (Midpoints)
        # z_edges accumulates the widths: [0, 10, 25, ...]
        z_edges = np.cumsum(np.insert(z_widths, 0, 0))
        # z_midpoints finds the center of each cell
        z_midpoints_m = (z_edges[:-1] + z_edges[1:]) / 2 

        # Step 3: Shift relative to elevation and convert to KM
        # (Midpoint - 1149) / 1000
        z_relative_km = (z_midpoints_m - H_ELEV_Z_COORD) / 1000.0

        # Step 4: Geographic Conversion
        lat_axis, lon_axis = convert_xy_to_latlon_axes(CENTER_LAT, CENTER_LON, north_w, east_w)
        
        print(f"\nProcessing {NZ} slices...")

        # Step 5: Generate Files
        for k in range(NZ):
            z_str = f"{z_relative_km[k]:.4f}"
            
            res_slice = resistivity_blocks[k]
            # Flatten, clean, and reshape
            res_clean = replace_invalid_resistivity_values(res_slice.values.flatten())
            res_df = pd.DataFrame(res_clean.reshape((N_EAST, N_NORTH)))
            
            file_path = generate_surfer_ascii_grd_file(
                OUTPUT_FOLDER, z_str, lat_axis, lon_axis, res_df
            )
            
            if k % 10 == 0 or k == NZ - 1:
                print(f"   > Saved: {os.path.basename(file_path)}")

        print(f"\n✅ SUCCESS! All files saved in: {OUTPUT_FOLDER}")
        
    except Exception as e:
        print("\n❌ FATAL ERROR:")
        traceback.print_exc()