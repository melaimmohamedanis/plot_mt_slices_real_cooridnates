import numpy as np
import math
from pyproj import Proj, transform, CRS, Transformer
import os

# --- UTILITY FUNCTION FOR UTM ZONE (New) ---
def get_utm_zone(longitude):
    """Calculates the UTM zone number for a given longitude."""
    # UTM zones are 6 degrees wide, starting from zone 1 at -180 degrees.
    # Zone = floor((longitude + 180) / 6) + 1
    return int(math.floor((longitude + 180) / 6)) + 1


def parse_rho_file(rho_file_path):
    # ... (function body remains the same)
    # The parsing logic for the .rho file is already correct.
    # I will omit the body here for brevity, but it is unchanged.
    """
    Parse the .rho file and extract the grid dimensions and resistivity values.
    
    Returns:
        tuple: Grid dimensions (nx, ny, nz), grid spacings (x, y, z), and resistivity values.
    """
    with open(rho_file_path, 'r') as f:
        lines = f.readlines()

    non_comment_lines = [line for line in lines if not line.strip().startswith('#')]

    header = non_comment_lines[0].strip().split()
    n_north, n_east, nz = int(header[0]), int(header[1]), int(header[2])
    # log_scale = header[4] == "LOGE" # Not needed for coordinate conversion

    north_spacing = list(map(float, non_comment_lines[1].strip().split()))
    east_spacing = list(map(float, non_comment_lines[2].strip().split()))
    z_spacing = list(map(float, non_comment_lines[3].strip().split()))

    resistivity_values = []
    for line in non_comment_lines[5:]:
        resistivity_values.extend(map(float, line.strip().split()))

    return (n_north, n_east, nz), (north_spacing, east_spacing, z_spacing), resistivity_values


# --- REPLACED/MODIFIED FUNCTION (Core Fix) ---
# --- REPLACED/MODIFIED FUNCTION (Core Fix) ---
def calculate_geo_coordinates(center_point, north_spacing_offsets, east_spacing_offsets):
    """
    Uses pyproj to accurately convert Cartesian offsets (dx, dy) from the center point
    into geographic coordinates (Latitude, Longitude).
    
    The inputs x_spacing_offsets and y_spacing_offsets are lists of variable cell
    center/interface positions (in meters) relative to the center_point.
    
    Parameters:
        center_point (tuple): Center point (latitude, longitude, altitude).
        
        # CLARIFICATION HERE: These are NOT fixed spacings (dx, dy), but variable
        # offsets in meters from the center point for the x and y grid lines.
        x_spacing_offsets (list): List of variable X offsets (East/West) in meters.
        y_spacing_offsets (list): List of variable Y offsets (North/South) in meters.
        
    Returns:
        tuple: Lists of final latitude and longitude grid points.
    """
    center_lat, center_lon, _ = center_point
    
    # 1. Define the UTM Projection based on the center point
    utm_zone = get_utm_zone(center_lon)
    # Determine hemisphere: N=Northern (326xx), S=Southern (327xx)
    utm_epsg = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
    
    # Define source (WGS84 Lat/Lon) and target (UTM meters) Coordinate Reference Systems (CRS)
    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm = CRS.from_epsg(utm_epsg)
    
    # 2. Get the Cartesian (UTM) coordinates of the center point
    # Transformer for WGS84 -> UTM
    transformer_to_utm = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    center_east_utm, center_north_utm = transformer_to_utm.transform(center_lon, center_lat)
    
    print(f"Using UTM Zone {utm_zone}. Center UTM: X={center_east_utm:.2f}m, Y={center_north_utm:.2f}m")

    # 3. Calculate absolute UTM coordinates for every grid point
    
    # Calculate the absolute X and Y UTM coordinates
    # Add the variable offsets (x_spacing_offsets) to the center UTM position
    absolute_east_utm = [center_east_utm + dx_offset for dx_offset in east_spacing_offsets]
    absolute_north_utm = [center_north_utm + dy_offset for dy_offset in north_spacing_offsets]

    # 4. Transform all absolute UTM coordinates back to Lat/Lon
    # Transformer for UTM -> WGS84
    transformer_to_wgs84 = Transformer.from_crs(crs_utm, crs_wgs84, always_xy=True)
    
    # pyproj can transform lists of coordinates simultaneously
    final_longitudes, final_latitudes = transformer_to_wgs84.transform(
        absolute_east_utm, absolute_north_utm
    )
    
    return list(final_latitudes), list(final_longitudes)

def convert_rho_to_geo(rho_file_path, center_point, output_file):
    """
    Convert the .rho file grid to geographic coordinates with bounding and depth conversion.
    """
    # Parse the .rho file
    (n_north, n_east, nz), (x_spacing, y_spacing, z_spacing), resistivity_values = parse_rho_file(rho_file_path)

    # Convert dx and dy to degrees using pyproj (THE NEW WAY)
    latitudes, longitudes = calculate_geo_coordinates(center_point, x_spacing, y_spacing)

    # Check that the number of coordinates matches the header
    if len(latitudes) != n_north:
        print(f"Warning: Latitude count ({len(latitudes)}) does not match N_north ({n_north}).")
    if len(longitudes) != n_east:
        print(f"Warning: Longitude count ({len(longitudes)}) does not match N_east ({n_east}).")

    # Prepare the output file
    with open(output_file, 'w') as f:
        # Write the header comment
        f.write("# 3D MT model written by ModEM in WS format\n")

        # Write the grid dimensions and LOGE flag
        f.write(f"{n_north} {n_east} {nz} LOGE\n")

        # Write latitude values (NX values)
        # Note: The original WS file format puts Latitudes (NX) on line 3 and Longitudes (NY) on line 4.
        # We must align the calculated coordinates with the file structure.
        f.write(" ".join(f"{lat:.6f}" for lat in latitudes) + "\n")

        # Write longitude values (NY values)
        f.write(" ".join(f"{lon:.6f}" for lon in longitudes) + "\n")

        # Write depth values
        f.write(" ".join(f"{z:.2f}" for z in z_spacing) + "\n")

        # Write resistivity values
        f.write(" ".join(f"{resistivity:.6e}" for resistivity in resistivity_values) + "\n")


# Example usage
if __name__ == "__main__":
    # Example input file content (you must replace this with an actual .rho file)
    # This example is illustrative of the format expected by parse_rho_file
    """
    # Grid File Header
    72 92 62 LOGE
    -1000 -900 -800 ... (72 x_spacing values in meters)
    -1500 -1400 -1300 ... (92 y_spacing values in meters)
    10.00 11.50 13.22 ... (62 z_spacing values in meters)
    3.914390e+01 3.914390e+01 ... (resistivity values)
    """
    
    # Example usage (Replace with your actual file and center point)
    print("--- ModEM .rho to Geographic WS File Converter (using pyproj) ---")
    
    try:
        rho_file_path = input("Enter the path to the .rho file: ").strip()
        
        # Example: Center near Los Angeles, UTM Zone 11N (approx)
        center_input = input("Enter the center point (latitude, longitude, altitude, separated by commas, e.g., 34.0,-118.0,0.0): ").strip().split(',')
        center_point = tuple(map(float, center_input))
        
        output_file = input("Enter the output file path (.ws or .rho_geo): ").strip()
        
        # Convert the .rho file to geographic grid
        convert_rho_to_geo(rho_file_path, center_point, output_file)
        print(f"\nSUCCESS! New geographic file saved to: {output_file}")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please check file path, input format, and ensure 'pyproj' is installed.")