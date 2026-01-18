import io
import os
import pandas as pd
import numpy as np

def parse_dat_file(dat_file_path):
    """
    Parses the .dat file and extracts unique station locations (Code, Latitude, Longitude, Depth).
    
    Parameters:
        dat_file_path (str): Path to the .dat file.
    
    Returns:
        list: A list of tuples containing (station_code, lat, lon, depth).
        tuple: Geographic center (center_lat, center_lon, center_z).
        dict: Bounding box of the stations (min_lat, max_lat, min_lon, max_lon).
    """
    HEADER_NAMES = ['Period(s)', 'Code', 'GG_Lat', 'GG_Lon', 'X', 'Y', 'Z', 'Component', 'Real', 'Imag', 'Error']
    
    if not os.path.exists(dat_file_path):
        raise FileNotFoundError(f"Error: Data file not found at {dat_file_path}")
    
    data_lines = []
    with open(dat_file_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            # Identify data lines: must not start with # or >, and must start with a digit (the period)
            if stripped_line and not stripped_line.startswith(('#', '>')) and stripped_line[0].isdigit():
                data_lines.append(stripped_line)
    
    if not data_lines:
        raise ValueError("No valid data lines found in the file after skipping headers/comments.")
    
    # Prepend the known clean header names so pandas can read the structured data
    final_header = ' '.join(HEADER_NAMES)
    data_stream = final_header + "\n" + "\n".join(data_lines)
    
    # Use io.StringIO to treat the collected data string as a file for pandas
    df = pd.read_csv(io.StringIO(data_stream), sep=r'\s+', header=0)
    
    # Extract unique station locations (Code, GG_Lat, GG_Lon, Z)
    site_locations = df[['Code', 'GG_Lat', 'GG_Lon', 'Z']].drop_duplicates().reset_index(drop=True)
    
    # Convert to list of tuples (station_code, lat, lon, depth)
    station_list = [(row['Code'], row['GG_Lat'], row['GG_Lon'], row['Z']) for _, row in site_locations.iterrows()]
    
    # Calculate geographic center (midpoint of lat, lon, and z)
    center_lat = calculate_bounding_box_center(site_locations['GG_Lat'].to_numpy())
    center_lon = calculate_bounding_box_center(site_locations['GG_Lon'].to_numpy())
    center_z = calculate_bounding_box_center(site_locations['Z'].to_numpy())
    
    # Calculate bounding box (min_lat, max_lat, min_lon, max_lon)
    bounding_box = {
        "min_lat": site_locations['GG_Lat'].min(),
        "max_lat": site_locations['GG_Lat'].max(),
        "min_lon": site_locations['GG_Lon'].min(),
        "max_lon": site_locations['GG_Lon'].max()
    }
    
    return station_list, (center_lat, center_lon, center_z), bounding_box


def parse_edi_file(edi_file_path):
    """
    Parses the .edi file and extracts DATAID, ELEV, and filename.
    
    Parameters:
        edi_file_path (str): Path to the .edi file.
    
    Returns:
        dict: A dictionary containing DATAID, ELEV, and filename.
    """
    data = {
        "DATAID": None,
        "ELEV": None,
        "filename": os.path.basename(edi_file_path)
    }
    
    with open(edi_file_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            
            # Match DATAID
            if stripped_line.startswith("DATAID="):
                data["DATAID"] = stripped_line.split("=")[1].strip('"')
            
            # Match ELEV
            elif stripped_line.startswith("ELEV="):
                try:
                    data["ELEV"] = float(stripped_line.split("=")[1])
                except ValueError:
                    raise ValueError(f"Invalid ELEV value in file: {edi_file_path}")
            
            # Stop parsing after the HEADER section
            if stripped_line.startswith(">INFO"):
                break
    
    # Validate required fields
    if data["DATAID"] is None or data["ELEV"] is None:
        raise ValueError(f"Missing DATAID or ELEV in file: {edi_file_path}")
    
    return data


def validate_and_update_stations(edi_data, stations, center_lat, center_lon, center_z):
    """
    Validates the DATAID against the station codes and updates altitude information.
    
    Parameters:
        edi_data (dict): Parsed data from the .edi file (DATAID, ELEV, filename).
        stations (list): List of tuples containing (station_code, lat, lon, depth).
        center_lat (float): Latitude center of the stations.
        center_lon (float): Longitude center of the stations.
        center_z (float): Depth center of the stations.
    
    Returns:
        list: Updated list of stations with altitude values.
        tuple: Model center point (latitude_center, longitude_center, altitude_center).
    """
    # Check if DATAID or filename matches any station code
    matching_station = None
    for station in stations:
        if station[0] == edi_data["DATAID"] or station[0] == os.path.splitext(edi_data["filename"])[0]:
            matching_station = station
            break
    
    if not matching_station:
        raise ValueError(f"No station code found for DATAID: {edi_data['DATAID']} or filename: {edi_data['filename']}")
    
    # Extract matching station details
    station_code, lat, lon, depth = matching_station
    
    # Calculate altitude_center and update altitude
    altitude_center = edi_data["ELEV"] + depth
    updated_stations = []
    for station in stations:
        station_code, lat, lon, depth = station
        altitude = altitude_center - depth
        updated_stations.append((station_code, lat, lon, altitude))
    
    # Update model center point
    model_center_point = (center_lat, center_lon, altitude_center)
    
    return updated_stations, model_center_point


def calculate_bounding_box_center(coords):
    """
    Calculates the center point using the Bounding Box Midpoint method.
    
    Parameters:
        coords (numpy.ndarray): Array of coordinates (e.g., latitudes, longitudes, or depths).
    
    Returns:
        float: The midpoint of the coordinates.
    """
    if coords.size == 0:
        return np.nan
    return (coords.max() + coords.min()) / 2.0