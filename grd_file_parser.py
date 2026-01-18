import numpy as np
from pyproj import Transformer
import sys

def parse_and_crop_grd(grd_path, n_north, n_east, north_widths_str, east_widths_str, bbox, center_lat, center_lon, margin_percent=0.1):
    print(f"--- ENTERED GRD PARSER ---")
    sys.stdout.flush() 

    # 1. Coordinate setup
    w_north = np.array([float(val) for val in north_widths_str.split()])
    w_east = np.array([float(val) for val in east_widths_str.split()])
    
    def adjust_arrays(north, east):
        c_north, c_east = len(north) // 2, len(east) // 2
        adj_north = np.concatenate([-np.cumsum(north[:c_north][::-1])[::-1], np.cumsum(north[c_north:])])
        adj_east = np.concatenate([-np.cumsum(east[:c_east][::-1])[::-1], np.cumsum(east[c_east:])])
        return adj_north, adj_east
    
    adj_north, adj_east = adjust_arrays(w_north, w_east)

    # 2. UTM Bounding Box
    utm_zone = int((center_lon + 180) / 6) + 1
    epsg = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
    to_utm = Transformer.from_crs("epsg:4326", f"epsg:{epsg}", always_xy=True)
    ce, cn = to_utm.transform(center_lon, center_lat)
    min_e, min_n = to_utm.transform(bbox['min_lon'], bbox['min_lat'])
    max_e, max_n = to_utm.transform(bbox['max_lon'], bbox['max_lat'])
    
    t_north_min, t_north_max = (min_n - cn), (max_n - cn)
    t_east_min, t_east_max = (min_e - ce), (max_e - ce)
    
    m_north, m_east = (t_north_max-t_north_min)*margin_percent, (t_east_max-t_east_min)*margin_percent
    id_north = np.where((adj_north >= t_north_min - m_north) & (adj_north <= t_north_max + m_north))[0]
    id_east = np.where((adj_east >= t_east_min - m_east) & (adj_east <= t_east_max + m_east))[0]

    # 3. ROBUST FILE READING
    with open(grd_path, 'r') as f:
        lines = f.readlines()
    
    # We ignore the first 5 lines (Headers + Min/Max line)
    # Data is joined and converted to a float array
    data_content = " ".join(lines[5:])
    raw_data = np.fromstring(data_content, sep=' ')
    #apply exp to data resistivity
    raw_data = np.exp(raw_data) 
    print('min grd resi',min(raw_data));
    print('max grd resi',max(raw_data));

    print(f"DEBUG: Found {raw_data.size} values. Expected {n_north*n_east} ({n_north}x{n_east})")
    sys.stdout.flush()

    # FORCE THE SIZE - This kills the 6626 error
    if raw_data.size >= n_north * n_east:
        print(f"Resistivity First: {raw_data[0]} | Last: {raw_data[n_north*n_east-1]}")
        raw_data = raw_data[:n_north*n_east]
    else:
        raise ValueError(f"File too small! Found {raw_data.size}, need {n_north*n_east}")

    # 4. Reshape
    # Surfer/SAA format is usually (ny, nx)
    full_grid = raw_data.reshape((n_east, n_north))
    
    # Optional Log check
    if raw_data.max() < 15:
        full_grid = np.exp(full_grid)

    cropped_grid = full_grid[id_east[0]:id_east[-1]+1, id_north[0]:id_north[-1]+1]
    
    return cropped_grid, adj_north[id_north], adj_east[id_east]