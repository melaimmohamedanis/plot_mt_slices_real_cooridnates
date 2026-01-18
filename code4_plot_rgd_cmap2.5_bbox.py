import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pyproj import CRS, Transformer
import os
import sys
import glob
import traceback

# --- MANDATORY IMPORTS FROM YOUR FILES ---
from dat_parser import parse_dat_file, parse_edi_file, validate_and_update_stations
from model_parse import convert_rho_to_geo, calculate_geo_coordinates 
from grd_file_parser import parse_and_crop_grd

# --- UPDATED PATHS ---
DAT_FILE_PATH = 'C:/Users/moham/Music/mtest20a/data.dat'
EDI_FILE_PATH = 'C:/edi/not_modified/all_fre/bi100.edi'
RHO_INPUT_FILE_PATH = 'C:/Users/moham/Music/mtest20a/Modular_NLCG_079.rho'
GRD_FOLDER_PATH = r'C:\Users\moham\Music\mtest20a\new_depth' 
OUTPUT_FOLDER = os.path.join(GRD_FOLDER_PATH, 'Output_Plots2.5')
USER_BBOX = [35.8, 36.1, 3.9, 4.3]  # [min_lat, max_lat, min_lon, max_lon]

def get_utm_zone(longitude):
    return int((longitude + 180) / 6) + 1

def plot_and_save_map(stations, bbox, center_pt, grd_path, w_north_str, w_east_str, save_dir):
    """
    Plots a single depth slice with CORRECT North/East alignment and overlays stations.
    
    Assumptions from your error:
      - Original grid: 72 (North) x 92 (East)
      - After cropping: 45 North, 53 East
      - But crop_rho.shape = (53, 45) → (n_east_cropped, n_north_cropped)
    
    Therefore: we transpose to (n_north, n_east) for geographic plotting.
    """
    # 1. Parse and crop the .grd file
    crop_rho, c_north, c_east = parse_and_crop_grd(
        grd_path, 72, 92, w_north_str, w_east_str, bbox, center_pt[0], center_pt[1]
    )
    
    print(f"DEBUG: crop_rho.shape = {crop_rho.shape}, len(c_north) = {len(c_north)}, len(c_east) = {len(c_east)}")
    # From your log: crop_rho=(53,45), cx=45, cy=53 → so cx = North, cy = East

    # 2. Interpret axes:
    #    cx corresponds to North direction (latitude) → length = n_north
    #    cy corresponds to East direction (longitude) → length = n_east
    n_north = len(c_north)
    n_east = len(c_east)

    # 3. Transpose crop_rho from (n_east, n_north) → (n_north, n_east)
    if crop_rho.shape == (n_east, n_north):
        resistivity_grid = crop_rho.T  # Now (n_north, n_east)
        print(f"Transposed crop_rho to {resistivity_grid.shape}")
    else:
        raise ValueError(f"Unexpected shape: expected ({n_east}, {n_north}) or ({n_north}, {n_east}), got {crop_rho.shape}")

    # 4. Compute geographic coordinates
    final_latitudes, _ = calculate_geo_coordinates(center_pt, c_north, [0] * len(c_north))
    _, final_longitudes = calculate_geo_coordinates(center_pt, [0] * len(c_east), c_east)

    # 5. Clip and log10 resistivity
    data_clipped = np.clip(resistivity_grid, 1, 1000)
    data_log10 = np.log10(data_clipped)

    # 6. Color map
    my_colors = [
        '#700060', '#e70010','#ff4000','#ff2000', '#ff6000','#ffdf00',
        '#befe3f', '#00ff70', '#1effe1','#0055ff', '#0000ff'
    ]
    custom_cmap = colors.LinearSegmentedColormap.from_list("my_palette", my_colors)
   
    cmap_discrete = colors.ListedColormap(custom_cmap(np.linspace(0, 1, 31)))
    
    cmap_discrete.set_over('#0000ff')
    norm = colors.Normalize(vmin=0, vmax=2.5)

    # 7. Meshgrid: longitude (cols), latitude (rows)
    LON_G, LAT_G = np.meshgrid(final_longitudes, final_latitudes)

    # --- STATION PLOTTING ---
    # Extract station coordinates
    s_lons = np.array([s[2] for s in stations])  # Longitude
    s_lats = np.array([s[1] for s in stations])  # Latitude

    # Convert to UTM for offset plotting
    utm_zone = get_utm_zone(center_pt[1])
    epsg = 32600 + utm_zone if center_pt[0] >= 0 else 32700 + utm_zone
    to_utm = Transformer.from_crs("epsg:4326", f"epsg:{epsg}", always_xy=True)
    ce, cn = to_utm.transform(center_pt[1], center_pt[0])  # Center point in UTM
    se, sn = to_utm.transform(s_lons, s_lats)  # Stations in UTM

    # --- MAP 1: UTM OFFSET ---
    fig1, ax1 = plt.subplots(figsize=(12, 11))
    im1 = ax1.pcolormesh(c_east, c_north, data_log10, cmap=cmap_discrete, norm=norm, shading='nearest', alpha=0.9)
    ax1.scatter(se - ce, sn - cn, color='white', marker='^', s=150, edgecolors='k', zorder=10)  # Plot stations
    ax1.grid(True, linestyle='--', alpha=0.5, color='grey')
    ax1.set_title(f"UTM Offset Map: {os.path.basename(grd_path).replace('.grd', '')}", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Easting Offset (m)")
    ax1.set_ylabel("Northing Offset (m)")
    ax1.set_aspect('equal')

    # Horizontal Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', shrink=0.7, aspect=30, pad=0.08)
    cbar1.set_ticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    cbar1.set_label('LOG10[Resistivity (Ohm.m)]', fontsize=12, fontweight='bold')
    for label in cbar1.ax.get_xticklabels():
        label.set_fontweight('bold')

    plt.savefig(os.path.join(save_dir, f"{os.path.basename(grd_path).replace('.grd', '')}_UTM.png"), dpi=200, bbox_inches='tight')
    plt.close(fig1)

    # --- MAP 2: LAT/LON ---
    fig2, ax2 = plt.subplots(figsize=(12, 11))
    im2 = ax2.pcolormesh(LON_G, LAT_G, data_log10, cmap=cmap_discrete, norm=norm, shading='nearest', alpha=0.9)
    ax2.scatter(s_lons, s_lats, color='white', marker='^', s=150, edgecolors='k', zorder=10)  # Plot stations

    # Add station labels
    for i, s in enumerate(stations):
        ax2.text(s_lons[i], s_lats[i], s[0], fontsize=10, fontweight='bold', color='black')  # Station names

    ax2.grid(True, linestyle='--', alpha=0.5, color='grey')
    ax2.set_title(f"Geographic Map: {os.path.basename(grd_path).replace('.grd', '')}", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Longitude (Decimal Degrees)")
    ax2.set_ylabel("Latitude (Decimal Degrees)")
    ax2.set_aspect('equal')

    # Horizontal Colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', shrink=0.7, aspect=30, pad=0.08)
    cbar2.set_ticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    cbar2.set_label('LOG10[Resistivity (Ohm.m)]', fontsize=12, fontweight='bold')
    for label in cbar2.ax.get_xticklabels():
        label.set_fontweight('bold')

    plt.savefig(os.path.join(save_dir, f"{os.path.basename(grd_path).replace('.grd', '')}_Geo.png"), dpi=200, bbox_inches='tight')
    plt.close(fig2)

    print(f"✅ Processed: {os.path.basename(grd_path)}")# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        print("--- STEP 1: Parsing Survey Data ---")
        stations, (c_lat, c_lon, c_z), data_bbox = parse_dat_file(DAT_FILE_PATH)
        
        final_bbox = {
            'min_lat': float(USER_BBOX[0]),
            'max_lat': float(USER_BBOX[1]),
            'min_lon': float(USER_BBOX[2]),
            'max_lon': float(USER_BBOX[3])
        }

        updated_stations, model_center_point = validate_and_update_stations(
            parse_edi_file(EDI_FILE_PATH), stations, c_lat, c_lon, c_z
        )

        print("--- STEP 2: Extracting Grid Widths ---")
        with open(RHO_INPUT_FILE_PATH, 'r') as f:
            lines = [line for line in f if not line.strip().startswith('#')]
            # Line 1: header, Line 2: x (North) widths, Line 3: y (East) widths
            w_north_str = lines[1].strip()  # North widths
            w_east_str = lines[2].strip()  # East widths

        print("--- STEP 3: Processing GRD files ---")
        grd_files = glob.glob(os.path.join(GRD_FOLDER_PATH, "*.grd"))
        
        if not grd_files:
            print("No .grd files found!")
        else:
            for grd_file in grd_files:
                plot_and_save_map(
                    updated_stations, final_bbox, model_center_point,
                    grd_file, w_north_str, w_east_str, OUTPUT_FOLDER
                )

        print(f"\n✅ SUCCESS! Plots saved in: {OUTPUT_FOLDER}")

    except Exception as e:
        print(f"\n❌ FATAL ERROR:")
        traceback.print_exc()
        sys.exit(1)