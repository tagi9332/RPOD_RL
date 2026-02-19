import os
import numpy as np
from matplotlib import animation, pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.frame_conversions.MRP_conversions import MRP2C



def animate_results(df, output_folder="results"):
    if df is None or df.empty: return
    print("Generating Animation...")

    # Data subsampling (Plot every Nth frame to make animation smoother/faster)
    step = 1 
    df_anim = df.iloc[::step].reset_index(drop=True)
    n_frames = len(df_anim)

    # Setup Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale calculation for axis limits and objects
    max_range = df[['hill_x', 'hill_y', 'hill_z']].abs().max().max()
    limit = max_range * 1.1
    scale_factor = max_range * 0.05 # Base scale for objects

    # --- Static Objects ---
    # RSO (Target) at Origin
    ax.scatter([0], [0], [0], color='red', marker='*', s=200, label='RSO (Target)')

    # --- Dynamic Objects ---
    # 1. Trajectory Line
    line, = ax.plot([], [], [], color='blue', alpha=0.5, linewidth=1, label='Trajectory')
    
    # 2. Inspector Box
    box_scale = scale_factor
    v_box = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * box_scale
    
    faces_box = [[v_box[j] for j in [0, 1, 2, 3]], [v_box[j] for j in [4, 5, 6, 7]], 
                 [v_box[j] for j in [0, 3, 7, 4]], [v_box[j] for j in [1, 2, 6, 5]], 
                 [v_box[j] for j in [0, 1, 5, 4]], [v_box[j] for j in [2, 3, 7, 6]]]
    
    inspector_box = Poly3DCollection(faces_box, facecolors='green', edgecolors='black', alpha=0.8)
    ax.add_collection3d(inspector_box)

    # 3. Boresight Cone (NEW)
    # Define cone parameters in body frame (pointing along +x axis)
    cone_len = scale_factor * 5.0 # Make it significantly longer than the box
    cone_angle_deg = 15
    cone_radius = cone_len * np.tan(np.radians(cone_angle_deg))
    
    # Generate base circle points in y-z plane
    theta = np.linspace(0, 2*np.pi, 20)
    y_base = cone_radius * np.cos(theta)
    z_base = cone_radius * np.sin(theta)
    x_base = np.full_like(y_base, cone_len)
    
    # Vertices: Apex at origin + base circle points
    v_cone_base = np.vstack((x_base, y_base, z_base)).T
    v_cone_apex = np.array([[0.0, 0.0, 0.0]])
    v_cone = np.vstack((v_cone_apex, v_cone_base))

    # Faces: Triangles connecting apex to each base segment
    faces_cone = []
    for i in range(len(theta) - 1):
        faces_cone.append([v_cone[0], v_cone[i+1], v_cone[i+2]])
    faces_cone.append([v_cone[0], v_cone[-1], v_cone[1]]) # Close the loop
    
    # Create cone collection
    boresight_cone = Poly3DCollection(faces_cone, facecolors='cyan', edgecolors='none', alpha=0.4)
    ax.add_collection3d(boresight_cone)

    # Axis Labels & View
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel('Radial (x)')
    ax.set_ylabel('Along-Track (y)')
    ax.set_zlabel('Cross-Track (z)')
    ax.set_title("Inspector Agent Reference Trajectory")
    ax.legend()

    def update(frame):
        # Get data for current frame
        row = df_anim.iloc[frame]
        pos = np.array([row['hill_x'], row['hill_y'], row['hill_z']])
        sigma = np.array([row['sigma_1'], row['sigma_2'], row['sigma_3']])
        
        # Calculate Rotation Matrix (Body to Inertial/Hill)
        R_BN = MRP2C(sigma)
        R_NB = R_BN.T # Transpose for Body -> Inertial rotation

        # --- Update Trajectory Line ---
        current_data = df_anim.iloc[:frame+1]
        line.set_data(current_data['hill_x'], current_data['hill_y'])
        line.set_3d_properties(current_data['hill_z'])

        # --- Update Box ---
        # Rotate and translate vertices
        v_box_rot = (R_NB @ v_box.T).T + pos
        
        new_faces_box = [[v_box_rot[j] for j in [0, 1, 2, 3]], [v_box_rot[j] for j in [4, 5, 6, 7]], 
                         [v_box_rot[j] for j in [0, 3, 7, 4]], [v_box_rot[j] for j in [1, 2, 6, 5]], 
                         [v_box_rot[j] for j in [0, 1, 5, 4]], [v_box_rot[j] for j in [2, 3, 7, 6]]]
        inspector_box.set_verts(new_faces_box)

        # --- Update Cone ---
        # Rotate and translate vertices
        v_cone_rot = (R_NB @ v_cone.T).T + pos
        
        # Reconstruct faces with rotated vertices
        new_faces_cone = []
        for i in range(len(theta) - 1):
            new_faces_cone.append([v_cone_rot[0], v_cone_rot[i+1], v_cone_rot[i+2]])
        new_faces_cone.append([v_cone_rot[0], v_cone_rot[-1], v_cone_rot[1]])
        
        boresight_cone.set_verts(new_faces_cone)
        
        return line, inspector_box, boresight_cone

    #Create Animation
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    
    # Save as GIF
    output_file = os.path.join(output_folder, 'inspector_maneuver_with_cone.gif')
    print(f"Saving animation to {output_file}...")
    try:
        ani.save(output_file, writer='pillow', fps=20)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Could not save GIF (missing ImageMagick/Pillow?): {e}")
        plt.show()