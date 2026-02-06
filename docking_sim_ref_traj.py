import numpy as np
import pandas as pd
from matplotlib import animation, pyplot as plt
import gymnasium as gym
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

# BSK-RL Imports
from bsk_rl import ConstellationTasking, scene, data
from bsk_rl.utils.orbital import cd2hill 
from Basilisk.architecture import bskLogging

# Import definitions from train.py
from docking_sim_training import RSOSat, InspectorSat, sat_arg_randomizer

# Silence logs
bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

# --- HELPER: MRP to Rotation Matrix ---
def MRP2C(sigma):
    """Converts Modified Rodrigues Parameters to a Direction Cosine Matrix (Rotation Matrix)."""
    sigma = np.array(sigma)
    s_sq = np.dot(sigma, sigma)
    if s_sq > 1.0: # Check for shadow set switching if needed, though BSK handles this usually
        pass 
        
    skew_s = np.array([
        [0, -sigma[2], sigma[1]],
        [sigma[2], 0, -sigma[0]],
        [-sigma[1], sigma[0], 0]
    ])
    
    # MRP Rotation Matrix Formula
    return np.eye(3) + (8 * np.dot(skew_s, skew_s) - 4 * (1 - s_sq) * skew_s) / ((1 + s_sq)**2)

# --- 1. Custom Wrapper (Updated for Attitude Data) ---
class SB3CompatibleEnv(gym.Env):
    def __init__(self, env, agent_name="Inspector"):
        self.env = env
        self.agent_name = agent_name
        self.observation_space = env.observation_space(agent_name)
        self.action_space = env.action_space(agent_name)
        
        self.sim_rate = getattr(env, 'sim_rate', 5.0) 
        self.current_sim_time = 0.0

    def reset(self, **kwargs):
        self.current_sim_time = 0.0 
        obs_dict, info = self.env.reset(**kwargs)
        return obs_dict[self.agent_name], info

    def step(self, action):
        obs_dict, reward_dict, terminated_dict, truncated_dict, info = self.env.step({self.agent_name: action})
        
        self.current_sim_time += self.sim_rate

        # --- DATA EXTRACTION ---
        rso = self.env.satellites[0]
        inspector = self.env.satellites[1]
        
        # Orbital States
        rso_r_N = np.array(rso.dynamics.r_BN_N)
        rso_v_N = np.array(rso.dynamics.v_BN_N)
        insp_r_N = np.array(inspector.dynamics.r_BN_N)
        insp_v_N = np.array(inspector.dynamics.v_BN_N)
        
        # Attitude State (Required for Plot 6)
        sigma_BN = np.array(inspector.dynamics.sigma_BN)

        # Hill Frame Conversion
        hill_state = cd2hill(rso_r_N, rso_v_N, insp_r_N, insp_v_N)
        
        info = {
            "metrics": {
                "sim_time": self.current_sim_time,
                "r_DC_Hc": hill_state[0],
                "v_DC_Hc": hill_state[1],
                "sigma_BN": sigma_BN,  # <--- NEW: Capture Attitude
                "reward": reward_dict[self.agent_name],
                "dV_remaining": getattr(inspector.fsw, 'dv_available', 0.0)
            }
        }
        
        return obs_dict[self.agent_name], reward_dict[self.agent_name], terminated_dict[self.agent_name], truncated_dict[self.agent_name], info

# --- 2. Inference Loop ---
def run_inference():
    # Setup Environment
    rso_args = dict(conjunction_radius=2.0, batteryStorageCapacity=1e12, u_max=0.01)
    insp_args = dict(
    imageAttErrorRequirement=1.0,
    imageRateErrorRequirement=None,
    instrumentBaudRate=1,
    dataStorageCapacity=1e6,
    batteryStorageCapacity=1e12,
    storedCharge_Init=1e12,
    conjunction_radius=2.0,
    dv_available_init=50.0,
    max_range_radius=10000,
    chief_name="RSO",
    u_max=0.01
)
    
    scenario = scene.SphericalRSO(n_points=100, radius=1.0, theta_max=np.radians(30), range_max=250)
    rewarders = (
        data.RSOInspectionReward(completion_bonus=1.0, completion_threshold=0.90),
        data.ResourceReward(resource_fn=lambda s: s.fsw.dv_available if hasattr(s.fsw, 'dv_available') else 0.0, reward_weight=0.001),
        data.RelativeRangeLogReward(alpha=-1, delta_x_max=np.array([1000, 1000, 1000, 1, 1, 1])),
    )

    env = ConstellationTasking(
        satellites=[RSOSat("RSO", sat_args=rso_args), InspectorSat("Inspector", sat_args=insp_args)],
        sat_arg_randomizer=sat_arg_randomizer,
        scenario=scenario,
        rewarder=rewarders,
        time_limit=60000,
        sim_rate=0.50,
    )

    env_sb3 = SB3CompatibleEnv(env)
    env_sb3 = FlattenObservation(env_sb3)
    
    print("Loading Model...")
    try:
        model = PPO.load("ppo_inspector_v1")
    except FileNotFoundError:
        print("Model not found. Ensure ppo_inspector_v1.zip exists.")
        return None

    print("Generating Trajectory...")
    obs, _ = env_sb3.reset()
    done = False
    reference_data_log = [] 

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_sb3.step(action)
        
        if "metrics" in info:
            reference_data_log.append(info["metrics"])
            
        done = terminated or truncated

    # Process Data
    df = pd.DataFrame(reference_data_log)
    
    # Unpack Vectors
    df['hill_x'] = df['r_DC_Hc'].apply(lambda v: v[0])
    df['hill_y'] = df['r_DC_Hc'].apply(lambda v: v[1])
    df['hill_z'] = df['r_DC_Hc'].apply(lambda v: v[2])
    
    # Unpack Attitude (Sigma)
    df['sigma_1'] = df['sigma_BN'].apply(lambda v: v[0])
    df['sigma_2'] = df['sigma_BN'].apply(lambda v: v[1])
    df['sigma_3'] = df['sigma_BN'].apply(lambda v: v[2])

    df.to_csv("reference_sim_data.csv", index=False)
    print(f"Simulation Complete. {len(df)} steps collected.")
    return df

# --- 3. Plotting (Matches Image) ---
def plot_results(df):
    if df is None or df.empty: return

    # Conversions
    df['time_min'] = df['sim_time'] / 60.0
    df['range_mag'] = df['r_DC_Hc'].apply(np.linalg.norm)
    df['vel_mag'] = df['v_DC_Hc'].apply(np.linalg.norm)

    # Create 2x3 Grid
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Inspector Agent Analysis (Steps: {len(df)})", fontsize=16)

    # --- Plot 1: In-Plane Motion (Hill Frame) ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(df['hill_y'], df['hill_x'], label='Trajectory', color='blue')
    ax1.scatter(0, 0, color='red', marker='*', s=150, label='Target (RSO)', zorder=5)
    ax1.scatter(df['hill_y'].iloc[0], df['hill_x'].iloc[0], color='green', s=100, label='X0 (Inspector)', zorder=5)
    ax1.set_xlabel("Along-Track [m] (y)")
    ax1.set_ylabel("Radial [m] (x)")
    ax1.set_title("In-Plane Motion (Hill Frame)")
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Plot 2: 3D Relative Trajectory ---
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot(df['hill_x'], df['hill_y'], df['hill_z'], label='Trajectory')
    ax2.scatter(0, 0, 0, color='red', marker='*', s=150)
    ax2.scatter(df['hill_x'].iloc[0], df['hill_y'].iloc[0], df['hill_z'].iloc[0], color='green', s=100) # type: ignore
    ax2.set_title("3D Relative Trajectory")
    ax2.set_xlabel("Radial (x)")
    ax2.set_ylabel("Along-Track (y)")
    ax2.set_zlabel("Cross-Track (z)") #type: ignore

    # --- Plot 3: Approach Metrics (Range/Vel) ---
    ax3 = fig.add_subplot(2, 3, 3)
    ln1 = ax3.plot(df['time_min'], df['range_mag'], label='Range [m]', color='steelblue')
    ax3.set_ylabel("Range [m]", color='steelblue')
    ax3.tick_params(axis='y', labelcolor='steelblue')
    
    ax3_twin = ax3.twinx()
    ln2 = ax3_twin.plot(df['time_min'], df['vel_mag'], label='Velocity [m/s]', color='darkorange', linestyle='--')
    ax3_twin.set_ylabel("Velocity [m/s]", color='darkorange')
    ax3_twin.tick_params(axis='y', labelcolor='darkorange')
    
    ax3.set_xlabel("Time [min]")
    ax3.set_title("Approach Metrics")
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Fuel Remaining ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(df['time_min'], df['dV_remaining'], color='forestgreen')
    ax4.set_xlabel("Time [min]")
    ax4.set_ylabel("Delta-V [m/s]")
    ax4.set_title("Fuel Remaining")
    ax4.grid(True)

    # --- Plot 5: Reward Function ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(df['time_min'], df['reward'], color='purple', alpha=0.6, label='Step Reward')
    ax5.set_xlabel("Time [min]")
    ax5.set_ylabel("Reward")
    ax5.set_title("Reward Function")
    ax5.legend(loc='lower right')
    ax5.grid(True)

    # --- Plot 6: Inspector Attitude (MRPs) ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(df['time_min'], df['sigma_1'], label='Sigma 1', color='red')
    ax6.plot(df['time_min'], df['sigma_2'], label='Sigma 2', color='forestgreen')
    ax6.plot(df['time_min'], df['sigma_3'], label='Sigma 3', color='blue')
    ax6.set_xlabel("Time [min]")
    ax6.set_ylabel("$\sigma$")
    ax6.set_title("Inspector Attitude (MRPs)")
    ax6.grid(True)
    ax6.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("inspector_agent_analysis.png")

    # --- 3. Animation Function ---
def animate_results(df):
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
    ax.set_title("Inspector Agent Maneuver with Boresight")
    ax.legend()

    def update(frame):
        # Get data for current frame
        row = df_anim.iloc[frame]
        pos = np.array([row['hill_x'], row['hill_y'], row['hill_z']])
        sigma = row['sigma_BN']
        
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
    output_file = 'inspector_maneuver_with_cone.gif'
    print(f"Saving animation to {output_file}...")
    try:
        ani.save(output_file, writer='pillow', fps=20)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Could not save GIF (missing ImageMagick/Pillow?): {e}")
        plt.show()
    

if __name__ == "__main__":
    df = run_inference()
    plot_results(df)
    animate_results(df)
