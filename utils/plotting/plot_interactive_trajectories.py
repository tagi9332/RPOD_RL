import os
import webbrowser
import numpy as np
import plotly.graph_objects as go

def plot_interactive_trajectories(all_runs_data, summary_df, output_folder):
    """Generates an interactive HTML 3D plot of all Monte Carlo trajectories."""
    print("Generating Interactive 3D Plot (Plotly)...")

    fig = go.Figure()

    # 1. Plot the Target (RSO) at the origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='gold', symbol='diamond'),
        name='Target (RSO)',
        legendgroup='target'
    ))

    # 1b. Determine standoff waypoint position(s) from logged run data.
    #     r_waypoint_H is in Hill frame: [radial, in-track, cross-track].
    #     The plot axes swap Hill x/y: plot_x=hill_y, plot_y=hill_x, plot_z=hill_z.
    has_waypoint_data = all(
        all(f"r_waypoint_H_{c}" in df.columns for c in "xyz")
        for df in all_runs_data
    )

    waypoints_plot = []  # (plot_x, plot_y, plot_z) per run
    if has_waypoint_data:
        for run_df in all_runs_data:
            hill_x = run_df["r_waypoint_H_x"].iloc[0]
            hill_y = run_df["r_waypoint_H_y"].iloc[0]
            hill_z = run_df["r_waypoint_H_z"].iloc[0]
            waypoints_plot.append((hill_y, hill_x, hill_z))

        wp_arr = np.array(waypoints_plot)
        spread = np.max(np.linalg.norm(wp_arr - wp_arr[0], axis=1))
        same_waypoint = spread < 1.0  # all runs share the same waypoint

        if same_waypoint:
            px, py, pz = waypoints_plot[0]
            fig.add_trace(go.Scatter3d(
                x=[px], y=[py], z=[pz],
                mode='markers',
                marker=dict(size=8, color='magenta', symbol='diamond-open'),
                name='Standoff Waypoint (30m)',
                legendgroup='waypoint'
            ))
    else:
        same_waypoint = True  # no data — skip per-run waypoints

    success_plotted = False
    fail_plotted = False
    start_plotted = False
    waypoint_legend_added = False

    # 2. Loop through each run and plot the trajectory
    for idx, run_df in enumerate(all_runs_data):
        is_success = summary_df.loc[idx, "success"]

        color = '#00CC96' if is_success else '#EF553B'
        opacity = 0.8 if is_success else 0.4

        if is_success and not success_plotted:
            show_legend = True
            legend_name = "Successful Approach"
            success_plotted = True
        elif not is_success and not fail_plotted:
            show_legend = True
            legend_name = "Failed/Timeout"
            fail_plotted = True
        else:
            show_legend = False
            legend_name = "Successful Approach" if is_success else "Failed/Timeout"

        fig.add_trace(go.Scatter3d(
            x=run_df["hill_y"],
            y=run_df["hill_x"],
            z=run_df["hill_z"],
            mode='lines',
            line=dict(color=color, width=3),
            opacity=opacity,
            name=legend_name,
            legendgroup=legend_name,
            showlegend=show_legend,
            hoverinfo='skip'
        ))

        show_start_legend = False
        if not start_plotted:
            show_start_legend = True
            start_plotted = True

        fig.add_trace(go.Scatter3d(
            x=[run_df["hill_y"].iloc[0]],
            y=[run_df["hill_x"].iloc[0]],
            z=[run_df["hill_z"].iloc[0]],
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.7),
            name="Initial Position",
            legendgroup="initial_pos",
            showlegend=show_start_legend,
            hovertext=f"Run {idx + 1} Start",
            hoverinfo="text"
        ))

        # Per-run waypoint markers when attitude varies across runs
        if has_waypoint_data and not same_waypoint:
            px, py, pz = waypoints_plot[idx]
            fig.add_trace(go.Scatter3d(
                x=[px], y=[py], z=[pz],
                mode='markers',
                marker=dict(size=6, color=color, symbol='diamond-open', opacity=0.8),
                name='Standoff Waypoint (30m)',
                legendgroup='waypoint',
                showlegend=not waypoint_legend_added,
                hovertext=f"Run {idx + 1} Waypoint",
                hoverinfo="text"
            ))
            waypoint_legend_added = True

    # 3. Format the Layout
    fig.update_layout(
        title="Monte Carlo Trajectories (RSO aligned to Velocity)",
        scene=dict(
            xaxis_title='In-Track / Velocity (m)',
            yaxis_title='Radial (m)',
            zaxis_title='Cross-Track (m)',
            xaxis=dict(gridcolor='gray', showbackground=False),
            yaxis=dict(gridcolor='gray', showbackground=False),
            zaxis=dict(gridcolor='gray', showbackground=False),
            aspectmode='data'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
    )

    # 4. Save and automatically open
    plot_path = os.path.join(output_folder, 'interactive_trajectories.html')
    fig.write_html(plot_path)
    print(f"Saved interactive plot to: {plot_path}")
    webbrowser.open('file://' + os.path.realpath(plot_path))
