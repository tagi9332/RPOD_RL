import os
import webbrowser
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

    success_plotted = False
    fail_plotted = False
    start_plotted = False # <--- Add this flag

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

        # The trajectory line
        fig.add_trace(go.Scatter3d(
            x=run_df["hill_x"], 
            y=run_df["hill_y"], 
            z=run_df["hill_z"],
            mode='lines',
            line=dict(color=color, width=3),
            opacity=opacity,
            name=legend_name,
            legendgroup=legend_name,
            showlegend=show_legend,
            hoverinfo='skip' 
        ))

        # --- Updated Start Position Logic ---
        # Only show legend for the very first blue dot
        show_start_legend = False
        if not start_plotted:
            show_start_legend = True
            start_plotted = True

        fig.add_trace(go.Scatter3d(
            x=[run_df["hill_x"].iloc[0]], 
            y=[run_df["hill_y"].iloc[0]], 
            z=[run_df["hill_z"].iloc[0]],
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.7),
            name="Initial Position", # <--- Cleaned up name
            legendgroup="initial_pos", # <--- Separate group for clarity
            showlegend=show_start_legend,
            hovertext=f"Run {idx + 1} Start",
            hoverinfo="text"
        ))

    # 3. Format the Layout
    fig.update_layout(
        title="Interactive Monte Carlo Trajectories",
        scene=dict(
            xaxis_title='Relative X (m)',
            yaxis_title='Relative Y (m)',
            zaxis_title='Relative Z (m)',
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