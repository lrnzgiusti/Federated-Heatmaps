"""Privacy Heatmaps Simulation and Evaluation
This script simulates user data, runs the adaptive hierarchical algorithm (Algorithm 2),
and evaluates the results using MSE and L1 distance metrics.
"""



import numpy as np
from config import DEFAULT_SENSITIVITY, Config

from federated_heatmaps.algorithms import AdaptiveHist
from federated_heatmaps.utils import (
    calculate_l1_dist,
    calculate_mse,
    compute_true_heatmap,
    generate_simulated_user_data,
    reconstruct_flat_heatmap_from_tree,
)

# --- Main Simulation Example ---
if __name__ == "__main__":
    # Setup Configuration
    sim_config = Config.from_yaml("config.yaml")

    # Generate Simulated Data (Population)
    total_population_size = 5000 # All users available

    # Example for multi-location
    use_multi_location = False # Set to True to test multi-location extension
    w = sim_config.tree.grid_width
    h = sim_config.tree.grid_height
    if use_multi_location:
        user_population = generate_simulated_user_data(total_population_size, w, h, "multi_uniform")
    else:
        user_population = generate_simulated_user_data(total_population_size, w, h, "clustered")


    # Initialize Algorithm 2
    adaptive_hist_algo = AdaptiveHist(sim_config, user_population)

    # Run the adaptive algorithm
    print("Running Adaptive Hierarchical Algorithm (Algorithm 2)...")
    final_noisy_hist_vector, final_hist_map_idx_to_id, final_query_tree, total_comm = \
        adaptive_hist_algo.run(
            sensitivity_delta=DEFAULT_SENSITIVITY, # Or updated for multi-loc inside
            multi=use_multi_location
        )

    print("\n--- Results ---")
    if final_noisy_hist_vector is not None and final_noisy_hist_vector.size > 0:
        print(f"Final Noisy Histogram Vector (first 10 elements): {final_noisy_hist_vector[:10]}")
        print(f"Map from hist index to node ID (first 5): {{ {', '.join(f'{k}: "{v}"' for k,v in list(final_hist_map_idx_to_id.items())[:5])} }}")
        num_reporting_cells_final = len(final_hist_map_idx_to_id)
        print(f"Number of reporting cells in final histogram: {num_reporting_cells_final}")
    else:
        print("Final noisy histogram is empty or None.")

    print(f"Total Communication Cost (sum of vector sizes): {total_comm}")

    # Evaluation (MSE)
    # 1. Get true heatmap based on the *final tree structure* from the algorithm
    #    (So comparison is fair: same cells, different counts)
    if final_noisy_hist_vector is not None :
        true_density_on_final_tree_cells, _ = compute_true_heatmap(user_population, final_query_tree, is_multi_location=use_multi_location)

        # Normalize noisy histogram to density
        sum_noisy_counts = np.sum(final_noisy_hist_vector)
        est_density_on_final_tree_cells = final_noisy_hist_vector / sum_noisy_counts if sum_noisy_counts > 0 else final_noisy_hist_vector

        if true_density_on_final_tree_cells.shape == est_density_on_final_tree_cells.shape and true_density_on_final_tree_cells.size > 0:
            mse_on_tree_cells = np.mean((true_density_on_final_tree_cells - est_density_on_final_tree_cells)**2)
            l1_on_tree_cells = np.sum(np.abs(true_density_on_final_tree_cells - est_density_on_final_tree_cells))
            print(f"MSE on final tree's reporting cells (densities): {mse_on_tree_cells:.6e}")
            print(f"L1 dist on final tree's reporting cells (densities): {l1_on_tree_cells:.6f}")
        else:
            print("Could not compute MSE on tree cells due to shape mismatch or empty vectors.")

        # For a more visual comparison, reconstruct full flat heatmaps
        print("\nReconstructing full flat heatmaps for comparison (this can be slow for large grids)...")
        true_flat_map = reconstruct_flat_heatmap_from_tree(true_density_on_final_tree_cells,
                                                           final_hist_map_idx_to_id,
                                                           final_query_tree, (w, h))
        est_flat_map = reconstruct_flat_heatmap_from_tree(est_density_on_final_tree_cells,
                                                          final_hist_map_idx_to_id,
                                                          final_query_tree, (w, h))

        # Normalize flat maps (sum to 1) as reconstruction might not preserve it perfectly
        true_flat_map_sum = np.sum(true_flat_map)
        if true_flat_map_sum > 0: true_flat_map /= true_flat_map_sum
        est_flat_map_sum = np.sum(est_flat_map)
        if est_flat_map_sum > 0: est_flat_map /= est_flat_map_sum

        flat_mse = calculate_mse(true_flat_map, est_flat_map)
        flat_l1 = calculate_l1_dist(true_flat_map, est_flat_map)
        print(f"MSE on reconstructed flat {w}x{h} heatmaps (densities): {flat_mse:.6e}")
        print(f"L1 dist on reconstructed flat {w}x{h} heatmaps (densities): {flat_l1:.6f}")

        # Optional: Plotting with matplotlib
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            if np.any(true_flat_map):
                im_true = axes[0].imshow(true_flat_map, cmap="viridis", origin="lower", interpolation="nearest")
                axes[0].set_title("True Heatmap (Reconstructed)")
                fig.colorbar(im_true, ax=axes[0])
            else:
                axes[0].set_title("True Heatmap (Reconstructed - Empty)")


            if np.any(est_flat_map):
                im_est = axes[1].imshow(est_flat_map, cmap="viridis", origin="lower", interpolation="nearest")
                axes[1].set_title("Estimated DP Heatmap (Reconstructed)")
                fig.colorbar(im_est, ax=axes[1])
            else:
                axes[1].set_title("Estimated DP Heatmap (Reconstructed - Empty)")

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Skipping heatmap plots.")
    else:
        print("Final histogram vector is empty, cannot perform full evaluation.")
