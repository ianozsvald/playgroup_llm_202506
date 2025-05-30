import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Observations
# written using aider with claude-3.7-sonnet and then claude-opus-4
# I overwrite the solution with an out of date version in MS Visual Code
# which confused me, I realised I had no idea what the code was doing
# I see code duplication

# prompt for opus 4, to go from a single grid solution to two grids
# > draw_grid currently draws a single grid. I need to achieve a figure that contains two grids, of different sizes. I'd like the two grids to be drawn side by side with a small gap to separate them. the grids will have identical cell sizes
# >  in pixels, so a grid with smaller dimensions will take less space. align the two grids so the top line of the grids has the same y coordinate. arr1 will be drawn on the left, arr2 will be drawn on the right. please update the draw_grid
# >  to take two arrays (which could each be square or rectangular) and to draw them side by side as described


# example heatmaps
# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-pyrm


def create_two_grids(arr1, arr2):
    """
    Draw two grids side by side from 2D numpy arrays.
    The grids can be of shape (2,2) to (30,30) and can be square or rectangular.
    The values of the grid are in the range 0-9

    Parameters:
    arr1 (numpy.ndarray): First 2D numpy array to be visualized as a grid (left).
    arr2 (numpy.ndarray): Second 2D numpy array to be visualized as a grid (right).

    Returns:
    matplotlib.figure.Figure: The figure object containing the grid visualization.
    """
    # Define cell size in inches
    cell_size = 0.5
    gap = 0.5  # gap between grids in inches

    # Calculate figure dimensions
    width1 = arr1.shape[1] * cell_size
    width2 = arr2.shape[1] * cell_size
    height1 = arr1.shape[0] * cell_size
    height2 = arr2.shape[0] * cell_size

    fig_width = width1 + gap + width2
    fig_height = max(height1, height2)

    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Colormap matches the arc agi colour scheme
    # verify at: https://arcprize.org/play
    # https://www.kaggle.com/code/levkayumov/eda-visualization-of-solutionsk
    cmap = mcolors.ListedColormap(
        [
            "#000000",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F012BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",
        ]
    )
    norm = mcolors.Normalize(vmin=0, vmax=9)

    # Calculate positions for subplots
    # Left grid (arr1)
    ax1_left = 0
    ax1_bottom = fig_height - height1
    ax1_width = width1
    ax1_height = height1

    ax1 = fig.add_axes(
        [
            ax1_left / fig_width,
            ax1_bottom / fig_height,
            ax1_width / fig_width,
            ax1_height / fig_height,
        ]
    )

    # Right grid (arr2)
    ax2_left = width1 + gap
    ax2_bottom = fig_height - height2
    ax2_width = width2
    ax2_height = height2

    ax2 = fig.add_axes(
        [
            ax2_left / fig_width,
            ax2_bottom / fig_height,
            ax2_width / fig_width,
            ax2_height / fig_height,
        ]
    )

    # Draw first grid
    im1 = ax1.imshow(arr1, cmap=cmap, norm=norm)

    # Add grid lines for first grid
    ax1.set_xticks(np.arange(-0.5, arr1.shape[1], 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, arr1.shape[0], 1), minor=True)
    ax1.grid(which="minor", color="black", linestyle="-", linewidth=2)

    # Add value labels in each cell for first grid
    # for i in range(arr1.shape[0]):
    #    for j in range(arr1.shape[1]):
    #        ax1.text(j, i, str(arr1[i, j]),
    #                ha="center", va="center",
    #                color="white", fontsize=12, fontweight="bold")

    # Remove ticks for first grid
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Draw second grid
    im2 = ax2.imshow(arr2, cmap=cmap, norm=norm)

    # Add grid lines for second grid
    ax2.set_xticks(np.arange(-0.5, arr2.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, arr2.shape[0], 1), minor=True)
    ax2.grid(which="minor", color="black", linestyle="-", linewidth=2)

    # Add value labels in each cell for second grid
    # for i in range(arr2.shape[0]):
    #    for j in range(arr2.shape[1]):
    #        ax2.text(j, i, str(arr2[i, j]),
    #                ha="center", va="center",
    #                color="white", fontsize=12, fontweight="bold")

    # Remove ticks for second grid
    ax2.set_xticks([])
    ax2.set_yticks([])

    return fig


def create_and_save_grid(arr1, arr2, filename="grid.png"):
    """
    Create a grid from two arrays and save it to a file.

    Parameters:
    arr1 (numpy.ndarray): First 2D numpy array.
    arr2 (numpy.ndarray): Second 2D numpy array.
    filename (str): The name of the file to save the grid image.
    """
    fig = create_two_grids(arr1, arr2)
    print(f"Saving grid as {filename}")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    # Draw and save the example grids
    # example grids
    arr1 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    arr2 = np.random.randint(0, 10, size=(30, 30))
    # arr2 = np.ones((30, 30), dtype=int)# * np.random.randint(0, 10, size=(30, 30))

    create_and_save_grid(arr1, arr2, filename='grid_example.png')
    # fig = draw_grid(arr1, arr2)
    # plt.close(fig)  # Close the figure to free memory
