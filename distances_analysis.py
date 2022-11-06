from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import global_variables

def distances_analysis(dists, query_solutions):
    # dists is a dictionary with the distances of each painting in the query to each painting in the database
    # dists = {query_painting1: {db_painting1: distance, db_painting2: distance, ...}, query_painting2: {db_painting1: distance, db_painting2: distance, ...}, ...}
    # query_solutions is an array of arrays with the solutions of each query image
    # query_solutions = [[db_painting1, db_painting2, ...], [db_painting1, db_painting2, ...], ...]
    # If the painting is not in the database, the solution is -1
    # Save for each query image the smaller distance of the database paintings
    shorter_distance = {}
    for key_query, _ in dists.items():
        # ! Idea: Compute the variance of the top distances and use it as a threshold
        shorter_distance[key_query] = {}
        sorted_dists = [dist.dist for key, dist in sorted(dists[key_query].items(), key=lambda item: item[1].dist)]
        shorter_distance[key_query]['std_slope_top1_relation'] = (np.std(sorted_dists[:2])) / (np.std(sorted_dists) + 0.0001)
        query_image = int(key_query.split('_')[0])
        query_part_idx = int(key_query.split('_')[1][-1]) - 1
        if query_solutions != []:
            shorter_distance[key_query]['in_db'] = False if query_solutions[query_image][query_part_idx] == -1 else True

    # Sort the dictionary by the smaller first second slope
    shorter_distance = dict(sorted(shorter_distance.items(), key=lambda item: item[1]['std_slope_top1_relation']))

    # Now we have the shorter distance for each query painting and if it is in the database or not
    # Plot a bar chart with the shorter_mean_dist of each query painting
    # If the painting is in the database, the bar is green, if not, the bar is red
    # The x axis is the query painting and the y axis is the shorter_mean_dist
    # Using matplotlib
    query_names = list(shorter_distance.keys())
    query_distances_slope = [shorter_distance[key]['std_slope_top1_relation'] for key in query_names]
    if query_solutions != []:
        query_in_db = [shorter_distance[key]['in_db'] for key in query_names]

    # Plot the bar chart
    fig, ax = plt.subplots()
    ax.axhline(y = global_variables.in_db_threshold, color = 'blue', linestyle = '-')
    color = 'green'
    if query_solutions != []:
        color = ['green' if in_db else 'red' for in_db in query_in_db]
    ax.bar(query_names, query_distances_slope, color = color)
    ax.set_title('Std difference between top 2 distances and the rest for each query painting')
    ax.set_xlabel('Query painting')
    ax.set_ylabel('Std difference between top 2 and the rest')
    # Rotate the x axis labels
    plt.xticks(rotation = 90)
    # Add horizontal line to define treshold
    # Add a legend to explain the colors
    if query_solutions != []:
        green_patch = mpatches.Patch(color = 'green', label = 'In database')
        red_patch = mpatches.Patch(color = 'red', label = 'Not in database')
        blue_patch = mpatches.Patch(color = 'blue', linewidth = 0.5, label = f'Threshold at {round(global_variables.in_db_threshold, 3)} to decide if in database')
        plt.legend(handles=[green_patch, red_patch, blue_patch])
    else:
        blue_patch = mpatches.Patch(color = 'blue', linewidth = 0.5, label = f'Threshold at {round(global_variables.in_db_threshold, 3)} to decide if in database')
        plt.legend(handles=[blue_patch])
    plt.show()
    # Save the plot
    fig.savefig(global_variables.dir_query + global_variables.dir_query_aux + 'distances_analysis.png', bbox_inches='tight')