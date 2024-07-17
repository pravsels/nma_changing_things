import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_performance(paths, output_file=None, title='Model Performance'):
    """
    Plots the performance of multiple models and saves the plot to a file.

    Parameters:
    - paths (list of str): Paths to the experiment directories.
    - output_file (str, optional): Path to save the output plot. If None, the plot is displayed instead.
    - title (str): Title for the plot.
    """
    # Set the Seaborn style
    sns.set(style="whitegrid")
    colors = sns.color_palette("colorblind")  # Colorblind-friendly palette

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, path in enumerate(paths):
        # Extract the model name from the path
        model_name = os.path.basename(path.rstrip('/'))

        # Load data
        df = pd.read_csv(os.path.join(path, 'log.csv'))
        scores = df['test/episode_score/mean']
        lengths = df['test/episode_length/mean']
        steps = np.cumsum(lengths)
        sns.lineplot(x=steps, y=scores, ax=ax, label=model_name, color=colors[index % len(colors)])

    ax.set_xscale('log')
    ax.set_xlabel('Cumulative Time Steps')
    ax.set_ylabel('Max Episode Score')
    ax.legend()
    ax.set_title(title)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":

    mlp_model_folder = 'mlp_32'
    ncap_model_folder1 = 'ppo_ncap_model_32'
    # ncap_model_folder2 = 'd4pg_ncap_model_32'
    task_name = 'swimmer-swim_task'

    # Replace these paths with the paths to the models you want to plot
    paths = [
        f'data/local/experiments/tonic/{task_name}/{ncap_model_folder1}',
        # f'data/local/experiments/tonic/{task_name}/{ncap_model_folder2}',
        f'data/local/experiments/tonic/{task_name}/{mlp_model_folder}'
    ]

    # Set the output file path, or set to None to display the plot instead
    output_file = 'model_performance_comparison.png'

    plot_performance(paths, output_file, title='MLP vs NCAP')