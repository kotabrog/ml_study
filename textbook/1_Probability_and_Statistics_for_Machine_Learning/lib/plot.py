import matplotlib.pyplot as plt

def plot_hist(array):
    """Drawing a Histogram
    Args:
        array (np.ndarray):
    """
    fig, ax = plt.subplots()
    ax.hist(array, bins="auto")
    plt.plot()
