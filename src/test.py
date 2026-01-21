from datetime import datetime
import numpy as np
from ripser import ripser
from persim import plot_diagrams
from gtda.homology import VietorisRipsPersistence
from sklearn.datasets import make_circles
from numpy.random import default_rng
rng = default_rng(42)  # Create a random number generator

def main():
    # point_cloud = np.asarray([[[-1, 0], [0, -1], [1,0], [0,1], [0.8, 0.8], [-0.8, 0.8], [-0.8, -0.8], [0.8, -0.8]]])
    # ripser_result = ripser(point_cloud, maxdim = 2)
    # print(ripser_result["dgms"])

    X = np.asarray([
        make_circles(5000, factor=np.random.random())[0]
        for i in range(1)
    ])
    #print(X)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[START] {now}")

    ripser_result = ripser(X[0], maxdim = 1)
    print(ripser_result["dgms"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {now}: Ripser")

    VR = VietorisRipsPersistence(homology_dimensions=[0, 1])
    Xt = VR.fit_transform(X)
    print(Xt[0])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {now}: Giotto")

def giotto_test():
    # Create a single weighted adjacency matrix of a FCW graph
    n_vertices = 10
    x = rng.random((n_vertices, n_vertices))
    # Fill the diagonal with zeros (not always necessary, see below)
    np.fill_diagonal(x, 0)

    # Create a trivial collection of weighted adjacency matrices, containing x only
    X = [x]

    # Instantiate topological transformer
    VR = VietorisRipsPersistence(metric="precomputed")

    # Compute persistence diagrams corresponding to each entry (only one here) in X
    diagrams = VR.fit_transform(X)
    print(diagrams)
    print(f"diagrams.shape: {diagrams.shape} ({diagrams.shape[1]} topological features)")

    res = ripser(x, maxdim=2, distance_matrix=True)
    print(res["dgms"])


if __name__ == "__main__":
    main()
    #giotto_test()