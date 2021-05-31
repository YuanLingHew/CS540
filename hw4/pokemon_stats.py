import csv
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage


class Cluster:
    def __init__(self, points_list, index):
        self.points = points_list
        self.index = index

    def getPointsList (self):
        return self.points

    def getIndex(self):
        return self.index

    def getSize(self):
        return len(self.points)

    def addPoint (self, point):
        self.points.append(point)

    def setIndex (self, index):
        self.index = index


def point_distance(point1, point2):
    return math.sqrt( pow(point1[0]-point2[0], 2) + pow(point1[1]-point2[1], 2))


def cluster_distance(cluster1, cluster2):
    dist_list = []
    for point in cluster1.getPointsList():
        for point_compare in cluster2.getPointsList():
            dist_list.append(point_distance(point, point_compare))

    return min(dist_list)


def load_data(filepath):
    with open(filepath, newline="") as csvfile:
        return_list = []
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            if count != 20:
                del row["Generation"]
                del row["Legendary"]
                return_list.append(row)
            else:
                break
            count += 1

        # Converts strings to integers
        for dic in return_list:
            for key in dic:
                if key != "Name" and key != "Type 1" and key != "Type 2":
                    dic[key] = int(dic[key])

    return return_list


def calculate_x_y(stats):
    return_tuple = ()
    x = stats["Attack"] + stats["Sp. Atk"] + stats["Speed"]
    y = stats["Defense"] + stats["Sp. Def"] + stats["HP"]
    return_tuple = return_tuple + (x,)
    return_tuple = return_tuple + (y,)

    return return_tuple


def hac(dataset):
    # Removing NaN and infinite values
    for points in dataset:
        if math.isnan(points[0]) or math.isnan(points[1]):
            dataset.remove(points)
        elif not math.isfinite(points[0]) or not math.isfinite(points[1]):
            dataset.remove(points)

    # Initializing Z[][]
    Z = []
    for x in range (0, len(dataset)-1):
        c = []
        for y in range (0, 4):
            c.append(0)
        Z.append(c)

    dist_track = []
    cluster_to_num = {}
    num_to_cluster = {}
    dataset_cluster = []
    new_clusters = {}

    # Numbers every point (initial cluster numbers 0-19) tuple->int & int->tuple
    for x in range(0, len(dataset)):
        cluster_to_num[dataset[x]] = x
        num_to_cluster[x] = dataset[x]

    # Converts points into its own cluster
    for x in range (0, len(dataset)):
        clus = Cluster([dataset[x]], x)
        dataset_cluster.append(clus)

    # Compares clusters and calculates distance
    for x in range(0, len(dataset)):
        for y in range(x + 1, len(dataset)):
            dist = cluster_distance(dataset_cluster[x], dataset_cluster[y])
            dist_track.append([dataset_cluster[x].getIndex(), dataset_cluster[y].getIndex(), dist])

    # Find minimum distance
    dist_track.sort(key=lambda x: x[2])

    # Updating Z[][]
    i = 0
    for pair in dist_track:
        paired = False
        merged = False
        double_clus = False
        skipped = False

        # If points/clusters are already in the same cluster, don't merge
        if len(new_clusters) != 0:
            for key in new_clusters:
                if all(x in new_clusters[key].getPointsList() for x in [num_to_cluster[pair[0]], num_to_cluster[pair[1]]]):
                    paired = True
                    skipped = True
                    break

        # Merging time!
        if not paired:
            # Checks if points are already in another cluster
            for key in sorted(new_clusters.keys(), reverse=True):
                # If one of the points are already in another cluster
                if all(x in new_clusters[key].getPointsList() for x in [num_to_cluster[pair[0]]]) or all(x in new_clusters[key].getPointsList() for x in [num_to_cluster[pair[1]]]):
                    # Checks which point has been clustered and gets it's cluster number
                    if all(x in new_clusters[key].getPointsList() for x in [num_to_cluster[pair[0]]]):
                        # Checks if the OTHER point is in another cluster
                        for key3 in sorted(new_clusters.keys(), reverse=True):
                            if all(x in new_clusters[key3].getPointsList() for x in [num_to_cluster[pair[1]]]):
                                double_clus = True
                                temp_clust2 = new_clusters[key3]
                                temp_clust2.setIndex(key3)
                                break

                        for key_ref in sorted(new_clusters.keys(), reverse=True):
                            if all(x in new_clusters[key_ref].getPointsList() for x in [num_to_cluster[pair[0]]]):
                                # If merging an existing cluster with another existing cluster
                                if double_clus:
                                    temp_clust = new_clusters[key_ref]
                                    for point in temp_clust2.getPointsList():
                                        temp_clust.addPoint(point)

                                    temp_clust.setIndex(key_ref)
                                    new_clusters[len(dataset) + i] = temp_clust
                                    Z[i][0] = min(key_ref, key3)

                                else:
                                    temp_clust = new_clusters[key_ref]
                                    temp_clust.addPoint(num_to_cluster[pair[1]])
                                    temp_clust.setIndex(key_ref)
                                    new_clusters[len(dataset) + i] = temp_clust
                                    Z[i][0] = min(key_ref, pair[1])

                                break

                    else:
                        # Checks if the OTHER point is in another cluster
                        for key2 in sorted(new_clusters.keys(), reverse=True):
                            if all(x in new_clusters[key2].getPointsList() for x in [num_to_cluster[pair[0]]]):
                                double_clus = True
                                temp_clust2 = new_clusters[key2]
                                temp_clust2.setIndex(key2)
                                break

                        for key_ref in sorted(new_clusters.keys(), reverse=True):
                            if all(x in new_clusters[key_ref].getPointsList() for x in [num_to_cluster[pair[1]]]):
                                # If merging an existing cluster with another existing cluster
                                if double_clus:
                                    temp_clust = new_clusters[key_ref]
                                    for point in temp_clust2.getPointsList():
                                        temp_clust.addPoint(point)

                                    temp_clust.setIndex(key_ref)
                                    new_clusters[len(dataset) + i] = temp_clust
                                    Z[i][0] = min(key_ref, key2)

                                else:
                                    temp_clust = new_clusters[key_ref]
                                    temp_clust.addPoint(num_to_cluster[pair[0]])
                                    temp_clust.setIndex(key_ref)
                                    new_clusters[len(dataset) + i] = temp_clust
                                    Z[i][0] = min(key_ref, pair[0])

                                break

                    # Merging cluster
                    Z[i][1] = key_ref
                    Z[i][2] = pair[2]
                    Z[i][3] = new_clusters[len(dataset)+i].getSize()

                    merged = True
                    break

            if not merged:
                # If both points are not previously clustered
                new_clusters[len(dataset)+i] = Cluster ([num_to_cluster[pair[0]], num_to_cluster[pair[1]]], len(dataset)+i)
                Z[i][0] = min(pair[0], pair[1])

                if Z[i][0] == pair[0]:
                    Z[i][1] = pair[1]
                else:
                    Z[i][1] = pair[0]

                Z[i][2] = pair[2]
                Z[i][3] = new_clusters[len(dataset)+i].getSize()

        if not skipped:
            i += 1
        paired = False
        merged = False
        double_clus = False

    return np.matrix(Z)


def random_x_y(m):
    return_list = []

    for x in range (0, m):
        val = (random.randint(1, 359), random.randint(1, 359))
        return_list.append(val)

    return return_list


def imshow_hac(dataset):
    # Basically the entirety of hac()
    # Removing NaN and infinite values
    for points in dataset:
        if math.isnan(points[0]) or math.isnan(points[1]):
            dataset.remove(points)
        elif not math.isfinite(points[0]) or not math.isfinite(points[1]):
            dataset.remove(points)

    # Initializing Z[][]
    Z = []
    for x in range(0, len(dataset) - 1):
        c = []
        for y in range(0, 4):
            c.append(0)
        Z.append(c)

    dist_track = []
    cluster_to_num = {}
    num_to_cluster = {}
    dataset_cluster = []
    new_clusters = {}
    filtered_dist_track = []

    # Numbers every point (initial cluster numbers 0-19) tuple->int & int->tuple
    for x in range(0, len(dataset)):
        cluster_to_num[dataset[x]] = x
        num_to_cluster[x] = dataset[x]

    # Converts points into its own cluster
    for x in range(0, len(dataset)):
        clus = Cluster([dataset[x]], x)
        dataset_cluster.append(clus)

    # Compares clusters and calculates distance
    for x in range(0, len(dataset)):
        for y in range(x + 1, len(dataset)):
            dist = cluster_distance(dataset_cluster[x], dataset_cluster[y])
            dist_track.append([dataset_cluster[x].getIndex(), dataset_cluster[y].getIndex(), dist])

    # Find minimum distance
    dist_track.sort(key=lambda x: x[2])

    # Updating Z[][]
    i = 0
    for pair in dist_track:
        paired = False
        merged = False
        double_clus = False
        skipped = False

        # If points/clusters are already in the same cluster, don't merge
        if len(new_clusters) != 0:
            for key in new_clusters:
                if all(x in new_clusters[key].getPointsList() for x in
                       [num_to_cluster[pair[0]], num_to_cluster[pair[1]]]):
                    paired = True
                    skipped = True
                    break

        # Merging time!
        if not paired:
            filtered_dist_track.append(pair)
            # Checks if points are already in another cluster
            for key in sorted(new_clusters.keys(), reverse=True):
                # If one of the points are already in another cluster
                if all(x in new_clusters[key].getPointsList() for x in [num_to_cluster[pair[0]]]) or all(
                        x in new_clusters[key].getPointsList() for x in [num_to_cluster[pair[1]]]):
                    # Checks which point has been clustered and gets it's cluster number
                    if all(x in new_clusters[key].getPointsList() for x in [num_to_cluster[pair[0]]]):
                        # Checks if the OTHER point is in another cluster
                        for key3 in sorted(new_clusters.keys(), reverse=True):
                            if all(x in new_clusters[key3].getPointsList() for x in [num_to_cluster[pair[1]]]):
                                double_clus = True
                                temp_clust2 = new_clusters[key3]
                                temp_clust2.setIndex(key3)
                                break

                        for key_ref in sorted(new_clusters.keys(), reverse=True):
                            if all(x in new_clusters[key_ref].getPointsList() for x in [num_to_cluster[pair[0]]]):
                                # If merging an existing cluster with another existing cluster
                                if double_clus:
                                    temp_clust = new_clusters[key_ref]
                                    for point in temp_clust2.getPointsList():
                                        temp_clust.addPoint(point)

                                    temp_clust.setIndex(key_ref)
                                    new_clusters[len(dataset) + i] = temp_clust
                                    Z[i][0] = min(key_ref, key3)

                                else:
                                    temp_clust = new_clusters[key_ref]
                                    temp_clust.addPoint(num_to_cluster[pair[1]])
                                    temp_clust.setIndex(key_ref)
                                    new_clusters[len(dataset) + i] = temp_clust
                                    Z[i][0] = min(key_ref, pair[1])

                                break

                    else:
                        # Checks if the OTHER point is in another cluster
                        for key2 in sorted(new_clusters.keys(), reverse=True):
                            if all(x in new_clusters[key2].getPointsList() for x in [num_to_cluster[pair[0]]]):
                                double_clus = True
                                temp_clust2 = new_clusters[key2]
                                temp_clust2.setIndex(key2)
                                break

                        for key_ref in sorted(new_clusters.keys(), reverse=True):
                            if all(x in new_clusters[key_ref].getPointsList() for x in [num_to_cluster[pair[1]]]):
                                # If merging an existing cluster with another existing cluster
                                if double_clus:
                                    temp_clust = new_clusters[key_ref]
                                    for point in temp_clust2.getPointsList():
                                        temp_clust.addPoint(point)

                                    temp_clust.setIndex(key_ref)
                                    new_clusters[len(dataset) + i] = temp_clust
                                    Z[i][0] = min(key_ref, key2)

                                else:
                                    temp_clust = new_clusters[key_ref]
                                    temp_clust.addPoint(num_to_cluster[pair[0]])
                                    temp_clust.setIndex(key_ref)
                                    new_clusters[len(dataset) + i] = temp_clust
                                    Z[i][0] = min(key_ref, pair[0])

                                break

                    # Merging cluster
                    Z[i][1] = key_ref
                    Z[i][2] = pair[2]
                    Z[i][3] = new_clusters[len(dataset) + i].getSize()

                    merged = True
                    break

            if not merged:
                # If both points are not previously clustered
                new_clusters[len(dataset) + i] = Cluster([num_to_cluster[pair[0]], num_to_cluster[pair[1]]],
                                                         len(dataset) + i)
                Z[i][0] = min(pair[0], pair[1])

                if Z[i][0] == pair[0]:
                    Z[i][1] = pair[1]
                else:
                    Z[i][1] = pair[0]

                Z[i][2] = pair[2]
                Z[i][3] = new_clusters[len(dataset) + i].getSize()

        if not skipped:
            i += 1
        paired = False
        merged = False
        double_clus = False

    # hac() ends here ########################################################################################
    x_vals = []
    y_vals = []
    col = []

    for x in range (0, len(dataset)):
        r = random.random()
        g = random.random()
        b = random.random()
        col.append((r, g, b))

    for point in dataset:
        x_vals.append(point[0])
        y_vals.append(point[1])

    # plots initial points
    plt.scatter(x_vals, y_vals, color=col)

    x_vals2 = []
    y_vals2 = []

    # plots the linkage process
    for pair in filtered_dist_track:
        x_vals2.append(num_to_cluster[pair[0]][0])
        x_vals2.append(num_to_cluster[pair[1]][0])
        y_vals2.append(num_to_cluster[pair[0]][1])
        y_vals2.append(num_to_cluster[pair[1]][1])
        plt.plot(x_vals2, y_vals2)
        plt.pause(0.1)
        x_vals2.clear()
        y_vals2.clear()

    plt.show()

