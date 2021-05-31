from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # TODO: add your code here
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x


def get_covariance(dataset):
    # TODO: add your code here
    ret = np.dot(np.transpose(dataset), dataset)
    return ret * (1 / (len(dataset) - 1))


def get_eig(S, m):
    # TODO: add your code here
    egval, ret_egvec = eigh(S, subset_by_index=[1024-m, 1023])
    egval = -np.sort(-egval)
    ret_egvec = np.fliplr(ret_egvec)
    ret_eigenvals = np.zeros((m, m))
    np.fill_diagonal(ret_eigenvals, egval)
    return ret_eigenvals, ret_egvec


def get_eig_perc(S, perc):
    # TODO: add your code here
    sum = 0
    index = 1025
    egval = eigh(S, eigvals_only=True)
    for x in egval:
        sum += x
    for m in egval:
        index -= 1
        if (m/sum) > perc:
            break

    return get_eig(S, index)


def project_image(img, U):
    # TODO: add your code here
    alpha = np.dot(U.T, img)
    proj = np.ndarray(shape=(1024,))
    index = 0
    for j in U:
        proj[index] = np.dot(j, alpha)
        index += 1
    return proj


def display_image(orig, proj):
    # TODO: add your code here
    project = np.reshape(proj, (32, 32), order="F")
    original = np.reshape(orig, (32, 32), order="F")
    figure, axs = plt.subplots (1,2, figsize=(14.6, 4.8))
    axs[0].set_title("Original")
    axs[1].set_title("Projection")

    cb1 = axs[0].imshow(original, aspect="equal")
    cb2 = axs[1].imshow(project, aspect="equal")

    figure.colorbar(cb1, ax=axs[0])
    figure.colorbar(cb2, ax=axs[1])
    plt.show()

