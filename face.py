from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import interpolate
from scipy.stats import pearsonr
from scipy.stats import spearmanr


# Get the basises for interpolation
def getInterval(freq_power_path: str):
    spectrum = pd.read_csv(freq_power_path)
    spectrum['group'] = (spectrum['freq'].shift(1) > spectrum['freq']).cumsum()
    grouped_spectrum = spectrum.groupby('group')
    freq_list = []
    power_list = []

    for _, group in grouped_spectrum:
        freq_list.append(group['freq'].tolist())
        power_list.append(group['power'].tolist())

    return freq_list, power_list

# Interpolate
def getF(freq_list: list, power_list: list):
    f = interpolate.interp1d(freq_list, power_list, fill_value="extrapolate")
    return f

# Interpolate all pairs of sequences between two files
def alignPoints(filepath1: str, filepath2: str):
    freq_list_list_1, power_list_list_1 = getInterval(filepath1)
    freq_list_list_2, power_list_list_2 = getInterval(filepath2)
    y1listlist, y2listlist = [], []
    short_length = len(freq_list_list_1) if len(freq_list_list_1) < len(
        freq_list_list_2) else len(freq_list_list_2)

    for i in range(short_length):
        freq_list1 = freq_list_list_1[i]
        power_list1 = power_list_list_1[i]
        freq_list2 = freq_list_list_2[i]
        power_list2 = power_list_list_2[i]

        func1 = getF(freq_list1, power_list1)
        func2 = getF(freq_list2, power_list2)
        # interpolate
        x = np.linspace(0, 0.5, 1000)
        y1 = func1(x)
        y2 = func2(x)
        y1listlist.append(y1)
        y2listlist.append(y2)

    return x, y1listlist, y2listlist


# Compute Spectral Overlap (SO)
def getSO(filepath1: str, filepath2: str):
    area_floor_list, area_roof_list, so_list = [], [], []
    xlist, y1listlist, y2listlist = alignPoints(filepath1, filepath2)

    for i in range(len(y1listlist)):
        y1list = y1listlist[i]
        y2list = y2listlist[i]
        y1list = [abs(i) for i in y1list]
        y2list = [abs(i) for i in y2list]
        ylists = []
        ylists.append(y1list)
        ylists.append(y2list)

        y_intersection = np.amin(ylists, axis=0)
        y_roof = np.amax(ylists, axis=0)
        area_floor = np.trapz(y_intersection, xlist)
        area_roof = np.trapz(y_roof, xlist)

        area_floor_list.append(area_floor)
        area_roof_list.append(area_roof)
        so_list.append(round(area_floor / area_roof, 4))

    return area_floor_list, area_roof_list, so_list


# Compute Spearman Rank Correlation (SPEAR)
def getSpearmanr(filepath1: str, filepath2: str):
    xlist, y1listlist, y2listlist = alignPoints(filepath1, filepath2)
    corr_list = []
    for i in range(len(y1listlist)):
        y1list = y1listlist[i]
        y2list = y2listlist[i]
        corr, _ = spearmanr(y1list, y2list)
        corr_list.append(corr)
    return corr_list

# Compute Pearson Correlation (CORR)
def getPearson(filepath1: str, filepath2: str):
    xlist, y1listlist, y2listlist = alignPoints(filepath1, filepath2)
    corr_list = []
    for i in range(len(y1listlist)):
        y1list = y1listlist[i]
        y2list = y2listlist[i]
        corr, _ = pearsonr(y1list, y2list)
        corr_list.append(corr)
    return corr_list


# Compute Spectral Angle Mapper (SAM)
def getSAM(filepath1: str, filepath2: str):
    xlist, y1listlist, y2listlist = alignPoints(filepath1, filepath2)
    sam_list = []

    for i in range(len(y1listlist)):
        y1list = y1listlist[i]
        y2list = y2listlist[i]
        ylists = []
        ylists.append(y1list)
        ylists.append(y2list)

        # Normalize the spectra
        y1list /= np.linalg.norm(y1list)
        y2list /= np.linalg.norm(y2list)
        # Calculate the dot product
        dot_product = np.dot(y1list, y2list)
        # Calculate the SAM similarity
        sam_similarity = np.arccos(dot_product) / np.pi
        sam_list.append(sam_similarity)

    return sam_list


##
# Demo example
##
def demo():
    pass

if __name__ == '__main__':
    demo()