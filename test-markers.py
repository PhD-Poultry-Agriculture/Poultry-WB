#%%
def cluster_timepoints(animalnum, arr): # receives the animal number nad returns all timepoints values for this animal
    arrlst = arr.tolist()
    clustertime = [arrlst[animalnum]]
    for ii in range(animalnum+16, 64, 16):
       clustertime = clustertime + [arrlst[ii]]
    return clustertime

def extracttp(tp, lst): # extracts the tp (0,1,2 or 3) from the total list of animals - returns the list of animals, but with only the specific time point
    timepoint0 = []
    for ii in lst:
       tmp = ii[tp]
       timepoint0 = timepoint0 + [tmp]
    return timepoint0

def calcmedian(lst): # calculates the median of al markers over the 8 animals cohort (cont/WB) for a specific time point (tp)
    arrtp = np.array(lst) # every row is an animal, every column feature
    # calculate the median over the columns
    med0 = np.median(arrtp, axis = 0)
    print('snippet of vector before median')
    print(arrtp)
    return med0


def replacedata(featnum, timepoint, lst): # replace one feature in one time point for all animals to "1" in cont and "2" in WB
    newlst = lst[:]
    for ii in range(0, 16):
        if ii < 8:
            newlst[ii][timepoint][featnum] = 1
        elif ii > 7:
            newlst[ii][timepoint][featnum] = 2
    return newlst


def exctractfeat(featnum, lst): #extract all values of the given feature number from the list of 4 time ponits and 16 animals
    valuefeature = []
    for ii in lst:
        for jj in range(0, 16):
            tmp = ii[jj][featnum-1] # feature number 1, is element number 0
            valuefeature = valuefeature + [tmp]
    return valuefeature

def fdrandmarkers(lst): # receives a list of p-vaslues, returns the list of markers that passes FDR
    # fdr0 = statsmodels.stats.multitest.multipletests(lst, alpha=0.25, method='fdr_bh', is_sorted=False, returnsorted=False)  # (lst, alpha=0.1, method='fdr_bh', is_sorted=False, returnsorted=False)
    fdr0 = statsmodels.stats.multitest.multipletests(lst, alpha=0.1, method='fdr_bh', is_sorted=False, returnsorted=False) # (lst, alpha=0.1, method='fdr_bh', is_sorted=False, returnsorted=False)
    mrks = []  # the markers passing FDR
    jj = 1
    for ii in fdr0[0]:
        if ii == True:
            mrks.append(jj)
        jj = jj + 1
    return mrks

def calmediantp(mrkerstable): # for each animal, calculate the median value over 4 timepoints for each marker
    mrkerstablelst = mrkerstable.tolist()
    repvalue = [] # representative value for each marker per each of the 16 animals
    for ii in mrkerstablelst:
        repmarkerval = []
        for jj in range(0, 16):
            tmp = ii[jj][jj] + ii[jj][jj+16] + ii[jj][jj+32] + ii[jj][jj+48]
            tmp1 = np.median(tmp)
            repmarkerval.append(tmp1)
        repvalue.append(repmarkerval)
    return repvalue

from scipy.stats import hypergeom
import statsmodels
import statsmodels.stats.multitest
import statistics
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd

#%%

# Load the data
df2 = pd.read_csv('polar data_1 for python.csv')
print("Data loaded successfully. First 5 rows of the dataframe:")
print(df2.head())  # Display first 5 rows of the dataframe

# Convert dataframe to a NumPy array
arr0 = df2.iloc[:, :].values
print("\nConverted dataframe to NumPy array. First and last element:")
print("First element:", arr0[0, 0], "Last element:", arr0[-1, -1])

# Transpose the array for processing
arr0t = arr0.transpose()
print("\nTransposed the array. Example value after transpose (first row, second column):")
print("Transposed value:", arr0t[0, 1])

# Cluster time points for the first animal
test0 = cluster_timepoints(0, arr0t)
print("\nClustered time points for first animal. Example features:")
print("Timepoint 1, feature 1:", test0[0][0], "Timepoint 2, feature 1:", test0[1][0])
print("Length of features in timepoint 2:", len(test0[1]), "Number of timepoints:", len(test0))

# Cluster time points for all control animals
control0 = []
for ii in range(0, 8):
    tmp = cluster_timepoints(ii, arr0t)
    control0.append(tmp)
print("\nClustered time points for control animals. Validation:")
print("Number of controls:", len(control0), "Timepoints in control 2:", len(control0[1]))
print("Number of features in timepoint 4, control 4:", len(control0[3][3]))
print("First control, timepoint 1, feature 1:", control0[0][0][0])
print("Second control, timepoint 1, feature 2:", control0[1][0][1])
print("Eighth control, timepoint 4, feature 4:", control0[7][3][3])

# Cluster time points for all 16 animals
allchick0 = []
for ii in range(0, 16):
    tmp = cluster_timepoints(ii, arr0t)
    allchick0.append(tmp)
print("\nClustered time points for all animals. Validation:")
print("Number of animals:", len(allchick0), "Timepoints in animal 9:", len(allchick0[8]))
print("First animal, timepoint 1, feature 1:", allchick0[0][0][0])
print("Second animal, timepoint 1, feature 2:", allchick0[1][0][1])
print("Last animal, timepoint 4, feature 4:", allchick0[15][3][3])

# Extract all time points for analysis
time00 = extracttp(0, allchick0)
print("\nExtracted all time points for t0. Validation:")
print("Number of animals:", len(time00), "Number of features:", len(time00[0]))
print("First control chick, all markers from t0:", time00[0])

# Test calcmedian function
tmptest = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
tmptest1 = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
testmed = calcmedian(tmptest)
testmed1 = calcmedian(tmptest1)
print("\nTested calcmedian function:")
print("Test median result:", testmed, "Test median result 2:", testmed1)

# Calculate the medians of t0 for control and WB groups
time00cont = time00[:8]
time00wb = time00[8:]
time0medianscont = calcmedian(time00cont)
time0medianswb = calcmedian(time00wb)
print("\nCalculated medians for control and WB at t0:")
print("Number of features for control:", len(time0medianscont))
print("Number of features for WB:", len(time0medianswb))
print("Type of result:", type(time0medianscont))

# Calculate the differences in medians
diffmedtp0 = time0medianscont - time0medianswb
print("\nCalculated differences in medians. First four differences:")
print(diffmedtp0[:4])

# Extract all time points for further analysis
timedata = []
for ii in range(0, 4):
    timetmp = extracttp(ii, allchick0)
    timedata.append(timetmp)
print("\nExtracted all time points for all animals. Validation:")
print("Number of timepoints:", len(timedata), "Number of animals per timepoint:", len(timedata[0]))
print("Number of features in first animal, first timepoint:", len(timedata[0][0]))

#%% Calculate median differences for each time point
meddiff = []
for ii in range(0, 4):
    timeconttmp = timedata[ii][:8]
    timewbtmp = timedata[ii][8:]
    timecontmedianstmp = calcmedian(timeconttmp)
    timewbmedianstmp = calcmedian(timewbtmp)
    diffmedtptmp = timecontmedianstmp - timewbmedianstmp
    meddiff.append(diffmedtptmp)

print("\nCalculated median differences across all time points:")
print("Number of timepoints:", len(meddiff), "Number of features in last timepoint:", len(meddiff[3]))
print("Difference in feature 4, timepoint 1:", meddiff[0][3])

# Convert meddiff to array and sum absolute differences
timemeddiffarr = np.array(meddiff)
sumtimemeddiff = abs(np.sum(timemeddiffarr, axis=0))
print("\nConverted median differences to array and calculated sums:")
print("Number of features:", len(sumtimemeddiff), "First summed feature:", sumtimemeddiff[0])

# Permute the labels and calculate the median differences for significance testing
diffgreaterthanunpermute = [0] * 234
n_permutations = 500  # Number of permutations

for ii in range(n_permutations):
    permchick = np.random.permutation(allchick0)
    timedataperm = []
    for tp in range(0, 4):
        timetmp = extracttp(tp, permchick)
        timedataperm.append(timetmp)
    
    # Calculate the medians for each time point in the permutation
    meddiffperm = []
    for tp in range(0, 4):
        timeconttmp = timedataperm[tp][:8]
        timewbtmp = timedataperm[tp][8:]
        timecontmedianstmp = calcmedian(timeconttmp)
        timewbmedianstmp = calcmedian(timewbtmp)
        diffmedtptmp = timecontmedianstmp - timewbmedianstmp
        meddiffperm.append(diffmedtptmp)
    
    timemeddiffarrperm = np.array(meddiffperm)
    sumtimemeddiffperm = abs(np.sum(timemeddiffarrperm, axis=0))
    
    # Count how often permuted difference exceeds unpermuted difference
    for jj in range(234):
        if sumtimemeddiffperm[jj] > sumtimemeddiff[jj]:
            diffgreaterthanunpermute[jj] += 1

print("\nCompleted significance testing through permutations.")

# Calculate the p-values based on permutation results
p_values = np.array(diffgreaterthanunpermute) / n_permutations
significance_threshold = 0.05

# Find significant features where p-value < significance threshold
significant_features = np.where(p_values < significance_threshold)[0]

# Print out the significant features
if len(significant_features) > 0:
    print("\nSignificant features found:")
    for feature_idx in significant_features:
        feature_name = df2.columns[feature_idx]
        print(f"Feature: {feature_name}, p-value: {p_values[feature_idx]:.4f}, "
              f"Permuted Diff: {sumtimemeddiffperm[feature_idx]}, Unpermuted Diff: {sumtimemeddiff[feature_idx]}")
else:
    print("\nNo significant features found.")

# %%


# %%
