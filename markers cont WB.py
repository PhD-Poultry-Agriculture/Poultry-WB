# finds markers between control and WB
# 8 animals, 4 time points per animal
# for each feature find the median in each time point. then take abs of (sum over all the diff between the medians over the 4 timepoints).
# permute the labels, and find in how many cases the abs of the sum is bigger than the original. This is the p-value

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
df2 = pd.read_csv('polar data_1 for python.csv') #

print(df2.head()) # [5 rows x 64 columns]
#            T1C1          T1C2  ...         T4WB7         T4WB8
# 0  1.083365e+08  1.251111e+08  ...  7.086391e+07  9.151850e+07
# 1  1.501600e+08  1.784446e+08  ...  1.282625e+08  1.591225e+08
# 2  6.292404e+06  6.447890e+06  ...  5.260634e+06  1.332565e+07
# 3  7.992693e+06  9.065125e+06  ...  8.434318e+05  1.231232e+06
# 4  5.585072e+06  9.806985e+06  ...  4.648566e+06  4.076419e+06
#%%
arr0 = df2.iloc[:, :].values
print(arr0[0,0], arr0[-1,-1]) # 108336452.6 166399478.4

# each row is a feature, each column an animal
# transpose
# make a nested list such that each animal has its four time points together
arr0t = arr0.transpose()
print(arr0t[0,1]) # 150160019.7 - yey
#%%
test0 = cluster_timepoints(0,arr0t) # first animal, four time points, each time point is a list of 234 features
print(test0[0][0], test0[1][0], len(test0[1]), len(test0)) # 108336452.6 83056246.67 234 4 - looking good
# eight controls
control0 = []
for ii in range (0,8):
    tmp = cluster_timepoints(ii,arr0t)
    control0 = control0 + [tmp]
print(len(control0), len(control0[1]), len(control0[3][3]), control0[0][0][0], control0[1][0][1], control0[7][3][3]) # 8 4 234 108336452.6 178444641.3 1035672.596 (8th control aniumal, timepoint4, marker4) - yey

#%% now for all 16 animals
allchick0 = []
for ii in range (0,16):
    tmp = cluster_timepoints(ii,arr0t)
    allchick0 = allchick0 + [tmp]
print(len(allchick0), len(allchick0[1]), len(allchick0[3][3]), allchick0[0][0][0], allchick0[1][0][1], allchick0[7][3][3]) # 16 4 234 108336452.6 178444641.3 1035672.596
print(len(allchick0), len(allchick0[8]), len(allchick0[8][3]), allchick0[8][0][0], allchick0[9][0][1], allchick0[15][3][3]) # 16 4 234 144166113.4 79828260.3 1231231.824 - yey!

#%% test medians difference by changing time point #2 feature #2 by "1" in control, "2" in WB
# allchick0 = replacedata(1, 1, allchick0)

# so allchicks is a list of 16 animals, each animal is a list of 4 time points, each time point has its 234 features
# to calculate the median value of feature #1 in time point number 1 is all 8 controls: median(animlas1-8[0][0])
time00 = extracttp(0, allchick0)
print(len(time00), len(time00[1]), time00[0][0], time00[7][1], time00[8][3]) # 16 234 108336452.6 91429761.94 9922076.487 - yey
print(time00[0]) # first cont chick, all markers from t0
# [108336452.6, 150160019.7, 6292403.651, 7992693.411, 5585071.924, 91712790.15, 1471131.235, 15432340.34, 1071340809.0, 3259382.78, 284311483.0, 8633371.413, 56867871.34, 7788315.251, 54188324.78, 66925627.46, 6570817.492, 70418784.59, 16740344.77, 258683957.9, 1554767451.0, 73696455.21, 309885917.4, 5597416.762, 8183752.319, 3284346.851, 5262678.695, 7897777.297, 13659557.66, 8313571.859, 162795482.8, 860541.6665, 676110.7381, 8006379.489, 18057492.8, 131376731.2, 20886291.5, 2774427728.0, 4974759.743, 95450696.03, 745857827.7, 49034905.34, 2117079.949, 13735712.54, 40317699.46, 34447723.65, 8711250.714, 15615388.45, 26692725.73, 3755988.415, 7630147.121, 30456701.24, 46775519.49, 19940562.84, 4967085.975, 174205049.8, 21503325.35, 8014143.591, 19379654.32, 791684.1298, 72477680.96, 72613909.34, 3146982464.0, 266860147.8, 1059158666.0, 193899191.8, 644728.1683, 104948315.5, 213956704.7, 60723763.61, 12431362.47, 1384255339.0, 3525268.939, 525950541.6, 629876844.2, 1225578357.0, 628837907.2, 711069889.9, 115286775.4, 1961184.878, 505411781.6, 411176027.6, 28015637.37, 4277078235.0, 32174691.16, 19639004.48, 61495375.82, 154851319.5, 515834366.6, 269352621.6, 32988253.84, 110120032.4, 245615.94, 4007892.176, 31238902.9, 13568857.16, 923476137.7, 24897038.74, 7061791.162, 15819566.74, 28885428766.0, 45141450.21, 44433438.99, 59819179.15, 31990354.49, 3139867.282, 67248036.87, 71865102.23, 2004513591.0, 19343627.97, 2121099.213, 52011953.18, 606198855.4, 3659928977.0, 135977.1824, 247203867.0, 964496326.3, 4977246705.0, 27247475.69, 359718772.1, 798373193.7, 300047647.2, 4283943.184, 8555329.422, 6910226.951, 61958389.82, 1329203.4, 103753902.4, 58558354.59, 4559704.135, 15787741.53, 129541779.1, 110873773.5, 10631009.21, 25211382.83, 20387182.62, 108663296.0, 8810182.703, 464712.49, 39407125.46, 1058766298.0, 6633763.826, 446714.4549, 2370452704.0, 1471773.23, 129855230.3, 47900806.63, 243235524.5, 18066799.64, 3128995270.0, 90135294.51, 80653148.46, 256356443.6, 10418388.82, 212313712.6, 26983654.82, 676314494.9, 7189902.239, 346755739.4, 51105336.53, 354943.4273, 21045904.02, 23971384.48, 41300695.33, 258900010.6, 17931568.08, 122412394.5, 3409368.141, 17053005.69, 9833877.122, 4574250.552, 71382573.23, 20784172.84, 180531621.4, 8178221.411, 188351947.8, 1149464.321, 6907275.786, 1600976056.0, 352024502.8, 22762618.11, 13473955.51, 163700929.2, 122695778.1, 7295924.618, 1857861194.0, 912387590.5, 2561044.903, 1521968446.0, 5330426.116, 60636936.11, 20254860.28, 42867348.43, 8103236575.0, 38194870.44, 1576141182.0, 16330510.85, 23914009.06, 84439577.13, 2029000067.0, 11921869.55, 45140957.52, 379809.5927, 49698578.27, 720335199.0, 1738688.608, 294153.0901, 226414766.2, 11233572.23, 8332984.089, 507884839.1, 54041386.8, 5236037.815, 7141469414.0, 20552329.48, 2057961216.0, 23784623.0, 1825607.035, 2415763.911, 57767903.24, 19329938.77, 12415683.92, 19159973.31, 19947757.6, 1292449464.0, 775690119.6, 35057513.83, 1701845.829, 7494094364.0, 320867827.2, 271562268.0, 14080880.9, 126244708.5, 147342133.1]

#%% test calcmedian
tmptest = [[1,2,3], [1,2,3], [1,2,3]]
tmptest1 = [[1,1,1], [2,2,2], [3,3,3]]
testmed = calcmedian(tmptest)
testmed1 = calcmedian(tmptest1)
print(testmed, testmed1) # [1. 2. 3.] [2. 2. 2.] - yey

# calculate the medians of t0
time00cont = time00[:8]
time00wb = time00[8:]
time0medianscont = calcmedian(time00cont)
time0medianswb = calcmedian(time00wb)

print(len(time0medianscont)) # 234 - for each marker its median over cont animals
print(len(time0medianswb)) # 234 - for each marker its median over wb animals
print(type(time0medianscont)) # <class 'numpy.ndarray'>
#%% for each marker, the diff between
diffmedtp0 = time0medianscont - time0medianswb
print(len(diffmedtp0)) # 234
print(diffmedtp0[:4]) # first four
# [-10532808.69999999  36914374.55000001   -694406.1775 429758.293     ]

#%% extract the four time points
timedata = []
for ii in range(0,4):
    timetmp = extracttp(ii, allchick0)
    timedata = timedata + [timetmp]

print(len(timedata), len(timedata[1]), len(timedata[0][0]), timedata[0][0][0], timedata[3][8][-1]) # 4 16 234 108336452.6 158043428.6 - yey

#%% calculate the medians of timedata[0]
time0cont = timedata[0][:8]
time0wb = timedata[0][8:]
time0contmedians = calcmedian(time0cont)
time0wbmedians = calcmedian(time0wb)
print(len(time0contmedians)) # 234 - for each marker its median over cont animals
print(len(time0wbmedians)) # 234 - for each marker its median over wb animals
print(type(time0contmedians)) # <class 'numpy.ndarray'>
# for each marker, the diff between
diffmedtp00 = time0contmedians - time0wbmedians
print(len(diffmedtp00)) # 234
print(diffmedtp00[:4]) #
# [-10532808.69999999  36914374.55000001   -694406.1775 429758.293     ] - yey

# calculate the medians for each time point
meddiff = []
for ii in range(0, 4):
    timeconttmp = timedata[ii][:8]
    timewbtmp = timedata[ii][8:]
    timecontmedianstmp = calcmedian(timeconttmp)
    timewbmedianstmp = calcmedian(timewbtmp)
    diffmedtptmp = timecontmedianstmp - timewbmedianstmp
    meddiff = meddiff + [diffmedtptmp]
print(len(meddiff), len(meddiff[3]), meddiff[0][3]) # 4 234 429758.2929999996 - yey

# make meddiff an array - 4 rows (time points diff) and 234 columns (features)
# sum the abs over the columns (axis = 0) - this will give the 234 total values of the diffrences between cont and wb for the unpermuted, true system forthe 4 time points.

# also test by replacing one feature in one time point over all 16 animals by "1" in cont "2" in WB
# look at replacedata, allchick0
# print(meddiff[1][1]) # -1.0 - yey

timemeddiffarr = np.array(meddiff) # 4 rows (each time point) 234 columns (the median difference for each feature)
sumtimemeddiff = abs(np.sum(timemeddiffarr, axis = 0))
# for each of the 234 features it gives the abs of the sum of median difference between cont and wb over all 4 timepoints.
print(len(sumtimemeddiff), sumtimemeddiff) # 234, indeed all positive
print(timemeddiffarr[0:4,0], sumtimemeddiff[0])
# [-10532808.69999999  28641961.21499999  12911503.75999999 11326192.77499999] 42346849.04999998 (4.23468490e+07) - yey
# print to file
# ff1 = open('mediandiff unpermuted cont vs  WB', 'w')
# for ii in sumtimemeddiff:
#     ff1.write(str(ii) + '\n')
# ff1.close() # yey
# permute tha labels (animals) and test for significance
diffgreaterthanunpermute = [0]*234

# for each feature will tell how many times (out of 100/500) the abs of sum of diff over 4 timepoints is greater than it initial, unpermuted atate
for ii in range(0,500):
    permchick = np.random.permutation(allchick0)
    timedataperm = []
    for ii in range(0, 4):
        timetmp = extracttp(ii, permchick)
        timedataperm = timedataperm + [timetmp]
    # calculate the medians for each time point
    meddiffperm = []
    for ii in range(0, 4):
        timeconttmp = timedataperm[ii][:8]
        timewbtmp = timedataperm[ii][8:]
        timecontmedianstmp = calcmedian(timeconttmp)
        timewbmedianstmp = calcmedian(timewbtmp)
        diffmedtptmp = timecontmedianstmp - timewbmedianstmp
        meddiffperm = meddiffperm + [diffmedtptmp]
    timemeddiffarrperm = np.array(meddiffperm)  # 4 rows (each time point) 234 columns (the median difference for each feature)
    sumtimemeddiffperm = abs(np.sum(timemeddiffarrperm, axis=0))
    for jj in range(0,234):
        if sumtimemeddiffperm[jj] > sumtimemeddiff[jj]:
            diffgreaterthanunpermute[jj] = diffgreaterthanunpermute[jj] + 1


# test for bugs.. - test directly feature # 25 after 1 permutation
print(len(timedataperm), len(timedataperm[0]), len(timedataperm[0][0])) # 4 16 234 - yey

# # specifically feature extract feature # 25
# feat25perm = exctractfeat(25, timedataperm)
# print(len(feat25perm)) # 64 - yey
# print(feat25perm) # two iterations - test that the animals are premutated: all first 16 are t1,
# # first animal at time 2 (num 17) t2of t1animal1! checking al values - implicitly the animals are permuted!
# # [6011560.014, 10924294.32, 7714847.981, 5277724.454, 19313425.59, 8829279.916, 12341236.63, 7085710.285, 14514056.01, 7502747.112, 4153419.918, 15839632.01, 8183752.319, 13123449.26, 8389464.705, 6821299.427, 5038859.624, 10408060.04, 3143891.656, 4005828.953, 19057645.97, 5397370.368, 4200985.092, 5103283.075, 3688404.863, 4168520.935, 7155568.326, 4132423.5, 11298827.81, 5104155.823, 4326046.349, 4136726.969, 42354596.26, 28578808.61, 12090951.87, 19057506.59, 24254171.56, 40791309.62, 22629972.7, 18761644.18, 5283635.831, 6942406.723, 17879704.99, 11310738.76, 19797355.16, 33155184.25, 6212622.869, 8013716.075, 14696029.38, 4810133.417, 18022890.6, 22168151.39, 15772849.15, 30992025.27, 26178632.32, 43750500.31, 13400639.5, 7101348.785, 10882089.29, 4047841.008, 23053091.05, 7070199.938, 19493500.05, 9624012.407]
# #  [8183752.319, 12341236.63, 4153419.918, 13123449.26, 8389464.705, 5277724.454, 10924294.32, 14514056.01, 15839632.01, 19313425.59, 6821299.427, 8829279.916, 7085710.285, 7714847.981, 6011560.014, 7502747.112, 11298827.81, 4200985.092, 7155568.326, 5104155.823, 4326046.349, 4005828.953, 10408060.04, 3688404.863, 4132423.5, 19057645.97, 4136726.969, 5397370.368, 5103283.075, 3143891.656, 5038859.624, 4168520.935, 19797355.16, 22629972.7, 17879704.99, 33155184.25, 6212622.869, 19057506.59, 28578808.61, 5283635.831, 11310738.76, 24254171.56, 8013716.075, 40791309.62, 18761644.18, 12090951.87, 42354596.26, 6942406.723, 23053091.05, 26178632.32, 10882089.29, 7070199.938, 19493500.05, 22168151.39, 4810133.417, 13400639.5, 4047841.008, 15772849.15, 9624012.407, 30992025.27, 43750500.31, 18022890.6, 14696029.38, 7101348.785]

# print(len(permchick), len(permchick[8]), len(permchick[8][3])) # 16 4 234 - yey

print(diffgreaterthanunpermute)
# after 500 perutation, all elements smaller than 25 are supposedly p < 0.05 (need also FDR correction)
# [90, 227, 352, 404, 277, 241, 264, 85, 304, 191, 490, 330, 427, 274, 192, 481, 301, 319, 433, 423, 236, 466, 366, 58, 3, 55, 51, 290, 7, 489, 443, 384, 183, 257, 151, 332, 161, 389, 93, 117, 168, 5, 62, 327, 181, 144, 345, 0, 322, 313, 25, 34, 371, 40, 80, 23, 404, 16, 0, 0, 79, 2, 101, 101, 443, 291, 183, 223, 102, 438, 462, 185, 373, 3, 0, 196, 436, 60, 45, 9, 416, 22, 9, 24, 22, 1, 110, 1, 97, 383, 496, 333, 15, 1, 8, 223, 389, 2, 51, 230, 315, 3, 95, 34, 103, 340, 26, 178, 378, 443, 123, 46, 10, 193, 107, 388, 1, 12, 21, 32, 5, 8, 115, 27, 20, 340, 441, 192, 281, 3, 159, 220, 201, 109, 158, 64, 310, 16, 142, 46, 392, 319, 0, 0, 0, 140, 0, 406, 137, 262, 433, 77, 76, 306, 241, 178, 11, 350, 56, 109, 347, 141, 29, 330, 427, 325, 144, 6, 20, 194, 30, 497, 127, 37, 79, 488, 83, 109, 4, 186, 97, 65, 54, 324, 23, 8, 242, 65, 0, 294, 0, 217, 26, 1, 199, 52, 209, 120, 14, 499, 76, 3, 11, 358, 4, 33, 127, 15, 205, 454, 16, 22, 139, 115, 439, 462, 333, 499, 207, 10, 9, 0, 421, 74, 44, 11, 280, 10, 9, 476, 159, 46, 433, 51]
# potentially significants - 25, 29,...

# test feature # 25 directly
# median con/WB by excel, feature 25: 23968666.52, just like median diff file, yey!
# also, cont > WB at all time points, so it makes sense as a true marker

# get the p-values
permpval0 = [ii/500 for ii in diffgreaterthanunpermute]
# FDR
markrscorrlst = fdrandmarkers(permpval0) # returns the list of markers numbers (starts with 1, not 0!) that passes the threshold (0.25).
print(len(markrscorrlst)) # FDR 0.1 - 26
print(markrscorrlst)
# [25, 29, 48, 59, 60, 62, 74, 75, 86, 88, 94, 98, 102, 117, 118, 121, 130, 143, 144, 145, 147, 189, 191, 194, 202, 222]
# second iteration: 26
# [25, 29, 42, 48, 59, 60, 74, 75, 86, 88, 94, 98, 102, 117, 121, 143, 144, 145, 147, 189, 191, 194, 202, 205, 208, 222]
# third iteration: 38
# [25, 29, 42, 48, 59, 60, 62, 74, 75, 80, 83, 86, 88, 94, 95, 98, 102, 117, 118, 121, 122, 130, 143, 144, 145, 147, 168, 179, 186, 189, 191, 194, 202, 205, 208, 222, 226, 229]
# fourth iteration: 24
# [42, 48, 59, 60, 62, 74, 75, 86, 88, 94, 98, 102, 117, 130, 143, 144, 145, 147, 189, 191, 194, 202, 205, 222]
# fifth iteration: 29
# [25, 29, 42, 48, 59, 60, 62, 74, 75, 88, 94, 95, 98, 117, 121, 130, 143, 144, 145, 147, 168, 179, 189, 191, 194, 202, 203, 205, 222]

# sixth iteration 1st @ my desktop: 29
# [25, 29, 42, 48, 59, 60, 62, 74, 75, 83, 86, 88, 94, 98, 102, 117, 121, 122, 130, 143, 144, 145, 147, 189, 191, 194, 205, 222, 229]

# seventh iteration: 30
# [29, 42, 48, 59, 60, 62, 74, 75, 86, 88, 94, 98, 102, 117, 121, 130, 143, 144, 145, 147, 168, 179, 189, 191, 194, 202, 203, 205, 222, 228]

# 8th iteration: 33
# [25, 29, 42, 48, 59, 60, 62, 74, 75, 80, 86, 88, 94, 95, 98, 102, 117, 121, 122, 130, 143, 144, 145, 147, 179, 189, 191, 194, 202, 205, 220, 222, 226]

# 9th iteration: 34
# [25, 29, 42, 48, 59, 60, 62, 74, 75, 86, 88, 94, 98, 102, 117, 118, 121, 122, 130, 143, 144, 145, 147, 179, 186, 189, 191, 194, 202, 203, 205, 220, 222, 229]

# 10th iteration: 33
# [25, 29, 42, 48, 59, 60, 62, 74, 75, 83, 86, 88, 94, 95, 98, 102, 117, 121, 122, 130, 143, 144, 145, 147, 179, 189, 191, 194, 202, 205, 220, 222, 229]

# features that appear in more than 6 out of 10: 27
# [25, 29, 42, 48, 59, 60, 62, 74, 75, 86, 88, 94, 98, 102, 117, 121, 130, 143, 144, 145, 147, 189, 191, 194, 202, 205, 222]

# generate a file with the markers values for later PCA use
arr0lst = arr0.tolist()
markersnum = [25, 29, 42, 48, 59, 60, 62, 74, 75, 86, 88, 94, 98, 102, 117, 121, 130, 143, 144, 145, 147, 189, 191, 194, 202, 205, 222]
markerslst = []
for ii in range(0, 234):
    if ii + 1 in markersnum:
        markerslst.append(arr0lst[ii])
print(len(markerslst), len(markerslst[0]), markerslst[0][0], markerslst[25][0], markerslst[-1][0]) # 27 64 8183752.319 720335199.0  12415683.92 - yey
# # print to file
# ff1 = open('WB markers polar', 'w')
# for ii in markerslst:
#     for jj in range(0, len(ii)-1):
#         ff1.write(str(ii[jj]) + ',')
#     ff1.write(str(ii[len(ii) - 1]) + '\n')
# ff1.close() # yey

# To present data for PCA, we will have animals in row, marker values as columns.
# For each animal its corresponding marker values the medians over the 4 time points.
# We can also present per time point values. Then it is a regular animals-markers table















