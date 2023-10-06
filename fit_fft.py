import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt

# Set directory
current_directory = os.getcwd()

# List all .csv files in the 'rawdata' directory
rawdatadir = "rawdata"
filenames = [f for f in os.listdir(rawdatadir) if f.endswith('.csv')]
filelist = [os.path.splitext(f)[0] for f in filenames]
fileno = 0

# Print file list
for i, file in enumerate(filelist):
    prefix = "*" if i == fileno else " "
    print(f"{prefix}{i + 1}\t{file}")

def extract_sample_rate(filename):
    with open(filename, 'r') as file:
        # Read the first 20 lines (this number can be adjusted if needed)
        lines = [file.readline().strip() for _ in range(20)]

        # Search for the line containing "Sample Interval"
        for line in lines:
            if "Sample Interval" in line:
                # Split the line by commas and extract the sample rate
                parts = line.split(',')
                for i, part in enumerate(parts):
                    if "Sample Interval" in part:
                        return float(parts[i + 1])
    return None  # Return None if the sample rate is not found

def get_header_length(filename):
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if "TIME,CH1" in line:
                return i
    return None  # Return None if the header end is not found

# Test the function
filename = os.path.join(rawdatadir, filenames[fileno])
header_length = get_header_length(filename)
print(header_length)
sample_rate = extract_sample_rate(filename)
print(sample_rate)

# Read input data
inputdata = pd.read_csv(filename,skiprows=header_length,header=0,usecols=['CH1'], dtype={'CH1': np.float64})['CH1'].values
tracelength = len(inputdata)
deltaT = sample_rate*1e6
#timeaxis = np.arange(0, tracelength * deltaT, deltaT)
#timetrace = np.column_stack((timeaxis, inputdata))

timechunks = 60
blocksize = int(timechunks / deltaT)
print(blocksize)
#blocksize = 24999
numblocks = tracelength // blocksize
deltaf = 1 / (deltaT * blocksize)
frequencyaxis = np.arange(0, blocksize) * deltaf
blockdata = [inputdata[i * blocksize:(i + 1) * blocksize] for i in range(numblocks)]

powerspecstart = 5
powerspecstop = 500
FTvals = [np.fft.fft(block, norm="ortho") for block in blockdata]
powerspec = [np.column_stack((frequencyaxis, np.abs(ft)**2))[int(powerspecstart/deltaf):int(powerspecstop/deltaf)] for ft in FTvals]

estimateddeviation = 1
maxindices = [np.argmax(block[:, 1]) for block in powerspec]
maxfreqs, maxpower = zip(*[block[maxidx] for block, maxidx in zip(powerspec, maxindices)])
minindex = int((min(maxfreqs) - 5 - estimateddeviation) / deltaf)
maxindex = int((max(maxfreqs) - 4 + estimateddeviation) / deltaf)
powerspecreduced = [block[minindex:maxindex] for block in powerspec]
peakareas = [block[block[:, 1] > maxp / np.exp(1)**2] for block, maxp in zip(powerspecreduced, maxpower)]
maxwidth = [deltaf/2 if len(area) <= 1 else (len(area) - 1) * deltaf/2 for area in peakareas]

def gaussian(x, x0, a, b):
    return a * np.exp(-(x - x0)**2 / (2 * b**2))

gaussfits = []
for block, mf, mp, mw in zip(powerspecreduced, maxfreqs, maxpower, maxwidth):
    try:
        fit = curve_fit(gaussian, block[:, 0], block[:, 1], p0=[mf, mp, mw], maxfev=5000)[0]
        gaussfits.append(fit)
    except RuntimeError:
        print("Failed to fit block with mf:", mf, "mp:", mp, "mw:", mw)
        gaussfits.append([np.nan, np.nan, np.nan])  # Append NaN values for failed fits

freq = [fit[0] for fit in gaussfits]
widths = [fit[2] for fit in gaussfits]
frequencycurve = np.column_stack((np.arange(0.5, numblocks + 0.5) / deltaf / 1000, freq))
widthscurve = np.column_stack((np.arange(0.5, numblocks + 0.5) / deltaf / 1000, np.array(widths) * 2.35482))

element=541
plt.plot(powerspecreduced[element][:,0],powerspecreduced[element][:,1])
plt.plot(np.arange(min(maxfreqs)-estimateddeviation,max(maxfreqs)+estimateddeviation,.01),gaussian(np.arange(min(maxfreqs)-estimateddeviation,max(maxfreqs)+estimateddeviation,.01),gaussfits[element][0],gaussfits[element][1],gaussfits[element][2]))
plt.show()

exportdir = os.path.join(current_directory, f"FT/blocksize_{blocksize}")
if not os.path.exists(exportdir):
    os.makedirs(exportdir)

pd.DataFrame(frequencycurve, columns=["Time", "Frequency"]).to_csv(os.path.join(exportdir, f"freq_{filelist[fileno]}.csv"), index=False)
pd.DataFrame(widthscurve, columns=["Time", "Width"]).to_csv(os.path.join(exportdir, f"sigma_{filelist[fileno]}.csv"), index=False)
