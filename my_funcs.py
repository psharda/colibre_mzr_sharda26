import numpy as np
import h5py
import pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit, root_scalar

simpath = 'simulated_data/'
simba_0_mass = np.load(simpath + 'SIMBA_z=0_mass.npy')
simba_0_metallicity = np.load(simpath + 'SIMBA_z=0_metallicity.npy')
simba_1_mass = np.load(simpath + 'SIMBA_z=1_mass.npy')
simba_1_metallicity = np.load(simpath + 'SIMBA_z=1_metallicity.npy')
simba_2_mass = np.load(simpath + 'SIMBA_z=2_mass.npy')
simba_2_metallicity = np.load(simpath + 'SIMBA_z=2_metallicity.npy')
simba_3_mass = np.load(simpath + 'SIMBA_z=3_mass.npy')
simba_3_metallicity = np.load(simpath + 'SIMBA_z=3_metallicity.npy')
simba_4_mass = np.load(simpath + 'SIMBA_z=4_mass.npy')
simba_4_metallicity = np.load(simpath + 'SIMBA_z=4_metallicity.npy')
simba_5_mass = np.load(simpath + 'SIMBA_z=5_mass.npy')
simba_5_metallicity = np.load(simpath + 'SIMBA_z=5_metallicity.npy')
simba_6_mass = np.load(simpath + 'SIMBA_z=6_mass.npy')
simba_6_metallicity = np.load(simpath + 'SIMBA_z=6_metallicity.npy')
simba_7_mass = np.load(simpath + 'SIMBA_z=7_mass.npy')
simba_7_metallicity = np.load(simpath + 'SIMBA_z=7_metallicity.npy')
simba_8_mass = np.load(simpath + 'SIMBA_z=8_mass.npy')
simba_8_metallicity = np.load(simpath + 'SIMBA_z=8_metallicity.npy')

#my functions

#retrieve the snapshot number corresponding to the given redshift
def get_snapshot_index(filename, target_z=0, tol=1e-6):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # skip header
    lines = lines[1:]

    for idx, line in enumerate(lines):
        z_str, _ = line.split(',')
        z = float(z_str.strip())

        if abs(z - target_z) < tol:
            return idx  # zero-based index

    raise ValueError(f"Redshift {target_z} not found in file.")

#retrieve threshold sSFR based on the redshift, following Chaikin+25b section 3.5 (sSFR_threshold = 0.2/t_Hubble(z))
def get_sSFR_thresh_thz(redshift=0.0):
    Hz = cosmo.H(redshift)
    t_H = (1/Hz).to(u.Gyr)
    sSFR_th = 0.2/t_H
    return sSFR_th.value.item()

# Impose lognormal scatter to take Eddington bias into account (equation 43 of Schaye+2025)
def scatter_mstar(bb, z, vary_with_z=True, scatter_val=0.0):
    # Compute the standard deviation
    if vary_with_z:
        sd = np.minimum(0.1 + 0.1 * z, 0.3)
    else:
        sd = scatter_val
    log_bb = np.log10(bb[0])
    log_bb_scattered = log_bb + np.random.normal(loc=0.0, scale=sd, size=bb[0].size)
    bb_scattered = 10**log_bb_scattered
    return bb_scattered

# npz version of above: Impose lognormal scatter to take Eddington bias into account (equation 43 of Schaye+2025)
def scatter_mstar_npz(bb, z, vary_with_z=True, scatter_val=0.0):
    # Compute the standard deviation
    if vary_with_z:
        sd = np.minimum(0.1 + 0.1 * z, 0.3)
    else:
        sd = scatter_val
    log_bb = np.log10(bb['mstar'])
    log_bb_scattered = log_bb + np.random.normal(loc=0.0, scale=sd, size=bb['mstar'].size)
    bb_scattered = 10**log_bb_scattered
    return bb_scattered

def plot_median(aa, bb, xlim_min=5.95, xlim_max=11.95, bins=61):
    x = aa #log10 stellar mass
    y = bb #12+log(O/H)

    # Define linear bins of 0.1 dex width - same bins as Garcia+2025
    bins = np.linspace(xlim_min, xlim_max, bins)
    bin_centers = (bins[:-1]+bins[1:])/2.0 #np.sqrt(bins[:-1] * bins[1:]) - geometric mean

    medians = []
    counts = []
    p16 = []
    p84 = []
    
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        counts.append(np.sum(mask))  # number of galaxies in bin

        if np.any(mask):
            medians.append(np.median(y[mask]))
            p16.append(np.percentile(y[mask], 16))
            p84.append(np.percentile(y[mask], 84))
        else:
            medians.append(np.nan)
            p16.append(np.nan)
            p84.append(np.nan)

    medians = np.array(medians)
    counts = np.array(counts)
    p16 = np.array(p16)
    p84 = np.array(p84)
    return bin_centers, medians, counts, p16, p84

def make_shades(hex_color, n=5):
    """
    Generate n shades from light to dark for a given hex color.
    """
    base = np.array(matplotlib.colors.to_rgb(hex_color))
    
    shades = []
    # factors go from very light (blend with white) to slightly darkened
    factors = np.linspace(0.79, 0.01, n)  # tweak if you want stronger contrast
    
    for f in factors:
        # blend with white: new = base*(1-f) + white*f
        color = base * (1 - f) + np.ones(3) * f
        shades.append(color)
    
    return shades


#curti2020 calibration for N2 based on Te-based metallicities, the inverse of which we use to convert from N2 reported in Wuyts+2016 to metallicity:
def Curti20N2(Z):
    x=Z-8.69
    c0=-0.489
    c1=1.513
    c2=-2.554
    c3=-5.293
    c4=-2.867
    return c0*x**0 + c1*x**1 + c2*x**2 + c3*x**3 + c4*x**4

def invert_Curti20N2(y, Zmin=7.0, Zmax=10.0, ngrid=1000):
    Zgrid = np.linspace(Zmin, Zmax, ngrid)
    fvals = Curti20N2(Zgrid) - y
    
    sign_change = np.where(np.diff(np.sign(fvals)))[0]
    
    if len(sign_change) == 0:
        return np.nan  #we skip data where N2>-0.3 where the curve bends over
    
    i = sign_change[0]
    a, b = Zgrid[i], Zgrid[i+1]
    
    sol = root_scalar(lambda Z: Curti20N2(Z) - y, bracket=[a, b], method='brentq')
    return sol.root

def plot_obsv_data(ax, zorder=3):
    median_ms = 100 #larger marker size for median datapoints
    #overplot observational data
    #z=0, Tremonti+2004
    df = h5py.File('observed_data/Tremonti2004_Data.hdf5')
    xx = np.log10(df['x/values'][:])
    yy = df['y/values'][:]
    ax[0][0].scatter(xx, yy, marker='>', facecolor='violet', edgecolor='k', label='SDSS DR2', zorder=7, 
                     s=median_ms)
    bb = pd.read_csv('observed_data/lee2006_spitzer_z=0.csv')
    ax[0][0].scatter(bb['logM'], bb['OH'], marker='s', facecolor='white', edgecolor='k', 
                     label='Lee+2006', zorder=zorder)
    #z=0 best-fit relation from Yates+2020
    xx = np.linspace(5.6,10.0,10)
    yy=0.293*xx + 5.575 #equation 14
    ax[0][0].scatter(xx,yy,marker='1',color='k',s=median_ms,zorder=zorder,label='SDSS MaNGA')
    #z=0, Curti+2020
    df = h5py.File('observed_data/Curti2020.hdf5')
    xx = np.log10(df['x/values'][:])
    yy = df['y/values'][:]
    ax[0][0].scatter(xx, yy, marker='o', facecolor='cyan', edgecolor='k', label='SDSS DR7', zorder=zorder,
                    s=median_ms)
    #z=0, FM+2022
    df = h5py.File('observed_data/Fraser-McKelvie_2021.hdf5')
    xx = np.log10(df['x/values'][:]/1.989e30) #base units are kg
    yy = df['y/values'][:]
    ax[0][0].scatter(xx, yy, marker='^', facecolor='white', edgecolor='k', label='SAMI', zorder=zorder,
                    s=median_ms)
    
    #z=0.8, Jain+2026
    bb = pd.read_csv('observed_data/jain2026_archival_z=0.8.csv')
    ax[0][1].scatter(bb['logM'], bb['O_H_O_based'], marker='D', facecolor='yellow', edgecolor='k', 
                     zorder=5, label='MOSDEF', s=median_ms)
    #z=0.8, Zahid+2011
    bb = pd.read_csv('observed_data/zahid2011_z0.8_deep2_fig6.csv')
    ax[0][1].scatter(bb['logMstar'], bb['OH'], marker='<', facecolor='blue', edgecolor='k', zorder=5, 
                     label='DEEP2', s=median_ms)
    
    #z=3.3, Jain+2026
    bb = pd.read_csv('observed_data/jain2026_archival_z=3.3.csv')
    ax[0][2].scatter(bb['logM'], bb['O_H_O_based'], marker='D', facecolor='yellow', edgecolor='k', 
                     zorder=zorder, s=median_ms)

    #z=3.5, Stanton+2024
    bb = pd.read_csv('observed_data/stanton2024_nirvandels_z3.5.csv')
    ax[0][2].scatter(bb['logMstar'], bb['OH'], marker='o', facecolor='red', edgecolor='k', 
                     zorder=zorder,label='NIRVANDELS')

    #z=6-7, Kotiwale+2026
    bb = fits.open('observed_data/kotiwale2026_MZRtable_GRISM_z6.fits')
    ax[1][1].scatter(bb[1].data['logM50'], bb[1].data['Zgas50'], marker='s', facecolor='gold', 
                     edgecolor='k', zorder=zorder, label=r'EIGER+ALT+COLA1',s=median_ms)

    #z=0.6-1.8, Gillmann+2021
    bb = pd.read_csv('observed_data/gillmann2021_kross_kges_z=0.6-1.8.csv')
    mask = (bb['z'] >= 0.7) & (bb['z'] <= 1.3)
    aa2,bb2,cc2,*_ = plot_median(np.log10(bb['Mass'][mask]), bb['OH'][mask], xlim_min=9, xlim_max=11, bins=11)
    ax[0][1].scatter(aa2,bb2,marker='*',facecolor='cyan',edgecolor='k',zorder=zorder,
                     label=r'KROSS+KGES', s=median_ms)

    #z=5, Faisst+2026 (filename: 2.csv uses direct Te abundances wherever available. .csv uses strong line abundances)
    df = pd.read_csv('observed_data/faisst2026_z=5_alpine_cristal_jwst2.csv')
    #xerr = np.vstack([df["Mstar_err_minus"], df["Mstar_err_plus"]])
    #yerr = np.vstack([df["OH_strong_err_minus"], df["OH_strong_err_plus"]])
    #ax[1][0].errorbar(df['Mstar'], df['OH_strong'], xerr, yerr, fmt='o', mfc='grey', mec='k', ecolor='black', label='CRISTAL')
    #ax[1][0].scatter(df['Mstar'], df['OH_strong'], marker='o', facecolor='grey', edgecolor='k', 
    #                 label='ALPINE-CRISTAL', zorder=zorder)
    ax[1][0].scatter(df['Mstar'], df['OH'], marker='D', facecolor='white', edgecolor='m', 
                     label='ALPINE-CRISTAL', zorder=zorder)

    #z=10.38 galaxy from Curtis-Lake+2023, Table 1
    ax[1][2].scatter(7.58, 6.78, marker='o', facecolor='k', edgecolor='green', 
                     label='Curtis-Lake+2023', zorder=5)
    #z=10.05 galaxy from Álvarez-Márquez+2026, abstract
    ax[1][2].scatter(np.log10(1.7e8), 7.292, marker='v', facecolor='k', edgecolor='green', 
                     label='Álvarez-Márquez+2026', zorder=5)
    #z=9.3 galaxy from Bik+2026
    ax[1][2].scatter(np.log10(1.6e9), 7.84, marker='s', facecolor='cyan', edgecolor='k', 
                     label='Bik+2026', zorder=5)
    #z=10.165 galaxy from Hsiao+2024
    ax[1][2].scatter(8.1, 7.8, marker='*', facecolor='white', edgecolor='gold', label='Hsiao+2024', s=100, 
                     zorder=7)
    #z=8 galaxy from Wilott+2025 (data from Table C2 of Isobe+2026)
    ax[1][1].scatter(7.67, 6.81, marker='^', facecolor='cyan', edgecolor='k', label='Wilott+2025', 
                     zorder=zorder)
    #z=8-10 galaxies from Koller+2026 (data from Table C2 of Isobe+2026)
    ax[1][1].scatter([6.77, 7.29], [7.11, 7.11], marker='<', facecolor='green', edgecolor='k', label='Koller+2026', 
                     zorder=zorder)
    #z=9.5 galaxy from Koller+2026 (data from Table C2 of Isobe+2026)
    ax[1][2].scatter([6.83], [7.23], marker='<', facecolor='green', edgecolor='k', 
                     zorder=zorder)
    #z=8 galaxy from Mowla+2024 (data from Table C2 of Isobe+2026)
    ax[1][1].scatter([7.88], [6.99], marker='<', facecolor='white', edgecolor='m',label='Mowla+2024', 
                     zorder=zorder)

    #z=9-10, Pollock+2025
    bb = pd.read_csv('observed_data/pollock2025_archival_z=9-10.csv')
    ax[1][2].scatter(bb['logMstar'], bb['OH'], marker='D', facecolor='white', edgecolor='k', zorder=7, 
                     label='Pollock+2026')
    
    #z=6-8, Chemerynska+2024
    bb = pd.read_csv('observed_data/chemerynska2024_uncover_z=6-8.csv')
    ax[1][1].scatter(bb['logMstar'], bb['OH'], marker='*', facecolor='r', edgecolor='k', s=100,zorder=zorder, 
                     label='UNCOVER')
    
    #z=6-7, Rowland+2025
    df = pd.read_csv('observed_data/rowland2025_z=6-7_rebels.csv')
    xerr = np.vstack([df["Mstar_err_minus"], df["Mstar_err_plus"]])
    yerr = df["OH_strong_err"]
    labels = False
    for idx, row in df.iterrows():
        if 4 <= row['z'] < 6:
            i, j = 1, 0
        elif 6 <= row['z'] < 9:
            i, j = 1, 1
        else:
            continue
        if labels==False:
            #print('First REBELS galaxy in ', i, j)
            #ax[i][j].errorbar(row['Mstar'], row['OH_strong'], xerr=[[row['Mstar_err_minus']], [row['Mstar_err_plus']]], yerr=[row['OH_strong_err']],
            #                  fmt='o', mfc='yellow', mec='k', ecolor='black', label='REBELS')
            ax[i][j].scatter(row['Mstar'], row['OH_strong'],
                              marker='X', facecolor='olive', edgecolor='k', label='REBELS', zorder=zorder)
            labels = True
        else:
            #ax[i][j].errorbar(row['Mstar'], row['OH_strong'], xerr=[[row['Mstar_err_minus']], [row['Mstar_err_plus']]], yerr=[row['OH_strong_err']],
            #                  fmt='o', mfc='yellow', mec='k', ecolor='black')
            ax[i][j].scatter(row['Mstar'], row['OH_strong'],
                             marker='X', facecolor='olive', edgecolor='k', zorder=zorder)

    #z=5-7, Hsiao+2025 (data from Table C2 of Isobe+2026)
    df = pd.read_csv('observed_data/hsiao2025_sapphires_z5-6.csv')
    labels = False
    for idx, row in df.iterrows():
        if 4 <= row['z'] < 6:
            i, j = 1, 0
        elif 6 <= row['z'] < 9:
            i, j = 1, 1
        else:
            continue
        if labels==False:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                              marker='s', facecolor='blue', edgecolor='k', label='SAPPHIRES', zorder=zorder)
            labels = True
        else:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='s', facecolor='blue', edgecolor='k', zorder=zorder)

    #z=6-7, Hsiao+2026
    df = pd.read_csv('observed_data/hsiao2026_z6_glimpsed.csv')
    labels = False
    for idx, row in df.iterrows():
        if 4 <= row['z'] < 6:
            i, j = 1, 0
        elif 6 <= row['z'] < 9:
            i, j = 1, 1
        else:
            continue
        if labels==False:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                              marker='s', facecolor='white', edgecolor='m', label='GLIMPSE-D', zorder=zorder)
            labels = True
        else:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                              marker='s', facecolor='white', edgecolor='m', zorder=zorder)

    #z=2-8, Stanton+2026
    df = pd.read_csv('observed_data/stanton2026_z=2-8_excels.csv')
    # Loop over each row and plot in the right subplot
    labels = False
    for idx, row in df.iterrows():
        if row['z_spec'] < 2:
            i, j = 0, 1
        elif 2 <= row['z_spec'] < 4:
            i, j = 0, 2
        elif 4 <= row['z_spec'] < 6:
            i, j = 1, 0
        elif 6<= row['z_spec'] < 9:
            i, j = 1, 1
        elif row['z_spec'] >= 9:
            i, j = 1, 2
        else:
            continue
    
        if labels==False:
            #print('First EXCELS galaxy in ', i, j)
            #ax[i][j].errorbar(row['logMstar_val'], row['OH_strong_val'], xerr=[[row['logMstar_err_minus']], [row['logMstar_err_plus']]], yerr=[[row['OH_strong_err_minus']], [row['OH_strong_err_plus']]],
            #                  fmt='o', mfc='white', mec='m', ecolor='black', label='EXCELS')
            ax[i][j].scatter(row['logMstar_val'], row['OH_strong_val'],
                             marker='o', facecolor='white', edgecolor='m', label='EXCELS', zorder=zorder)
            labels = True
        else:
            #ax[i][j].errorbar(row['logMstar_val'], row['OH_strong_val'], xerr=[[row['logMstar_err_minus']], [row['logMstar_err_plus']]], yerr=[[row['OH_strong_err_minus']], [row['OH_strong_err_plus']]],
            #                  fmt='o', mfc='white', mec='m', ecolor='black')
            ax[i][j].scatter(row['logMstar_val'], row['OH_strong_val'],
                             marker='o', facecolor='white', edgecolor='m', zorder=zorder)

    #z=1-7, Sanders+2025
    df = pd.read_csv('observed_data/sanders2025_aurora_z=1-7.csv')
    # Loop over each row and plot in the right subplot
    labels = False
    for idx, row in df.iterrows():
        if row['z_spec'] < 2:
            i, j = 0, 1
        elif 2 <= row['z_spec'] < 4:
            i, j = 0, 2
        elif 4 <= row['z_spec'] < 6:
            i, j = 1, 0
        elif 6<= row['z_spec'] < 9:
            i, j = 1, 1
        elif row['z_spec'] >= 9:
            i, j = 1, 2
        else:
            continue
    
        if labels==False:
            #print('First AURORA galaxy in ', i, j)
            ax[i][j].scatter(row['logMstar'], row['O_H'],
                             marker='v', facecolor='white', edgecolor='k', label='AURORA', zorder=8, s=50)
            labels = True
        else:
            ax[i][j].scatter(row['logMstar'], row['O_H'],
                             marker='v', facecolor='white', edgecolor='k', zorder=8, s=50)

    #z=1-2, Henry+2021
    df = pd.read_csv('observed_data/henry2021_z=1-2_candels_wisp_tab5.csv')
    # Loop over each row and plot in the right subplot
    labels = False
    for idx, row in df.iterrows():
        if row['z'] < 2:
            i, j = 0, 1
        elif 2 <= row['z'] < 4:
            i, j = 0, 2
        elif 4 <= row['z'] < 6:
            i, j = 1, 0
        elif 6<= row['z'] < 9:
            i, j = 1, 1
        elif row['z'] >= 9:
            i, j = 1, 2
        else:
            continue
    
        if labels==False:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='v', facecolor='gold', edgecolor='k', label='CANDELS', zorder=zorder)
            labels = True
        else:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='v', facecolor='gold', edgecolor='k', zorder=zorder)

    
    #z=3-10, Curti+2024
    df = pd.read_csv('observed_data/curti2024_z=3-10_jades.csv')
    # Loop over each row and plot in the right subplot
    labels = False
    for idx, row in df.iterrows():
        if row['Redshift'] < 2:
            i, j = 0, 1
        elif 2 <= row['Redshift'] < 4:
            i, j = 0, 2
        elif 4 <= row['Redshift'] < 6:
            i, j = 1, 0
        elif 6 <= row['Redshift'] < 9:
            i, j = 1, 1
        elif row['Redshift'] >= 9:
            i, j = 1, 2
        else:
            continue
    
        if labels==False:
            #print('First JADES galaxy in ', i, j)
            #ax[i][j].errorbar(row['logMstar'], row['OH'], xerr=[[row['logMstar_err_minus']], [row['logMstar_err_plus']]], yerr=[[row['OH_err_minus']], [row['OH_err_plus']]],
            #                  fmt='o', ecolor='black', label='JADES', mfc='white', mec='b')
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='o', label='JADES', facecolor='grey', edgecolor='k', zorder=zorder)
            labels = True
        else:
            #ax[i][j].errorbar(row['logMstar'], row['OH'], xerr=[[row['logMstar_err_minus']], [row['logMstar_err_plus']]], yerr=[[row['OH_err_minus']], [row['OH_err_plus']]],
            #                  fmt='o', ecolor='black', mfc='white', mec='b')
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='o', facecolor='grey', edgecolor='k', zorder=zorder)

    #z=4-10, Sarkar+2025
    df = pd.read_csv('observed_data/sarkar2025_archival_z=4-10.csv')
    labels = False
    for idx, row in df.iterrows():
        if row['z'] < 2:
            i, j = 0, 1
        elif 2 <= row['z'] < 4:
            i, j = 0, 2
        elif 4 <= row['z'] < 6:
            i, j = 1, 0
        elif 6 <= row['z'] < 9:
            i, j = 1, 1
        elif row['z'] >= 9:
            i, j = 1, 2
        else:
            continue
    
        if labels==False:
            #print('First Sarkar galaxy in ', i, j)
            ax[i][j].scatter(row['logMstar'], row['logOH'],
                             marker='*', label='PRIMAL', facecolor='yellowgreen', edgecolor='k', zorder=5, s=100)
            labels = True
        else:
            ax[i][j].scatter(row['logMstar'], row['logOH'],
                             marker='*', facecolor='yellowgreen', edgecolor='k', zorder=5, s=100)

    
    #z=4-8, Nakajima+2023
    df = pd.read_csv('observed_data/nakajima2023_z=4-8_ero_glass_ceers.csv')
    # Loop over each row and plot in the right subplot
    labels = False
    for idx, row in df.iterrows():
        if row['zspec'] < 2:
            i, j = 0, 1
        elif 2 <= row['zspec'] < 4:
            i, j = 0, 2
        elif 4 <= row['zspec'] < 6:
            i, j = 1, 0
        elif 6 <= row['zspec'] < 9:
            i, j = 1, 1
        else:
            continue
    
        if labels==False:
            #print('First ERO+GLASS+CEERS galaxy in ', i, j)
            #ax[i][j].errorbar(row['logMstar'], row['logOH'], xerr=[[row['e_logMstar']], [row['E_logMstar']]], yerr=[[row['e_logOH']], [row['E_logOH']]],
            #                  fmt='o', ecolor='black', label='ERO+GLASS+CEERS', mfc='white', mec='cyan')
            ax[i][j].scatter(row['logMstar'], row['logOH'],
                             marker='o', label='ERO+GLASS+CEERS', facecolor='white', edgecolor='blue', zorder=zorder)
            labels = True
        else:
            #ax[i][j].errorbar(row['logMstar'], row['logOH'], xerr=[[row['e_logMstar']], [row['E_logMstar']]], yerr=[[row['e_logOH']], [row['E_logOH']]],
            #                  fmt='o', ecolor='black', mfc='white', mec='cyan')
            ax[i][j].scatter(row['logMstar'], row['logOH'],
                             marker='o', facecolor='white', edgecolor='blue', zorder=zorder)

    #z=1-10, Isobe+2026
    df = pd.read_csv('observed_data/isobe2026_jades_darkhorse_oasis_z1-10.csv')
    # Loop over each row and plot in the right subplot
    labels = False
    for idx, row in df.iterrows():
        if row['z'] < 2:
            i, j = 0, 1
        elif 2 <= row['z'] < 4:
            i, j = 0, 2
        elif 4 <= row['z'] < 6:
            i, j = 1, 0
        elif 6 <= row['z'] < 9:
            i, j = 1, 1
        else:
            i, j = 1,2
    
        if labels==False:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='o', label='JADES+DH+OASIS', facecolor='yellow', edgecolor='k', zorder=zorder)
            labels = True
        else:
           ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='o', facecolor='yellow', edgecolor='k', zorder=zorder)
 
    #z=7-9, Langeroodi+2023
    df = pd.read_csv('observed_data/langeroodi2023_archival_z=8.csv')
    # Loop over each row and plot in the right subplot
    labels = False
    for idx, row in df.iterrows():
        if row['z_spec'] < 2:
            i, j = 0, 1
        elif 2 <= row['z_spec'] < 4:
            i, j = 0, 2
        elif 4 <= row['z_spec'] < 6:
            i, j = 1, 0
        elif 6 <= row['z_spec'] < 9:
            i, j = 1, 1
        elif row['z_spec'] >= 9:
            i, j = 1, 2
        else:
            continue
    
        if labels==False:
            ax[i][j].scatter(row['logM_star'], row['O_H'],
                             marker='*', label='Langeroodi+2023', facecolor='white', edgecolor='k', s=100, zorder=zorder)
            labels = True
        else:
            ax[i][j].scatter(row['logM_star'], row['O_H'],
                             marker='*', facecolor='white', edgecolor='k', s=100, zorder=zorder)

    #z=2-4, Cataldi+2025
    df = pd.read_csv('observed_data/cataldi2025_z=2-4_marta_tab2.csv')
    # Loop over each row and plot in the right subplot
    labels = False
    for idx, row in df.iterrows():
        if row['z'] < 2:
            i, j = 0, 1
        elif 2 <= row['z'] < 4:
            i, j = 0, 2
        elif 4 <= row['z'] < 6:
            i, j = 1, 0
        elif 6 <= row['z'] < 9:
            i, j = 1, 1
        elif row['z'] >= 9:
            i, j = 1, 2
        else:
            continue
    
        if labels==False:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='X', label='MARTA', facecolor='white', edgecolor='k', s=50, zorder=8)
            labels = True
        else:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='X', facecolor='white', edgecolor='k', s=50, zorder=8)

    #z=2-4, Raptis+2025
    df = pd.read_csv('observed_data/raptis2025_z=2-4_cecilia.csv')
    # Loop over each row and plot in the right subplot
    labels = False
    for idx, row in df.iterrows():
        if row['z'] < 2:
            i, j = 0, 1
        elif 2 <= row['z'] < 4:
            i, j = 0, 2
        elif 4 <= row['z'] < 6:
            i, j = 1, 0
        elif 6 <= row['z'] < 9:
            i, j = 1, 1
        elif row['z'] >= 9:
            i, j = 1, 2
        else:
            continue
    
        if labels==False:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='o', label='CECILIA', facecolor='gold', edgecolor='k', zorder=8)
            labels = True
        else:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='o', facecolor='gold', edgecolor='k', zorder=8)

    #z=0.6-2.7, Wuyts+2016
    df = pd.read_csv('observed_data/wuyts2016_kmos3d_z=0.6-2.7.csv')
    xx=np.log10(df['NIIHa'])
    yy=np.array([invert_Curti20N2(x) for x in xx])
    # Loop over each row and plot in the right subplot
    labels = False
    for (idx, row), yval in zip(df.iterrows(), yy):
        if row['z'] < 1.5:
            i, j = 0, 1
        elif 2.5 <= row['z'] < 3:
            i, j = 0, 2
        else:
            continue
    
        if not labels:
            ax[i][j].scatter(row['logMstar'], yval, marker='+', label='KMOS3D', facecolor='green', 
                             zorder=zorder, s=100)
            labels = True
        else:
            ax[i][j].scatter(row['logMstar'], yval, marker='+', facecolor='green', zorder=zorder, s=100)

    #z=1-3, He+2026
    df = pd.read_csv('observed_data/he2026_z1-3_ngdeep_stacked.csv')
    # Loop over each row and plot in the right subplot
    labels = False
    for idx, row in df.iterrows():
        if row['z'] < 2:
            i, j = 0, 1
        elif 2 <= row['z'] < 4:
            i, j = 0, 2
        else:
            continue
    
        if labels==False:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='s', label=r'NGDEEP', facecolor='m', edgecolor='k', zorder=zorder,
                            s=median_ms)
            labels = True
        else:
            ax[i][j].scatter(row['logMstar'], row['OH'],
                             marker='s', facecolor='m', edgecolor='k', zorder=zorder, s=median_ms)

    return None



def plot_obsv_data_z(ax, zorder=3, logmsel=8, dm=0.25):
    median_ms = 100 #larger marker size for median datapoints
    #overplot observational data
    #z=0, Tremonti+2004
    df = h5py.File('observed_data/Tremonti2004_Data.hdf5')
    xx = np.log10(df['x/values'][:])
    yy = df['y/values'][:]
    mask = np.abs(xx - logmsel) <= dm
    ax.scatter(0*xx[mask]/xx[mask], yy[mask], marker='>', facecolor='violet', edgecolor='k', label='SDSS DR2', zorder=7, 
                     s=median_ms)

    bb = pd.read_csv('observed_data/lee2006_spitzer_z=0.csv')
    mask = np.abs(bb['logM'] - logmsel) <= dm
    ax.scatter(0*bb['logM'][mask]/bb['logM'][mask], bb['OH'][mask], marker='s', facecolor='white', edgecolor='k', 
                     label='Lee+2006', zorder=zorder)

    #z=0, Curti+2020
    df = h5py.File('observed_data/Curti2020.hdf5')
    xx = np.log10(df['x/values'][:])
    yy = df['y/values'][:]
    mask = np.abs(xx - logmsel) <= dm
    ax.scatter(0*xx[mask]/xx[mask], yy[mask], marker='o', facecolor='cyan', edgecolor='k', label='SDSS DR7', zorder=zorder,
                    s=median_ms)

    #z=0, FM+2022
    df = h5py.File('observed_data/Fraser-McKelvie_2021.hdf5')
    xx = np.log10(df['x/values'][:]/1.989e30) #base units are kg
    yy = df['y/values'][:]
    mask = np.abs(xx - logmsel) <= dm
    ax.scatter(0*xx[mask]/xx[mask], yy[mask], marker='^', facecolor='white', edgecolor='k', label='SAMI', zorder=zorder,
                    s=median_ms)
    
    #z=0.8, Jain+2026
    bb = pd.read_csv('observed_data/jain2026_archival_z=0.8.csv')
    mask = np.abs(bb['logM'] - logmsel) <= dm
    ax.scatter(0.8*bb['logM'][mask]/bb['logM'][mask], bb['O_H_O_based'][mask], marker='D', facecolor='yellow', edgecolor='k', 
                     zorder=5, label='MOSDEF', s=median_ms)

    #z=0.8, Zahid+2011
    bb = pd.read_csv('observed_data/zahid2011_z0.8_deep2_fig6.csv')
    mask = np.abs(bb['logMstar'] - logmsel) <= dm
    ax.scatter(0.8*bb['logMstar'][mask]/bb['logMstar'][mask], bb['OH'][mask], marker='<', facecolor='blue', edgecolor='k', zorder=5, 
                     label='DEEP2', s=median_ms)
    
    #z=3.3, Jain+2026
    bb = pd.read_csv('observed_data/jain2026_archival_z=3.3.csv')
    mask = np.abs(bb['logM'] - logmsel) <= dm
    ax.scatter(3.3*bb['logM'][mask]/bb['logM'][mask], bb['O_H_O_based'][mask], marker='D', facecolor='yellow', edgecolor='k', 
                     zorder=zorder, s=median_ms)

    #z=3.5, Stanton+2024
    bb = pd.read_csv('observed_data/stanton2024_nirvandels_z3.5.csv')
    mask = np.abs(bb['logMstar'] - logmsel) <= dm
    ax.scatter(bb['z'][mask], bb['OH'][mask], marker='o', facecolor='red', edgecolor='k', 
                     zorder=zorder, label='NIRVANDELS')

    #z=6-7, Kotiwale+2026
    bb = fits.open('observed_data/kotiwale2026_MZRtable_GRISM_z6.fits')
    mask = np.abs(bb[1].data['logM50'] - logmsel) <= dm
    ax.scatter(6.5*bb[1].data['logM50'][mask]/bb[1].data['logM50'][mask], bb[1].data['Zgas50'][mask], marker='s', facecolor='gold', 
                     edgecolor='k', zorder=zorder, label=r'EIGER+ALT+COLA1',s=median_ms)

    #z=0.6-1.8, Gillmann+2021
    bb = pd.read_csv('observed_data/gillmann2021_kross_kges_z=0.6-1.8.csv')
    mask = np.abs(np.log10(bb['Mass']) - logmsel) <= dm
    ax.scatter(bb['z'][mask],bb['OH'][mask],marker='*',facecolor='cyan',edgecolor='k',zorder=zorder,
                     label=r'KROSS+KGES', s=median_ms)

    #z=5, Faisst+2026 (filename: 2.csv uses direct Te abundances wherever available. .csv uses strong line abundances)
    df = pd.read_csv('observed_data/faisst2026_z=5_alpine_cristal_jwst2.csv')
    #xerr = np.vstack([df["Mstar_err_minus"], df["Mstar_err_plus"]])
    #yerr = np.vstack([df["OH_strong_err_minus"], df["OH_strong_err_plus"]])
    #ax.errorbar(df['Mstar'], df['OH_strong'], xerr, yerr, fmt='o', mfc='grey', mec='k', ecolor='black', label='CRISTAL')
    #ax.scatter(df['Mstar'], df['OH_strong'], marker='o', facecolor='grey', edgecolor='k', 
    #                 label='ALPINE-CRISTAL', zorder=zorder)
    mask = np.abs(df['Mstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], df['OH'][mask], marker='D', facecolor='white', edgecolor='m', 
                     label='ALPINE-CRISTAL', zorder=zorder)


    if np.abs(logmsel-7.58) <= dm:
        #z=10.38 galaxy from Curtis-Lake+2023, Table 1
        ax.scatter(10.38, 6.78, marker='o', facecolor='k', edgecolor='green', 
                         label='Curtis-Lake+2023', zorder=5)
    if np.abs(logmsel-np.log10(1.7e8)) <= dm:
        #z=10.05 galaxy from Álvarez-Márquez+2026, abstract
        ax.scatter(10.05, 7.292, marker='v', facecolor='k', edgecolor='green', 
                         label='Álvarez-Márquez+2026', zorder=5)
    if np.abs(logmsel-np.log10(1.6e9)) <= dm:
        #z=9.3 galaxy from Bik+2026
        ax.scatter(9.3, 7.84, marker='s', facecolor='cyan', edgecolor='k', 
                         label='Bik+2026', zorder=5)
    if np.abs(logmsel-8.1) <= dm:
        #z=10.165 galaxy from Hsiao+2024
        ax.scatter(10.165, 7.8, marker='*', facecolor='white', edgecolor='gold', label='Hsiao+2024', s=100, 
                         zorder=7)
    if np.abs(logmsel-6.77) <= dm:
        #z=8-10 galaxies from Koller+2026 (data from Table C2 of Isobe+2026)
        ax.scatter(8.45, 7.11, marker='<', facecolor='green', edgecolor='k', label='Koller+2026', 
                     zorder=zorder)

    if np.abs(logmsel-7.29) <= dm:
        #z=8-10 galaxies from Koller+2026 (data from Table C2 of Isobe+2026)
        ax.scatter(8.45, 7.11, marker='<', facecolor='green', edgecolor='k', label='Koller+2026', 
                     zorder=zorder)

    if np.abs(logmsel-6.83) <= dm:
        #z=9.5 galaxy from Koller+2026 (data from Table C2 of Isobe+2026)
        ax.scatter(9.51, 7.23, marker='<', facecolor='green', edgecolor='k', 
                         zorder=zorder)

    if np.abs(logmsel-7.88) <= dm:
        #z=8 galaxy from Mowla+2024 (data from Table C2 of Isobe+2026)
        ax.scatter(8.30, 6.99, marker='<', facecolor='white', edgecolor='m',label='Mowla+2024', 
                         zorder=zorder)
    if np.abs(logmsel-7.67) <= dm:
        #z=8 galaxy from Wilott+2025 (data from Table C2 of Isobe+2026)
        ax.scatter(8.20, 6.81, marker='^', facecolor='cyan', edgecolor='k',label='Wilott+2025', 
                         zorder=zorder)

    #z=9-10, Pollock+2025
    bb = pd.read_csv('observed_data/pollock2025_archival_z=9-10.csv')
    mask = np.abs(bb['logMstar'] - logmsel) <= dm
    ax.scatter(bb['z'][mask], bb['OH'][mask], marker='D', facecolor='white', edgecolor='k', zorder=7, 
                     label='Pollock+2026')
    
    #z=6-8, Chemerynska+2024
    bb = pd.read_csv('observed_data/chemerynska2024_uncover_z=6-8.csv')
    mask = np.abs(bb['logMstar'] - logmsel) <= dm
    ax.scatter(bb['z'][mask], bb['OH'][mask], marker='*', facecolor='r', edgecolor='k', s=100,zorder=zorder, 
                     label='UNCOVER')
    
    #z=6-7, Rowland+2025
    df = pd.read_csv('observed_data/rowland2025_z=6-7_rebels.csv')
    mask = np.abs(df['Mstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], df['OH_strong'][mask],
                      marker='X', facecolor='olive', edgecolor='k', label='REBELS', zorder=zorder)


    #z=5-7, Hsiao+2025 (data from Table C2 of Isobe+2026)
    df = pd.read_csv('observed_data/hsiao2025_sapphires_z5-6.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], df['OH'][mask],
                      marker='s', facecolor='blue', edgecolor='k', label='SAPPHIRES', zorder=zorder)

    #z=6-7, Hsiao+2026
    df = pd.read_csv('observed_data/hsiao2026_z6_glimpsed.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], df['OH'][mask],
                      marker='s', facecolor='white', edgecolor='m', label='GLIMPSE-D', zorder=zorder)

    #z=2-8, Stanton+2026
    df = pd.read_csv('observed_data/stanton2026_z=2-8_excels.csv')
    mask = np.abs(df['logMstar_val'] - logmsel) <= dm
    ax.scatter(df['z_spec'][mask], df['OH_strong_val'][mask],
                     marker='o', facecolor='white', edgecolor='m', label='EXCELS', zorder=zorder)

    #z=1-7, Sanders+2025
    df = pd.read_csv('observed_data/sanders2025_aurora_z=1-7.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['z_spec'][mask], df['O_H'][mask],
                     marker='v', facecolor='white', edgecolor='k', label='AURORA', zorder=8, s=50)

    #z=1-2, Henry+2021
    df = pd.read_csv('observed_data/henry2021_z=1-2_candels_wisp_tab5.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], df['OH'][mask],
                     marker='v', facecolor='gold', edgecolor='k', label='CANDELS', zorder=zorder)
    
    #z=3-10, Curti+2024
    df = pd.read_csv('observed_data/curti2024_z=3-10_jades.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['Redshift'][mask], df['OH'][mask],
                     marker='o', label='JADES', facecolor='grey', edgecolor='k', zorder=zorder)

    #z=4-10, Sarkar+2025
    df = pd.read_csv('observed_data/sarkar2025_archival_z=4-10.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], df['logOH'][mask],
                     marker='*', label='PRIMAL', facecolor='yellowgreen', edgecolor='k', zorder=5, s=100)
    
    #z=4-8, Nakajima+2023
    df = pd.read_csv('observed_data/nakajima2023_z=4-8_ero_glass_ceers.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['zspec'][mask], df['logOH'][mask],
                     marker='o', label='ERO+GLASS+CEERS', facecolor='white', edgecolor='blue', zorder=zorder)

    #z=7-9, Langeroodi+2023
    df = pd.read_csv('observed_data/langeroodi2023_archival_z=8.csv')
    mask = np.abs(df['logM_star'] - logmsel) <= dm
    ax.scatter(df['z_spec'][mask], df['O_H'][mask],
                     marker='*', label='Langeroodi+2023', facecolor='white', edgecolor='k', s=100, zorder=zorder)

    #z=2-4, Cataldi+2025
    df = pd.read_csv('observed_data/cataldi2025_z=2-4_marta_tab2.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], df['OH'][mask],
                     marker='X', label='MARTA', facecolor='white', edgecolor='k', s=50, zorder=8)

    #z=2-4, Raptis+2025
    df = pd.read_csv('observed_data/raptis2025_z=2-4_cecilia.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], df['OH'][mask],
                     marker='o', label='CECILIA', facecolor='gold', edgecolor='k', zorder=8)

    #z=1-10, Isobe+2026
    df = pd.read_csv('observed_data/isobe2026_jades_darkhorse_oasis_z1-10.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], df['OH'][mask],
                     marker='o', label='JADES+DH+OASIS', facecolor='yellow', edgecolor='k', zorder=zorder)

    #z=0.6-2.7, Wuyts+2016
    df = pd.read_csv('observed_data/wuyts2016_kmos3d_z=0.6-2.7.csv')
    xx=np.log10(df['NIIHa'])
    yy=np.array([invert_Curti20N2(x) for x in xx])
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], yy[mask], marker='+', label='KMOS3D', facecolor='green', 
                     zorder=zorder, s=100)

    #z=1-3, He+2026
    df = pd.read_csv('observed_data/he2026_z1-3_ngdeep_stacked.csv')
    mask = np.abs(df['logMstar'] - logmsel) <= dm
    ax.scatter(df['z'][mask], df['OH'][mask],
                     marker='s', label=r'NGDEEP', facecolor='m', edgecolor='k', zorder=zorder,
                    s=median_ms)

    return None


