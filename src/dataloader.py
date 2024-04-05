import math
import h5py
import pathlib
import parameters
import numpy as np
import pandas as pd
import seaborn as sns
import functions as func
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression


np.random.seed(1)
pars = parameters.get_syn_params()

win_size = pars.get("win_size")
training_length = pars.get("train_len")
prediction_length = pars.get("pred_len")
normalize = func.normalize

def load_river_data():
    # Load river discharges data
    stations = ["dillingen", "kempten", "lenggries"]
        # Read the average daily discharges at each of these stations and combine them into a single pandas dataframe
    average_discharges = None

    for station in stations:

        filename = pathlib.Path("../datasets/river_discharge_data/data_" + station + ".csv")
        new_frame = pd.read_csv(filename, sep=";", skiprows=range(10))
        new_frame = new_frame[["Datum", "Mittelwert"]]

        new_frame = new_frame.rename(columns={"Mittelwert": station.capitalize(), "Datum": "Date"})
        new_frame.replace({",": "."}, regex=True, inplace=True)

        new_frame[station.capitalize()] = new_frame[station.capitalize()].astype(float)

        if average_discharges is None:
            average_discharges = new_frame
        else:
            average_discharges = average_discharges.merge(new_frame, on="Date")
    
    dillingen = average_discharges.iloc[:, 1].tolist()
    kempton = average_discharges.iloc[:, 2].tolist()
    lenggries = average_discharges.iloc[:, 3].tolist()

    data = {'Kt': kempton, 'Dt': dillingen, 'Lt': lenggries}
    df = pd.DataFrame(data, columns=['Kt', 'Dt', 'Lt'])
    df = df.apply(normalize)

    return df


def load_climate_data():
    # Load river discharges data
    
    df = pd.read_csv(r'../datasets/environment_dataset/light.txt', sep=" ", header=None)
    df.columns = ["NEP", "PPFD"]
    df = df.apply(normalize)

    return df


def load_geo_data():
    # Load river discharges data
    path = r'../datasets/geo_dataset/xmoxa_data_H.csv'
    # vars = ['DateTime', 'rain', 'temperature_outside', 'pressure_outside', 'gw_mb',
    #    'gw_sr', 'gw_sg', 'gw_west', 'gw_knee', 'gw_south', 'wind_x', 'winx_y',
    #    'snow_load', 'humidity', 'glob_radiaton', 'strain_ew_uncorrected',
    #    'strain_ns_uncorrected', 'strain_ew_corrected', 'strain_ns_corrected',
    #    'tides_ew', 'tides_ns']
    # vars = ['DateTime', 'gwl_mb', 'gwl_sr', 'gwl_knee', 'gwl_south', 'strain_ew_corrected', 'strain_ns_corrected']
    vars = ['DateTime', 'temperature_outside', 'pressure_outside', 'strain_ns_corrected']
    data = pd.read_csv(path, usecols=vars)
    
    start_date = '2015-04-26'
    end_date = '2015-07-19'
    data = data.fillna(method='pad')
    data = data[(data['DateTime'] >= start_date) & (data['DateTime'] <= end_date)][vars]
    data = data.set_index('DateTime')
    data = data.apply(normalize)
    data.rename(columns={'temperature_outside': 'T', 'pressure_outside': 'P', 'snow_load': 'Snow$_{load}$', 'glob_radiaton': 'R$_g$', 'strain_ns_corrected':'Strain$_{ns}$',  'strain_ew_corrected':'Strain$_{ew}$', 'gwl_mb':'GW$_{mb}$', 'gwl_knee':'GW$_{knee}$', 'gwl_south':'GW$_{south}$', 'gwl_west':'GW$_{west}$', 'gwl_sg':'GW$_{sg}$', 'gwl_sr':'GW$_{sr}$', 'wind_x': 'Wind$_{ew}$', 'wind_y': 'Wind$_{ns}$'}, inplace=True)

    return data


def load_hackathon_data():
    # Load health data
    bot, bov = simple_load_csv(r"../datasets/hackathon_data/blood-oxygenation_interpolated_3600_pt_avg_14.csv")
    wt, wv = simple_load_csv(r"../datasets/hackathon_data/weight_interpolated_3600_pt_avg_6.csv")
    hrt, hrv = simple_load_csv(r"../datasets/hackathon_data/resting-heart-rate_interpolated_3600_iv_avg_4.csv")
    st, sv = simple_load_csv(r"../datasets/hackathon_data/step-amount_interpolated_3600_iv_ct_15.csv")
    it, iv = simple_load_csv(r"../datasets/hackathon_data/in-bed_interpolated_3600_iv_sp_19.csv")
    at, av = simple_load_csv(r"../datasets/hackathon_data/awake_interpolated_3600_iv_sp_18.csv")

        # plt.plot(bov)
        # plt.plot(wv)
        # plt.plot(hrv)
        # plt.show()

        # v15 = np.nan_to_num(aggregate_avg(ts_15, v_15, 60 * 60))
        # v3 = np.nan_to_num(aggregate_avg(ts_3, v_3, 60 * 60))
        # v2 = np.nan_to_num(aggregate_avg(ts_2, v_2, 60 * 60))
        # v1 = np.nan_to_num(aggregate_avg(ts_1, v_1, 60 * 60))

    data = {'BO': bov[7500:10000], 'WV': wv[7500:10000], 'HR': hrv[7500:10000], 'Step': sv[7500:10000], 'IB': iv[7500:10000], 'Awake': av[7500:10000]}
    df = pd.DataFrame(data, columns=['BO', 'WV', 'HR', 'Step', 'IB', 'Awake'])

    return df


def load_flux_data():

    # Load fluxnet 2015 data for grassland IT-Mbo site
    fluxnet = pd.read_csv(r"../datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
    org = fluxnet['SW_IN_F']
    otemp = fluxnet['TA_F']
    ovpd = fluxnet['VPD_F']
    # oppt = fluxnet['P_F']
    # nee = fluxnet['NEE_VUT_50']
    ogpp = fluxnet['GPP_NT_VUT_50']
    oreco = fluxnet['RECO_NT_VUT_50']
    
    # ------------------- Load FLUXNET2015 data --------------------
    rg = normalize(down_sample(org, win_size))
    temp = normalize(down_sample(otemp, win_size))
    # gpp = normalize(down_sample(nee, win_size, partition='gpp'))
    # reco = normalize(down_sample(nee, win_size, partition='reco'))
    gpp = normalize(down_sample(ogpp, win_size))
    reco = normalize(down_sample(oreco, win_size))
    # ppt = normalize(down_sample(oppt, win_size))
    vpd = normalize(down_sample(ovpd, win_size))
    # swc = normalize(down_sample(oswc, win_size))
    # heat = normalize(down_sample(oheat, win_size))

    data = {'Rg': rg[8000:12000], 'T': temp[8000:12000], 'GPP': gpp[8000:12000], 'Reco': reco[8000:12000]}
    df = pd.DataFrame(data, columns=['Rg', 'T', 'GPP', 'Reco'])

    return df


def load_syn_data():
    #****************** Load synthetic data *************************
    data = pd.read_csv(r"../datasets/synthetic_datasets/synthetic_data.csv")
    df = data.apply(normalize)
    return df


def load_multiregime_data():
    # *******************Load synthetic data *************************
    df = pd.read_csv(r"../datasets/synthetic_datasets/synthetic_data_regimes.csv")
    # df = df.apply(normalize)
    return df

def corr_heatmap(df):

    # Assuming you have a DataFrame named df
    # df = pd.DataFrame(...)

    # Compute the correlation matrix
    corr_matrix = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, center=0, square=True, linewidths=0.5)
    print(f'Correlation Matrix: {corr_matrix}')
    plt.xticks(ticks=np.arange(0.5, len(df.columns)), labels=df.columns, rotation=25, ha='right')
    plt.yticks(ticks=np.arange(0.5, len(df.columns)), labels=df.columns, rotation=0)
    plt.title('Correlation Heatmap')
    plt.show()

def causal_heatmap(cmatrix, columns):

    plt.figure(figsize=(10, 8))
    sns.heatmap(cmatrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, center=0, square=True, linewidths=0.5)
    plt.xticks(ticks=np.arange(0.5, len(columns)), labels=columns, rotation=25, ha='right')
    plt.yticks(ticks=np.arange(0.5, len(columns)), labels=columns, rotation=0)
    plt.title('Discovered Causal Structure')
    plot_path = r"../deepCausality/plots/"
    filename = pathlib.Path(plot_path + f"causal_matrix.pdf")
    plt.savefig(filename)

    # Show the plot
    plt.show()
