import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt


class RiverData():

    def __init__(self):

        self.abc = 0

    def get_data(self):

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

        return average_discharges

        # Look at the data
        # print(average_discharges.head())

        # fig = plt.figure()
        # ax1 = fig.add_subplot(311)
        # ax1.plot(average_discharges['Dillingen'])
        # ax1.set_ylabel(stations[0])
        #
        # ax2 = fig.add_subplot(312)
        # ax2.plot(average_discharges["Kempten"])
        # ax2.set_ylabel(stations[1])
        #
        # ax3 = fig.add_subplot(313)
        # ax3.plot(average_discharges["Lenggries"])
        # ax3.set_ylabel(stations[2])
        #
        # plt.show()

