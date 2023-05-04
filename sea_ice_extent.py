############################################################################################################################
############################################################################################################################
# Arctic Sea Ice Extent Analysis
# Author : Amaury Laridon
# Course : LPHYS2268 -Forecasts, predictions and projections in Climate Science
# Goal : Loading of the time series of September sea ice extent and seasonal prediction analysis
#        More information on the GitHub Page of the project : https://github.com/AugustinLambotte/LPHYS2268
# Date : 11/04/23
############################################################################################################################
#################################################### Packages ##############################################################
import numpy as np
import xarray as xr
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.stats import norm
import math

# from scipy.stats.stats import pearsonr

################################################### Parameters #############################################################
### Script parameters ###
year_0 = 1979
year_f = 2022
N_years = year_f - year_0
event_start_year_analysis = year_0 + 2
year_f_forecast = 2022


### Display parameters ###
plt.rcParams["text.usetex"] = True
figure = plt.figure(figsize=(16, 10))

### Data parameters ###
data_dir = "/home/amaury/Bureau/LPHYS2268 - Forecast prediction and projection in Climate Science/Projet Perso/Data/osisaf_nh_sie_monthly.nc"
save_dir = "/home/amaury/Bureau/LPHYS2268 - Forecast prediction and projection in Climate Science/Projet Perso/Figures/"

##################################################### Data Plotting #########################################################
# Correspond to question 3 of the report
##### Extraction of the Data #####


def extract_data(month, file=data_dir, dtype="dict"):
    """Extract september Sea Ice extent (sie) from 1979 to 2022. can return the data in two different parts depending on dtype parameter:
    dtype = "dict" return a dictionnary {'1979': 7.556, '1980': 8.244,...}
    dtype = "array" return a np.array [7.556, 8.244, ...]
    """
    data_set = xr.open_dataset(file)
    if dtype == "dict":
        month_sie = {}
        for year in range(year_0, year_f):
            month_sie[f"{year}"] = float(
                data_set["sie"].sel(time=datetime(year, month, 16))
            )

    elif dtype == "array":
        month_sie = []
        for year in range(year_0, year_f):
            month_sie.append(float(data_set["sie"].sel(time=datetime(year, month, 16))))
        month_sie = np.array(month_sie)
    else:
        print("! Incorrect data type !")
        data_set.close()
        return 0
    data_set.close()

    time_range = np.arange(year_0, year_f)

    return month_sie, time_range


##### Display #####


def plot_sie(data):
    """Plot of the SIE of the given month"""
    plt.plot(time_range, data, linewidth=4)
    plt.title("Sea Ice Extent (SIE) in September", size=30)
    plt.xlabel("Year", size=25)
    plt.xticks(np.arange(year_0, year_f, 2))
    plt.ylabel(r"Sea Ice Extent [$10^6$ km²]", size=25)
    plt.tick_params(axis="both", labelsize=20)
    plt.grid()
    plt.savefig(save_dir + "Sept_SIE_Data/sept_sie.png", dpi=300)
    # plt.show()
    plt.clf()


#############################################################################################################################
##################################################### Data Analysis #########################################################
#############################################################################################################################

############################################### 1 - Estimating the trend line ###############################################
# Correspond to question 4 of the report


def trend_line(time_range, data):
    """Computation of the coefficient of the trend line following definition in the instructions.
    Returns an array with coefficient a and b."""
    num = 0
    denum = 0
    mean_data = np.mean(ma.masked_invalid(data))
    mean_year = np.mean(time_range)
    for year in range(len(time_range)):
        num += (time_range[year] - mean_year) * (data[year] - mean_data)
        denum += (time_range[year] - mean_year) * (time_range[year] - mean_year)
    a = num / denum
    b = mean_data - (a * mean_year)
    return [a, b]


##### Display #####


def plot_trend_line_forecast(trend_line):
    """Plot of the SIE of september and the trend line"""
    plt.plot(time_range, sept_sie, label="Data", linewidth=4)
    plt.plot(time_range, trend_line, label="Trend line", linewidth=4)
    plt.title("Sea Ice Extent (SIE) in September", size=30)
    plt.xlabel("Year", size=25)
    plt.xticks(np.arange(year_0, year_f, 2))
    plt.ylabel(r"Sea Ice Extent [$10^6$ km²]", size=25)
    plt.tick_params(axis="both", labelsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    plt.savefig(save_dir + "Sept_SIE_Data/sept_sie_trend_line.png", dpi=300)
    # plt.show()
    plt.clf()


################################################### 2 - Frequency of event ###################################################
# Correspond to question 5 of the report


def event_freq(event_mode, data, time_range_data):
    """Computes the event frequency following the definition of the 'event' choosen. Returns an array
    with the frequency for each year of that event.
    event_mode = 1 corresponds to the event "September SIE will be below trend line"
    event_mode = 2 corresponds to the event "September SIE will be less than 4.5 million km²"
    event_mode = 3 corresponds to the event "September SIE will be less than previous year"
    Computes also the cheap and trivial climatological forecast and returns it for each year as an array
    """

    event_freq = np.zeros(year_f - event_start_year_analysis)
    clim_forecast = np.zeros(year_f - event_start_year_analysis)

    if event_mode == 1:
        for year in range(len(event_freq)):
            # data_used = np.zeros(year - event_start_year_analysis)
            data_used = data[0 : year + 2]
            year_used = time_range_data[0 : year + 2]
            n_years = len(year_used)
            ## event freq computation ##
            trend_line_coef = trend_line(year_used, data_used)
            forecast = (trend_line_coef[0] * year_used[year]) + trend_line_coef[1]
            if data_used[year] > forecast:
                event_freq[year] = 0
            else:
                event_freq[year] = 1
            ## clim forecast computation ##
            clim_frcst_val = np.sum(event_freq[0 : year - 1]) / n_years
            clim_forecast[year] = clim_frcst_val

    elif event_mode == 2:
        critical_sie = 4.5

        for year in range(len(event_freq)):
            # data_used = np.zeros(year - event_start_year_analysis)
            data_used = data[0 : year + 2]
            year_used = time_range_data[0 : year + 2]
            n_years = len(year_used)
            ## event freq computation ##
            if data_used[year] > critical_sie:
                event_freq[year] = 0
            else:
                event_freq[year] = 1
            ## clim forecast computation ##
            clim_frcst_val = np.sum(event_freq[0 : year - 1]) / n_years
            clim_forecast[year] = clim_frcst_val

    elif event_mode == 3:
        for year in range(len(event_freq)):
            # data_used = np.zeros(year - event_start_year_analysis)
            data_used = data[0 : year + 2]
            year_used = time_range_data[0 : year + 2]
            n_years = len(year_used)
            ## event freq computation ##
            if data_used[year] > data_used[year - 1]:
                event_freq[year] = 0
            else:
                event_freq[year] = 1
            ## clim forecast computation ##
            clim_frcst_val = np.sum(event_freq[0 : year - 1]) / n_years
            clim_forecast[year] = clim_frcst_val

    return event_freq, clim_forecast


##### Display #####


def plot_event_freq(data, time_range_data):
    """Plot of the three event frequency for the three different event_mode"""
    for i in range(3):
        plt.bar(time_range_data[2:], data[i, :])
        plt.title(
            "Arctic Sea Ice Evolution\nEvent mode {} Frequency".format(i + 1), size=30
        )
        plt.xlabel("Year", size=25)
        plt.xticks(np.arange(year_0 + 2, year_f, 2))
        plt.ylabel("Frequency", size=25)
        plt.tick_params(axis="both", labelsize=20)
        plt.grid()
        plt.savefig(save_dir + "Event/event_mode" + str(i + 1) + "_freq.png", dpi=300)
        # plt.show()
        plt.clf()


############################################## 3 - Statistical Forecasting System ############################################


##################################### Anomaly Persistent Forecast (APF) ########################################
# Corresponds to section 7 of the instruction report
def APF(year_f_forecast):
    """Computes a forecast for the SIE of a given month based on the Anomaly Persistent Forecast (APF) method.
    For more details read instructions. A mask is used because some data are missing (NaN).
    """
    # array that will collect the values of the mean SIE month forecast
    mean_sept_forecast = np.zeros(year_f_forecast - event_start_year_analysis)
    # array that will collect the values of the variance SIE month forecast
    var_sept_forecast = np.zeros(year_f_forecast - event_start_year_analysis)
    # array that will collect the values of the standard deviation SIE month forecast
    std_sept_forecast = np.zeros(year_f_forecast - event_start_year_analysis)
    # array collecting the years where a forecast is perform. Will be used for plots
    time_range_forecast = np.arange(event_start_year_analysis, year_f_forecast)
    ### Data Retrival ###
    sept_sie, time_range_s = extract_data(dtype="array", month=9)
    may_sie, time_range_m = extract_data(dtype="array", month=5)

    # array collecting the observed September SIE data at disposal
    observed_sept_data = [0, 0]
    # array collecting the observed May SIE data at disposal
    observed_may_data = [0, 0]

    # initialisation with the data know from the first two years of records data
    for year in range(2):
        observed_may_data[year] = may_sie[year]
        observed_sept_data[year] = sept_sie[year]

    ### Computation loop ###
    for year in range(len(mean_sept_forecast)):
        ## Computes the september mean forecast based on the APF method ##
        mean_sept_frcst_val = np.mean(ma.masked_invalid(observed_sept_data)) + (
            may_sie[year + 2] - np.mean(ma.masked_invalid(observed_may_data))
        )
        ## Check if there is an aberant value
        if math.isnan(mean_sept_frcst_val):
            mean_sept_frcst_val = mean_sept_forecast[
                year - 1
            ]  # replace the aberant value with a weighted value from the previous computed.
        ## Computes the september variance forecast based on the APF method ##
        n_i = (event_start_year_analysis + year) - year_0
        # print("year = ", year + event_start_year_analysis, "n_i = ", n_i)
        var_sept_frcst_val = (1 / n_i) * (
            np.var(ma.masked_invalid(observed_sept_data))
            + np.var(ma.masked_invalid(observed_may_data))
        )

        ## Computes the september standard deviation forecast based on the APF method ##
        std_sept_frcst_val = np.sqrt(var_sept_frcst_val)

        ## Save the september forecast in the arrays ##
        mean_sept_forecast[year] = mean_sept_frcst_val
        var_sept_forecast[year] = var_sept_frcst_val
        std_sept_forecast[year] = std_sept_frcst_val

        ## New observational data available from previous year ##
        # Add the data know from observation from the previous year for May SIE
        observed_may_data.append(may_sie[year + 2])
        # Add the data know from observation from the previous year for September SIE
        observed_sept_data.append(sept_sie[year + 2])

    # ## Output ##
    # print("September Forecast Mean based on APF method : ", mean_sept_forecast)
    # print(
    #     "----------------------------------------------------------------------------------------------------------------------------------"
    # )
    # print("September Forecast Variance based on APF method : ", var_sept_forecast)
    # print(
    #     "----------------------------------------------------------------------------------------------------------------------------------"
    # )

    return [
        mean_sept_forecast,
        var_sept_forecast,
        std_sept_forecast,
        observed_sept_data,
        time_range_forecast,
    ]


##################################### Correlated Anomaly Persistent Forecast (CAPF) ########################################


def CAPF(year_f_forecast):
    """Computes a forecast for the SIE of a given month based on the Correlated Anomaly Persistent Forecast (CAPF) method.
    The method is based on the APF except that the anomalyis multiplied by the correlation coefficient between the values
    of May and September"""
    # array that will collect the values of the mean SIE month forecast
    mean_sept_forecast = np.zeros(year_f_forecast - event_start_year_analysis)
    # array that will collect the values of the variance SIE month forecast
    var_sept_forecast = np.zeros(year_f_forecast - event_start_year_analysis)
    # array that will collect the values of the standard deviation SIE month forecast
    std_sept_forecast = np.zeros(year_f_forecast - event_start_year_analysis)
    # array collecting the years where a forecast is perform. Will be used for plots
    time_range_forecast = np.arange(event_start_year_analysis, year_f_forecast)
    # array collecting the correlation coefficient between May and September for each year of the forecast
    corr_ar = []
    ### Data Retrival ###
    sept_sie, time_range_s = extract_data(dtype="array", month=9)
    may_sie, time_range_m = extract_data(dtype="array", month=5)

    # array collecting the observed September SIE data at disposal
    observed_sept_data = [0, 0]
    # array collecting the observed May SIE data at disposal
    observed_may_data = [0, 0]

    # initialisation with the data know from the first two years of records data
    for year in range(2):
        observed_may_data[year] = may_sie[year]
        observed_sept_data[year] = sept_sie[year]

    ### Computation loop ###
    for year in range(len(mean_sept_forecast)):
        ## Computes the correlation coefficient between the SIE value of May and September for each year based on the observed data known at that time
        corr = np.corrcoef(
            ma.masked_invalid(observed_may_data), ma.masked_invalid(observed_sept_data)
        )[0, 1]
        corr_ar.append(corr)
        ## Computes the september mean forecast based on the APF method ##
        mean_sept_frcst_val = np.mean(ma.masked_invalid(observed_sept_data)) + (
            (may_sie[year + 2] - np.mean(ma.masked_invalid(observed_may_data))) * corr
        )
        ## Check if there is an aberant value
        if math.isnan(mean_sept_frcst_val):
            mean_sept_frcst_val = mean_sept_forecast[
                year - 1
            ]  # replace the aberant value with a weighted value from the previous computed.
        ## Computes the september variance forecast based on the APF method ##
        n_i = (event_start_year_analysis + year) - year_0
        # print("year = ", year + event_start_year_analysis, "n_i = ", n_i)
        var_sept_frcst_val = (1 / n_i) * (
            np.var(ma.masked_invalid(observed_sept_data))
            + np.var(ma.masked_invalid(observed_may_data))
        )

        ## Computes the september standard deviation forecast based on the APF method ##
        std_sept_frcst_val = np.sqrt(var_sept_frcst_val)

        ## Save the september forecast in the arrays ##
        mean_sept_forecast[year] = mean_sept_frcst_val
        var_sept_forecast[year] = var_sept_frcst_val
        std_sept_forecast[year] = std_sept_frcst_val

        ## New observational data available from previous year ##
        # Test if there is NaN values in the May or September SIE value.
        # If so we decide arbitrarily to define this value to the one of the previous year.
        # We can't let a NaN value in the array otherwise the correlation computation will fail
        if math.isnan(may_sie[year + 2]):
            may_sie[year + 2] = may_sie[year + 1]
        if math.isnan(sept_sie[year + 2]):
            sept_sie[year + 2] = sept_sie[year + 1]
        # Add the data know from observation from the previous year for May SIE
        observed_may_data.append(may_sie[year + 2])
        # Add the data know from observation from the previous year for September SIE
        observed_sept_data.append(sept_sie[year + 2])

    # ## Output ##
    # print("September Forecast Mean based on APF method : ", mean_sept_forecast)
    # print(
    #     "----------------------------------------------------------------------------------------------------------------------------------"
    # )
    # print("September Forecast Variance based on APF method : ", var_sept_forecast)
    # print(
    #     "----------------------------------------------------------------------------------------------------------------------------------"
    # )

    return [
        mean_sept_forecast,
        var_sept_forecast,
        std_sept_forecast,
        observed_sept_data,
        time_range_forecast,
    ]


### Display ###


def plot_forecast1(forecast, year_f_forecast, trend_line_opt, method):
    """Plot of the forecast of SIE for September using APF or CAPF method and the observed data."""

    plt.errorbar(
        forecast[4],
        forecast[0],
        2 * forecast[2],
        linestyle="None",
        marker="^",
        label=method + " Forecast + 2 std",
        color="orange",
    )
    plt.plot(forecast[4], forecast[3][2:], label="Observed Data", color="g")

    if trend_line_opt == True:
        ## Trend line of the Forecast ##
        trend_line_frcst_coef = trend_line(forecast[4], forecast[0])
        frcst_trend_line = [
            trend_line_frcst_coef[0] * year + trend_line_frcst_coef[1]
            for year in forecast[4]
        ]  # array collecting all the values associated to the trend line forecast
        mod_save_dir = "_trd_line"
        plt.plot(
            forecast[4], frcst_trend_line, label="Forecast Trend Line", color="orange"
        )
        ## Trend line of the Observation ##
        trend_line_data_coef = trend_line(forecast[4], forecast[3][2:])
        data_trend_line = [
            trend_line_data_coef[0] * year + trend_line_data_coef[1]
            for year in forecast[4]
        ]  # array collecting all the values associated to the trend line forecast
        mod_save_dir = "_trd_line"
        plt.plot(
            forecast[4], data_trend_line, label="Observation Trend Line", color="g"
        )
    else:
        mod_save_dir = ""
    plt.title("Arctic Sea Ice Extent\nSeptember SIE " + method + "-Forecast ", size=30)
    plt.xlabel("Year", size=25)
    plt.xticks(np.arange(year_0 + 2, year_f_forecast, 2))
    plt.ylabel(r"SIE [$10^6$ km²]", size=25)
    plt.tick_params(axis="both", labelsize=20)
    plt.grid()
    plt.legend(fontsize=20)
    plt.savefig(
        save_dir + "Sept_SIE_Forecast/SIE_Sept_" + method + "_Forecast" + mod_save_dir,
        dpi=300,
    )
    # plt.show()
    plt.clf()


def plot_forecast2(
    forecast1, forecast2, year_f_forecast, trend_line_opt, method1, method2
):
    """Plot of the forecast of SIE for September using APF or CAPF method and the observed data."""

    plt.errorbar(
        forecast1[4],
        forecast1[0],
        2 * forecast1[2],
        linestyle="None",
        marker="^",
        label=method1 + " Forecast + 2 std",
        color="orange",
    )
    plt.errorbar(
        forecast2[4],
        forecast2[0],
        2 * forecast2[2],
        linestyle="None",
        marker="^",
        label=method2 + " Forecast + 2 std",
        color="tab:blue",
    )
    plt.plot(forecast1[4], forecast1[3][2:], label="Observed Data", color="g")

    if trend_line_opt == True:
        ## Trend line of the First Forecast ##
        trend_line_frcst_coef = trend_line(forecast1[4], forecast1[0])
        frcst_trend_line = [
            trend_line_frcst_coef[0] * year + trend_line_frcst_coef[1]
            for year in forecast1[4]
        ]  # array collecting all the values associated to the trend line forecast
        mod_save_dir = "_trd_line"
        plt.plot(
            forecast1[4],
            frcst_trend_line,
            label=method1 + " Trend Line",
            color="orange",
        )
        ## Trend line of the Second Forecast ##
        trend_line_frcst2_coef = trend_line(forecast2[4], forecast2[0])
        frcst_trend_line2 = [
            trend_line_frcst2_coef[0] * year + trend_line_frcst2_coef[1]
            for year in forecast2[4]
        ]  # array collecting all the values associated to the trend line forecast
        mod_save_dir = "_trd_line"
        plt.plot(
            forecast2[4],
            frcst_trend_line2,
            label=method2 + " Trend Line",
            color="tab:blue",
        )

        ## Trend line of the observation ##

        trend_line_data_coef = trend_line(forecast1[4], forecast1[3][2:])
        data_trend_line = [
            trend_line_data_coef[0] * year + trend_line_data_coef[1]
            for year in forecast1[4]
        ]  # array collecting all the values associated to the trend line forecast
        mod_save_dir = "_trd_line"
        plt.plot(
            forecast1[4], data_trend_line, label="Observation Trend Line", color="g"
        )

    else:
        mod_save_dir = ""
    plt.title(
        "Arctic Sea Ice Extent\nSeptember SIE "
        + method1
        + "/"
        + method2
        + "-Forecast ",
        size=30,
    )
    plt.xlabel("Year", size=25)
    plt.xticks(np.arange(year_0 + 2, year_f_forecast, 2))
    plt.ylabel(r"SIE [$10^6$ km²]", size=25)
    plt.tick_params(axis="both", labelsize=20)
    plt.grid()
    plt.legend(fontsize=20)
    plt.savefig(
        save_dir
        + "Sept_SIE_Forecast/SIE_Sept_"
        + method1
        + "&"
        + method2
        + "_Forecast"
        + mod_save_dir,
        dpi=300,
    )
    # plt.show()
    plt.clf()


##################################### 4 - Retrospective probabilistic forecast of the event ###################################


def prob_frcst_event(event_mode, data, time_range_data):
    """Computes the probability of a forecast to give an event following the definition of the 'event' choosen. Returns an array
    with the probability that the event chosen happens accordind to the forecast."""

    event_prob = np.zeros(
        year_f - event_start_year_analysis
    )  # array that will store the event probability for a given year
    mean_forecast = data[0]  # assigning the mean_forecast array
    std_forecast = data[2]  # assigning the std_forecast array
    sept_sie, time_range_s = extract_data(
        dtype="array", month=9
    )  # extraction of the full observed data

    if event_mode == 1:
        for year in range(len(event_prob)):
            data_used = sept_sie[0 : year + 2]
            year_used = time_range_s[0 : year + 2]
            trend_line_coef = trend_line(year_used, data_used)
            trend_line_forecast = (
                trend_line_coef[0] * year_used[year]
            ) + trend_line_coef[1]
            proba = norm.cdf(
                (trend_line_forecast - mean_forecast[year]) / std_forecast[year]
            )
            event_prob[year] = proba

    elif event_mode == 2:
        critical_sie = 4.5
        for year in range(len(event_prob)):
            proba = norm.cdf((critical_sie - mean_forecast[year]) / std_forecast[year])
            event_prob[year] = proba

    elif event_mode == 3:
        for year in range(len(event_prob)):
            # data_used = np.zeros(year - event_start_year_analysis)
            data_used = sept_sie[0 : year + 2]
            year_used = time_range_s[0 : year + 2]
            proba = norm.cdf(
                (data_used[year + 1] - mean_forecast[year]) / std_forecast[year]
            )
            event_prob[year] = proba

    return event_prob


##### Display #####


def plot_event_frcst_prob(data, time_range_data, method):
    """Plot of the event probability forecast following the definition which has been choose"""
    for i in range(3):
        plt.bar(time_range_data, data[i, :])
        plt.title(
            "Arctic Sea Ice Evolution\nRetrospective probabilistic "
            + method
            + " forecast of the Event mode {}".format(i + 1),
            size=30,
        )
        plt.xlabel("Year", size=25)
        plt.xticks(np.arange(year_0 + 2, year_f, 2))
        plt.ylabel("Probability", size=25)
        plt.tick_params(axis="both", labelsize=20)
        plt.grid()
        plt.savefig(
            save_dir + "Event/event_mode" + str(i + 1) + "_" + method + "_prob.png",
            dpi=300,
        )
        # plt.show()
        plt.clf()


def plot_event_freq_and_prob(time_range, data_freq, data_prob, method):
    """Plot of the event frequency and the probability of the event using the forecast"""
    fig, axs = plt.subplots(3)
    fig.suptitle(
        "Probability of the event according to the "
        + method
        + " forecast and observed frequencies"
    )

    axs[0].scatter(time_range, data_freq[0, :], color="g")
    axs[0].bar(time_range, data_prob[0, :], width=0.5)
    axs[0].set_xlabel("Year")
    axs[0].set_xticks(np.arange(year_0 + 2, year_f, 4))
    axs[0].set_ylabel("Probability")
    axs[0].grid()
    axs[0].set_title("Event : SIE below trend line")

    axs[1].scatter(time_range, data_freq[1, :], color="g")
    axs[1].bar(time_range, data_prob[1, :], width=0.5)
    axs[1].set_xlabel("Year")
    axs[1].set_xticks(np.arange(year_0 + 2, year_f, 4))
    axs[1].set_ylabel("Probability")
    axs[1].set_title("Event : SIE less than 4.5 million km²")
    axs[1].grid()

    axs[2].scatter(time_range, data_freq[2, :], label="Observed Frequency", color="g")
    axs[2].bar(
        time_range, data_prob[2, :], width=0.5, label="Probability from Forecast"
    )
    axs[2].set_xlabel("Year")
    axs[2].set_xticks(np.arange(year_0 + 2, year_f, 4))
    axs[2].set_ylabel("Probability")
    axs[2].set_title("Event : SIE less than previous year")
    axs[2].grid()

    # fig.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(
        save_dir + "Event/prob_event_" + method + "_event_freq_obs_subplots.png",
        dpi=300,
    )
    # plt.show()
    plt.clf()


def plot_event_freq_and_prob_and_clim(
    time_range, data_freq, data_prob, clim_forecast, method
):
    """Plot of the event frequency and the probability of the event and the climatological forecast probability using the forecast"""
    fig, axs = plt.subplots(3)
    fig.suptitle(
        "Probability of the event according to the "
        + method
        + " forecast and observed frequencies"
    )

    axs[0].scatter(time_range, data_freq[0, :], color="g")
    axs[0].scatter(time_range, clim_forecast[0, :], color="r")
    axs[0].bar(time_range, data_prob[0, :], width=0.5)
    axs[0].set_xlabel("Year")
    axs[0].set_xticks(np.arange(year_0 + 2, year_f, 4))
    axs[0].set_ylabel("Probability")
    axs[0].grid()
    axs[0].set_title("Event : SIE below trend line")

    axs[1].scatter(time_range, data_freq[1, :], color="g")
    axs[1].scatter(time_range, clim_forecast[1, :], color="r")
    axs[1].bar(time_range, data_prob[1, :], width=0.5)
    axs[1].set_xlabel("Year")
    axs[1].set_xticks(np.arange(year_0 + 2, year_f, 4))
    axs[1].set_ylabel("Probability")
    axs[1].set_title("Event : SIE less than 4.5 million km²")
    axs[1].grid()

    axs[2].scatter(time_range, data_freq[2, :], label="Observed Frequency", color="g")
    axs[2].bar(
        time_range, data_prob[2, :], width=0.5, label="Probability from Forecast"
    )
    axs[2].scatter(time_range, clim_forecast[2, :], color="r")
    axs[2].set_xlabel("Year")
    axs[2].set_xticks(np.arange(year_0 + 2, year_f, 4))
    axs[2].set_ylabel("Probability")
    axs[2].set_title("Event : SIE less than previous year")
    axs[2].grid()

    # fig.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(
        save_dir + "Event/prob_event_" + method + "_event_freq_obs_subplots.png",
        dpi=300,
    )
    # plt.show()
    plt.clf()


def plot_event_freq_and_prob_and_clim_no_sub(
    time_range, data_freq, data_prob, clim_forecast, method
):
    """Plot of the event frequency and the probability of the event using the forecast"""
    for i in range(3):
        plt.title(
            "Arctic Sea Ice Evolution\nRetrospective probabilistic "
            + method
            + " forecast of the Event mode {}".format(i + 1),
            size=18,
        )
        plt.scatter(
            time_range, data_freq[i, :], color="g", label="Observed data frequency"
        )
        plt.scatter(
            time_range, clim_forecast[i, :], color="r", label="Climatological Forecast"
        )
        plt.bar(time_range, data_prob[i, :], width=0.5, label=method + " forecast")
        plt.xlabel("Year", size=12)
        # plt.xticks(np.arange(year_0 + 2, year_f, 2))
        plt.ylabel("Probability", size=12)
        plt.tick_params(axis="both", labelsize=10)
        plt.legend()
        plt.grid()
        plt.savefig(
            save_dir
            + "Event/event_mode"
            + str(i + 1)
            + "_"
            + method
            + "_prob_and_clim.png",
            dpi=300,
        )
        # plt.show()
        plt.clf()


########################################### 5 - Verification of retrospective forecast #######################################


def BS_ref(event_freq):
    """Computes de the reference Brier Score"""
    som = np.sum(event_freq)
    n = len(event_freq)
    p_c = som / n
    bs_ref = p_c * (1 - p_c)

    return bs_ref


def BS_and_BSS(time_range, data_freq, data_prob, event_mode):
    """Computes the Brier Score and Brier Skill Score for a given event mode definition"""
    n = len(time_range)
    bs = (1 / n) * (
        np.sum(
            [
                (data_prob[event_mode, i] - data_freq[event_mode, i]) ** 2
                for i in range(len(data_prob))
            ]
        )
    )
    # print(
    #     [
    #         (data_prob[event_mode, i] - data_freq[event_mode, i]) ** 2
    #         for i in range(len(data_prob))
    #     ]
    # )
    bs_ref = BS_ref(data_freq[event_mode, :])
    bss = (bs - bs_ref) / (0 - bs_ref)
    return bs, bs_ref, bss


########################################### 6 - Post processing of forecast #######################################

### Computations ###


def mean_bias(data_obs, data_frcst):
    """Computes the biais of the mean and return the Post-Process array"""
    for_min_ver = data_frcst - data_obs
    mb = np.mean(for_min_ver)
    new_data_frcst = data_frcst - mb
    return new_data_frcst


def var_bias(data_obs, data_frcst):
    """Computes the Biais of the variability and return the Post-Process array"""
    new_data_frcst = (data_frcst - np.mean(data_frcst)) / np.std(data_frcst) * np.std(
        data_obs
    ) + np.mean(data_frcst)

    return new_data_frcst


def trend_bias(time_range, data_obs, data_frcst):
    """Computes the Biais of the trend and return the Post-Process array"""

    frcst_trend_line_coef = trend_line(time_range, data_frcst)
    obs_trend_line_coef = trend_line(time_range, data_obs)

    frcst_trend_line = [
        frcst_trend_line_coef[0] * year + frcst_trend_line_coef[1]
        for year in time_range
    ]
    obs_trend_line = [
        obs_trend_line_coef[0] * year + obs_trend_line_coef[1] for year in time_range
    ]

    new_data_frcst = data_frcst - frcst_trend_line + obs_trend_line

    return new_data_frcst


### Display ###


def data_plot1_pp(time_range, data_obs, data_frcst, method):
    """Plot of the scatter plot of observed data against forecast data and use of post-processing"""

    ### Scatter plot and y=x plot ###
    plt.scatter(data_frcst[0], data_obs, color="0.4", label="Raw " + method)
    plt.plot((0.0, 100), (0.0, 100), "b--", label="y=x")

    ### Post processing ###
    ## Bias of the mean ##
    new_bias_frcst_1 = mean_bias(data_obs, data_frcst[0])
    # Biais of the mean plot #
    """plt.scatter(
        new_bias_frcst_1, data_obs, color="orange", label="PP Bias of the mean " + method
    )"""
    ## Bias of the variability ##
    new_bias_frcst_2 = var_bias(data_obs, new_bias_frcst_1)
    # Biais of the mean plot #
    """plt.scatter(
        new_bias_frcst_2, data_obs, color="orange", label="PP Bias of mean + var " + method
    )"""
    ## Bias of the trend line ##
    new_bias_frcst_3 = trend_bias(time_range, data_obs, new_bias_frcst_2)
    # Biais of the mean plot #
    plt.scatter(
        new_bias_frcst_3,
        data_obs,
        color="orange",
        label="PP Bias of (mean + var + trend) " + method,
    )

    ### Cosmetics ###
    plt.title("Observed against Forecast Arctic Sea Ice Evolution", size=14)
    plt.xlabel(r"Forecast SIE $[10^6 km^2]$", size=10)
    plt.ylabel(r"Observed SIE $[10^6 km^2]$", size=10)
    plt.xlim(4.1, 8.5)
    plt.ylim(4, 8.5)
    plt.tick_params(axis="both", labelsize=10)
    plt.legend()
    plt.grid()
    plt.savefig(
        save_dir + "Post-Processing/data_plot_verif_Biais_Mean & Var & Trend_" + method,
        dpi=300,
    )
    # plt.show()
    plt.clf()


def data_plot2_pp(time_range, data_obs, data_frcst, method):
    """SubPlot of the scatter plot of observed data against forecast data and use of post-processing and the normal data plot against the years"""
    fig = plt.figure(figsize=(20, 10))

    ### Post processing ###
    ## Bias of the mean ##
    new_bias_frcst_1 = mean_bias(data_obs, data_frcst[0])
    ## Bias of the variability & mean ##
    new_bias_frcst_2 = var_bias(data_obs, new_bias_frcst_1)
    ## Bias of the trend line & variability & mean ##
    new_bias_frcst_3 = trend_bias(time_range, data_obs, new_bias_frcst_2)

    fig, axs = plt.subplots(1, 2)

    axs[0].scatter(data_frcst[0], data_obs, color="0.4", label="Raw " + method)
    axs[0].plot((0.0, 100), (0.0, 100), "b--", label="y=x")
    axs[0].scatter(
        new_bias_frcst_3,
        data_obs,
        color="orange",
        label="PP Bias of (mean + var + trend) " + method,
    )
    axs[0].set_xlabel(r"Forecast SIE $[10^6 km^2]$", size=10)
    axs[0].set_ylabel(r"Observed SIE $[10^6 km^2]$", size=10)
    axs[0].set_xlim(4.1, 8.5)
    axs[0].set_ylim(4, 8.5)
    axs[0].grid()
    axs[0].set_title("Observed against Forecast SIE")
    axs[0].legend()

    axs[1].plot(
        data_frcst[4],
        data_frcst[0],
        linestyle="--",
        label=method + " Forecast No PP",
        color="0.4",
    )
    axs[1].errorbar(
        data_frcst[4],
        new_bias_frcst_3,
        2 * data_frcst[2],
        linestyle="None",
        marker="^",
        label=method + " Forecast PP",
        color="orange",
    )
    axs[1].plot(data_frcst[4], data_frcst[3][2:], label="Observed Data", color="g")
    axs[1].set_xlabel("Year", size=10)
    # axs[1].set_xticks(np.arange(year_0 + 2, year_f_forecast, 5))
    axs[1].set_ylabel(r"SIE [$10^6$ km²]", size=10)
    axs[1].set_title("SIE " + method + "-Forecast ")
    axs[1].grid()
    axs[1].legend()

    # fig.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(save_dir + "Post-Processing/subplot_data_pp_" + method, dpi=300)
    # plt.show()
    plt.clf()


def data_plot3_pp(time_range, data_obs, data_frcst, method):
    """Plot of the post-process forecast, the non post-process forecast and the observations"""
    fig = plt.figure(figsize=(20, 10))

    ### Post processing ###
    ## Bias of the mean ##
    new_bias_frcst_1 = mean_bias(data_obs, data_frcst[0])
    ## Bias of the variability & mean ##
    new_bias_frcst_2 = var_bias(data_obs, new_bias_frcst_1)
    ## Bias of the trend line & variability & mean ##
    new_bias_frcst_3 = trend_bias(time_range, data_obs, new_bias_frcst_2)

    plt.plot(
        data_frcst[4],
        data_frcst[0],
        linestyle="--",
        label=method + " Forecast No PP",
        color="0.4",
    )
    plt.errorbar(
        data_frcst[4],
        new_bias_frcst_3,
        2 * data_frcst[2],
        linestyle="None",
        marker="^",
        label=method + " Forecast PP",
        color="orange",
    )
    plt.plot(data_frcst[4], data_frcst[3][2:], label="Observed Data", color="g")
    plt.xlabel("Year", size=20)
    # axs[1].set_xticks(np.arange(year_0 + 2, year_f_forecast, 5))
    plt.ylabel(r"SIE [$10^6$ km²]", size=20)
    plt.title("SIE " + method + "-Forecast", size=25)
    plt.grid()
    plt.legend(fontsize=18)

    # fig.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(save_dir + "Post-Processing/data_pp_" + method, dpi=300)
    # plt.show()
    plt.clf()


def data_plot4_pp(time_range, data_obs, data_frcst1, data_frcst2, method1, method2):
    """Plot of the two post-process forecast, the two non post-process forecast and the observations"""
    fig = plt.figure(figsize=(20, 10))

    ### Post processing ###
    ## Bias of the mean ##
    new_bias_frcst_1 = mean_bias(data_obs, data_frcst1[0])
    new_bias_frcst_12 = mean_bias(data_obs, data_frcst2[0])
    ## Bias of the variability & mean ##
    new_bias_frcst_2 = var_bias(data_obs, new_bias_frcst_1)
    new_bias_frcst_22 = var_bias(data_obs, new_bias_frcst_12)
    ## Bias of the trend line & variability & mean ##
    new_bias_frcst_3 = trend_bias(time_range, data_obs, new_bias_frcst_2)
    new_bias_frcst_32 = trend_bias(time_range, data_obs, new_bias_frcst_22)

    plt.plot(
        data_frcst1[4],
        data_frcst1[0],
        linestyle="--",
        label=method1 + " Forecast No PP",
        color="0.4",
    )
    plt.plot(
        data_frcst2[4],
        data_frcst2[0],
        linestyle="dotted",
        label=method2 + " Forecast No PP",
        color="0.4",
    )
    plt.errorbar(
        data_frcst1[4],
        new_bias_frcst_3,
        2 * data_frcst1[2],
        linestyle="None",
        marker="^",
        label=method1 + " Fortab:blueecast PP",
        color="orange",
    )
    plt.errorbar(
        data_frcst2[4],
        new_bias_frcst_32,
        2 * data_frcst2[2],
        linestyle="None",
        marker="^",
        label=method2 + " Forecast PP",
        color="tab:blue",
    )
    plt.plot(data_frcst1[4], data_frcst1[3][2:], label="Observed Data", color="g")
    plt.xlabel("Year", size=20)
    # axs[1].set_xticks(np.arange(year_0 + 2, year_f_forecast, 5))
    plt.ylabel(r"SIE [$10^6$ km²]", size=20)
    plt.title("SIE Forecast", size=25)
    plt.grid()
    plt.legend(fontsize=18)

    # fig.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(
        save_dir + "Post-Processing/data_pp_" + method1 + "&" + method2, dpi=300
    )
    # plt.show()
    plt.clf()


##############################################################################################################################
##################################################### Execution ##############################################################
##############################################################################################################################

if __name__ == "__main__":
    ##########################################################################################################
    ##### Loading and visualizing Sea Ice Extent Data #####
    sept_sie, time_range = extract_data(dtype="array", month=9)
    ### Display ###
    plot_sie(sept_sie)

    ##########################################################################################################
    ##### Estimating the trend line #####
    # computation of the trend line coefficients
    trend_line_coef = trend_line(time_range, sept_sie)
    # What is the September 2023 (X = 2023) forecast based on simple extrapolation of the trend?
    sept_23_forecast_trend_line = trend_line_coef[1] + trend_line_coef[0] * 2023
    print(
        "----------------------------------------------------------------------------------------------------------------------------------"
    )
    print(
        "September 2023 forecast using simple extrapolation trend : {:.4f}.10^6 km².".format(
            sept_23_forecast_trend_line
        )
    )
    print(
        "----------------------------------------------------------------------------------------------------------------------------------"
    )
    sept_forecast_trend_line = [
        trend_line_coef[0] * year + trend_line_coef[1] for year in time_range
    ]  # array collecting all the values associated to the trend line forecast
    ### Display ###
    plot_trend_line_forecast(trend_line=sept_forecast_trend_line)

    ##########################################################################################################
    ##### Event frequency based on data #####

    event_freq_mode = np.zeros((3, year_f - event_start_year_analysis))
    clim_forecast = np.zeros((3, year_f - event_start_year_analysis))
    for i in range(3):
        event_freq_mode[i, :], clim_forecast[i, :] = event_freq(
            event_mode=(i + 1), data=sept_sie, time_range_data=time_range
        )
    ### Display ###
    plot_event_freq(data=event_freq_mode, time_range_data=time_range)

    ##########################################################################################################
    ##### APF Forecasting system #####
    apf_forecast = APF(year_f_forecast=year_f_forecast)
    ### Display ###
    plot_forecast1(
        forecast=apf_forecast, year_f_forecast=2022, trend_line_opt=True, method="APF"
    )
    plot_forecast1(
        forecast=apf_forecast, year_f_forecast=2022, trend_line_opt=False, method="APF"
    )
    ##### CAPF Forecasting system #####
    capf_forecast = CAPF(year_f_forecast=year_f_forecast)
    ### Display ###
    plot_forecast1(
        forecast=capf_forecast, year_f_forecast=2022, trend_line_opt=True, method="CAPF"
    )
    plot_forecast1(
        forecast=capf_forecast,
        year_f_forecast=2022,
        trend_line_opt=False,
        method="CAPF",
    )
    ## Comparaison ##
    plot_forecast2(
        apf_forecast,
        capf_forecast,
        year_f_forecast=2022,
        trend_line_opt=True,
        method1="APF",
        method2="CAPF",
    )
    plot_forecast2(
        apf_forecast,
        capf_forecast,
        year_f_forecast=2022,
        trend_line_opt=False,
        method1="APF",
        method2="CAPF",
    )

    ##########################################################################################################
    ##### Restrospective probabilistic forecast of the event #####
    event_APF_prob_mode = np.zeros((3, year_f - event_start_year_analysis))
    event_CAPF_prob_mode = np.zeros((3, year_f - event_start_year_analysis))
    for i in range(3):
        event_APF_prob_mode[i, :] = prob_frcst_event(
            event_mode=(i + 1), data=apf_forecast, time_range_data=apf_forecast[-1]
        )
        event_CAPF_prob_mode[i, :] = prob_frcst_event(
            event_mode=(i + 1), data=capf_forecast, time_range_data=capf_forecast[-1]
        )
    ### Display ###
    plot_event_frcst_prob(
        data=event_APF_prob_mode, time_range_data=apf_forecast[-1], method="APF"
    )
    plot_event_frcst_prob(
        data=event_CAPF_prob_mode, time_range_data=capf_forecast[-1], method="CAPF"
    )
    plot_event_freq_and_prob(
        apf_forecast[-1], event_freq_mode, event_APF_prob_mode, method="APF"
    )
    plot_event_freq_and_prob(
        capf_forecast[-1], event_freq_mode, event_CAPF_prob_mode, method="CAPF"
    )
    plot_event_freq_and_prob_and_clim_no_sub(
        apf_forecast[-1],
        event_freq_mode,
        event_APF_prob_mode,
        clim_forecast,
        method="APF",
    )
    plot_event_freq_and_prob_and_clim_no_sub(
        capf_forecast[-1],
        event_freq_mode,
        event_CAPF_prob_mode,
        clim_forecast,
        method="CAPF",
    )
    ##########################################################################################################
    ##### Verification of retrospective forecast without Post Processing #####
    ### Brier Score computation ###
    bs_mode_APF = np.zeros(3)
    bs_mode_CAPF = np.zeros(3)
    bs_ref_mode_APF = np.zeros(3)
    bs_ref_mode_CAPF = np.zeros(3)
    bss_mode_APF = np.zeros(3)
    bss_mode_CAPF = np.zeros(3)
    name_event = [
        "September SIE will be below the trend line",
        "September SIE will be less than 4.5 million km²",
        "September SIE will be less than previous year",
    ]
    print(
        "                                    VERIFICATION NO POST-PROCESSING                                                 |"
    )
    print(
        "--------------------------------------------------------------------------------------------------------------------"
    )
    for i in range(3):
        bs_val_APF, bs_ref_val_APF, bss_val_APF = BS_and_BSS(
            apf_forecast[-1], event_freq_mode, event_APF_prob_mode, event_mode=i
        )
        bs_val_CAPF, bs_ref_val_CAPF, bss_val_CAPF = BS_and_BSS(
            capf_forecast[-1], event_freq_mode, event_CAPF_prob_mode, event_mode=i
        )
        bs_mode_APF[i] = bs_val_APF
        bs_mode_CAPF[i] = bs_val_CAPF
        bs_ref_mode_APF[i] = bs_ref_val_APF
        bs_ref_mode_CAPF[i] = bs_ref_val_CAPF
        bss_mode_APF[i] = bss_val_APF
        bss_mode_CAPF[i] = bss_val_CAPF
        print(
            "APF Brier Score (BS) for event : "
            + name_event[i]
            + " : {}".format(bs_val_APF)
        )
        print(
            "CAPF Brier Score (BS) for event : "
            + name_event[i]
            + " : {}".format(bs_val_CAPF)
        )
        print("")
        print(
            "APF Reference Brier Score (BS_ref) for event : "
            + name_event[i]
            + " : {}".format(bs_ref_val_APF)
        )
        print(
            "CAPF Reference Brier Score (BS_ref) for event : "
            + name_event[i]
            + " : {}".format(bs_ref_val_CAPF)
        )
        print("")
        print(
            "APF Brier Skill Score (BSS) for event : "
            + name_event[i]
            + " : {}".format(bss_val_APF)
        )
        print(
            "CAPF Brier Skill Score (BSS) for event : "
            + name_event[i]
            + " : {}".format(bss_val_CAPF)
        )
        print(
            "--------------------------------------------------------------------------------------------------------------------"
        )

    print(
        "----------------------------------------------------------------------------------------------------------------------------------"
    )
    ##########################################################################################################
    ##### Post Processing of retrospective forecast #####

    ### APF Forecast ###

    data_plot1_pp(
        time_range=apf_forecast[-1],
        data_obs=sept_sie[2:],
        data_frcst=apf_forecast,
        method="APF",
    )
    data_plot2_pp(
        time_range=apf_forecast[-1],
        data_obs=sept_sie[2:],
        data_frcst=apf_forecast,
        method="APF",
    )
    data_plot3_pp(
        time_range=apf_forecast[-1],
        data_obs=sept_sie[2:],
        data_frcst=apf_forecast,
        method="APF",
    )
    ### CAPF Forecast ###
    data_plot1_pp(
        time_range=capf_forecast[-1],
        data_obs=sept_sie[2:],
        data_frcst=capf_forecast,
        method="CAPF",
    )
    data_plot2_pp(
        time_range=capf_forecast[-1],
        data_obs=sept_sie[2:],
        data_frcst=capf_forecast,
        method="CAPF",
    )
    data_plot3_pp(
        time_range=capf_forecast[-1],
        data_obs=sept_sie[2:],
        data_frcst=capf_forecast,
        method="CAPF",
    )
    ## Two Forecast Comparaison ##
    data_plot4_pp(
        time_range=capf_forecast[-1],
        data_obs=sept_sie[2:],
        data_frcst1=apf_forecast,
        data_frcst2=capf_forecast,
        method1="APF",
        method2="CAPF",
    )
    ##########################################################################################################
    ##### Verification of retrospective forecast with Post Processing #####
    ### Post processing ###
    ## Bias of the mean ##
    new_bias_frcst_1_APF = mean_bias(apf_forecast[-1], apf_forecast[0])
    new_bias_frcst_1_CAPF = mean_bias(capf_forecast[-1], capf_forecast[0])
    ## Bias of the variability & mean ##
    new_bias_frcst_2_APF = var_bias(apf_forecast[-1], new_bias_frcst_1_APF)
    new_bias_frcst_2_CAPF = var_bias(capf_forecast[-1], new_bias_frcst_1_CAPF)
    ## Bias of the trend line & variability & mean ##
    pp_APF_forecast_val = trend_bias(
        apf_forecast[-1], apf_forecast[-1], new_bias_frcst_2_APF
    )
    pp_CAPF_forecast_val = trend_bias(
        capf_forecast[-1], capf_forecast[-1], new_bias_frcst_2_CAPF
    )

    # Defining the new set of forecast data #
    pp_APF_forecast = apf_forecast
    pp_CAPF_forecast = capf_forecast
    pp_APF_forecast[0] = pp_APF_forecast_val
    pp_CAPF_forecast[0] = pp_CAPF_forecast_val

    ### Computation of the event prob mode with the post-processed forecast ###
    event_APF_prob_mode_pp = np.zeros((3, year_f - event_start_year_analysis))
    event_CAPF_prob_mode_pp = np.zeros((3, year_f - event_start_year_analysis))
    for i in range(3):
        event_APF_prob_mode_pp[i, :] = prob_frcst_event(
            event_mode=(i + 1),
            data=pp_APF_forecast,
            time_range_data=pp_APF_forecast[-1],
        )
        event_CAPF_prob_mode[i, :] = prob_frcst_event(
            event_mode=(i + 1),
            data=pp_CAPF_forecast,
            time_range_data=pp_CAPF_forecast[-1],
        )
    ### Display Event Prob with Post-Processing ###
    plot_event_frcst_prob(
        data=event_APF_prob_mode, time_range_data=apf_forecast[-1], method="APF_pp"
    )
    plot_event_frcst_prob(
        data=event_CAPF_prob_mode, time_range_data=capf_forecast[-1], method="CAPF_pp"
    )
    plot_event_freq_and_prob(
        apf_forecast[-1], event_freq_mode, event_APF_prob_mode, method="APF_pp"
    )
    plot_event_freq_and_prob(
        capf_forecast[-1], event_freq_mode, event_CAPF_prob_mode, method="CAPF_pp"
    )
    plot_event_freq_and_prob_and_clim_no_sub(
        apf_forecast[-1],
        event_freq_mode,
        event_APF_prob_mode,
        clim_forecast,
        method="APF_pp",
    )
    plot_event_freq_and_prob_and_clim_no_sub(
        capf_forecast[-1],
        event_freq_mode,
        event_CAPF_prob_mode,
        clim_forecast,
        method="CAPF_pp",
    )
    ### Brier Score computation ###
    bs_mode_APF = np.zeros(3)
    bs_mode_CAPF = np.zeros(3)
    bs_ref_mode_APF = np.zeros(3)
    bs_ref_mode_CAPF = np.zeros(3)
    bss_mode_APF = np.zeros(3)
    bss_mode_CAPF = np.zeros(3)
    name_event = [
        "September SIE will be below the trend line",
        "September SIE will be less than 4.5 million km²",
        "September SIE will be less than previous year",
    ]
    print(
        "                                    VERIFICATION WITH POST-PROCESSING                                                |"
    )
    print(
        "--------------------------------------------------------------------------------------------------------------------"
    )
    for i in range(3):
        bs_val_APF, bs_ref_val_APF, bss_val_APF = BS_and_BSS(
            apf_forecast[-1], event_freq_mode, event_APF_prob_mode_pp, event_mode=i
        )
        bs_val_CAPF, bs_ref_val_CAPF, bss_val_CAPF = BS_and_BSS(
            capf_forecast[-1], event_freq_mode, event_CAPF_prob_mode_pp, event_mode=i
        )
        bs_mode_APF[i] = bs_val_APF
        bs_mode_CAPF[i] = bs_val_CAPF
        bs_ref_mode_APF[i] = bs_ref_val_APF
        bs_ref_mode_CAPF[i] = bs_ref_val_CAPF
        bss_mode_APF[i] = bss_val_APF
        bss_mode_CAPF[i] = bss_val_CAPF
        print(
            "APF Brier Score (BS) for event : "
            + name_event[i]
            + " : {}".format(bs_val_APF)
        )
        print(
            "CAPF Brier Score (BS) for event : "
            + name_event[i]
            + " : {}".format(bs_val_CAPF)
        )
        print("")
        print(
            "APF Reference Brier Score (BS_ref) for event : "
            + name_event[i]
            + " : {}".format(bs_ref_val_APF)
        )
        print(
            "CAPF Reference Brier Score (BS_ref) for event : "
            + name_event[i]
            + " : {}".format(bs_ref_val_CAPF)
        )
        print("")
        print(
            "APF Brier Skill Score (BSS) for event : "
            + name_event[i]
            + " : {}".format(bss_val_APF)
        )
        print(
            "CAPF Brier Skill Score (BSS) for event : "
            + name_event[i]
            + " : {}".format(bss_val_CAPF)
        )
        print(
            "--------------------------------------------------------------------------------------------------------------------"
        )

    print(
        "----------------------------------------------------------------------------------------------------------------------------------"
    )
    ##########################################################################################################
    ##### Forecast for 2023 #####
