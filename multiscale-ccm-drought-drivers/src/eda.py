########################################################################################################################
################################            EXPLORATORY DATA ANALYSIS MODULE            ################################
########################################################################################################################
# This section contains all the functions used for the Exploratory Data Analysis (EDA) of the project.
# The functions are designed to support the analysis of time series data, specifically for drought indicators
# and macroclimatic indices. The main stages of the analysis covered by these functions are:
#  1. Data Loading and Preprocessing
#  2. Descriptive Statistics and Distribution Analysis
#  3. Temporal and Correlation Analysis (ACF, PACF, CCF)
#  4. Visualization of time series and relationships
#######################################################################################################################


#######################################################################################################################
################################                       LIBRARIES                       ################################
#######################################################################################################################
from statsmodels.tsa.stattools import ccf
import pandas as pd
import json
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import numpy as np


#######################################################################################################################
################################                       METADATA                        ################################
#######################################################################################################################
def read_metadata(file_path: str):
    """
    Reads the metadata from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary with the JSON file data if read successfully.
        None: If there is an error reading the file or decoding the JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding the file {file_path}. Ensure the file contains valid JSON.")
        return None


#######################################################################################################################
################################                        DISPLAY                        ################################
#######################################################################################################################
def configure_pandas_display():
    """
    Configures pandas to display all columns and adjust display width and rows.
    """
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)        # Do not truncate the table width
    pd.set_option('display.max_rows', None)     # Optionally, you can set this to limit the rows if needed
    pd.set_option('display.max_colwidth', None) # Ensure that no column gets truncated by width


#######################################################################################################################
################################                 DIFFERENCED DATAFRAME                 ################################
#######################################################################################################################
def create_differenced_dataframe(df, time_col='Time'):
    """
    Creates a differenced DataFrame for correlation analysis.

    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame
    time_col : str
        Name of the time column

    Returns:
    --------
    pandas.DataFrame
        Differenced DataFrame ready for analysis
    """
    # Copy and set time index
    df_diff = df.copy().set_index(time_col).sort_index()

    # Difference numeric columns only
    numeric_cols = df_diff.select_dtypes(include=['float64', 'int64']).columns
    df_diff[numeric_cols] = df_diff[numeric_cols].diff()

    # Reset index and remove NaN
    return df_diff.reset_index().dropna()


#######################################################################################################################
################################             SIGNIFICANT LAG CORRELATIONS              ################################
#######################################################################################################################
def quick_lag_analysis(df, variable, target, lags=20, threshold=0.06):
    """
    Identifies significant lags between a predictor and a target variable using cross-correlation.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing time-aligned series.
    - variable (str): The name of the predictor variable.
    - target (str): The name of the target variable.
    - lags (int): Number of lags to compute (default is 24).
    - threshold (float): Absolute correlation threshold to identify significant lags (default is 0.08).

    Returns:
    - sig_lags (list): List of lag indices with absolute correlation above the threshold.
    - max_corr (float): The highest absolute correlation value observed.
    """
    ccf_values = ccf(df[variable], df[target], adjusted=False)[:lags + 1]
    sig_lags = [lag for lag, val in enumerate(ccf_values) if abs(val) > threshold]
    return sig_lags, max(ccf_values, key=abs)


def classify_drought(value):
    """
    Classifies drought conditions based on the World Meteorological Organization's
    SPI/SPEI value guidelines (WMO-No. 1090).

    Parameters:
    - value (float): The SPI/SPEI drought index value.

    Returns:
    - str: A categorical drought classification based on WMO standards.
    """
    if pd.isna(value):
        return 'Missing'
    elif value <= -2.0:
        return 'Extremely Dry'
    elif value <= -1.5:
        return 'Severely Dry'
    elif value <= -1.0:
        return 'Moderately Dry'
    elif value >= 2.0:
        return 'Extremely Wet'
    elif value >= 1.5:
        return 'Severely Wet'
    elif value >= 1.0:
        return 'Moderately Wet'
    else:
        return 'Near Normal'


#######################################################################################################################
################################                     MINMAX SCALING                    ################################
#######################################################################################################################
def minmax_scale_dataframe(df, columns_to_scale, feature_range=(0, 1)):
    """
    Apply MinMax scaling to selected columns of a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_scale (list): List of column names to scale.
        feature_range (tuple): Desired range of transformed data. Default is (0, 1).

    Returns:
        df_scaled (pd.DataFrame): DataFrame with scaled values.
        scaler (MinMaxScaler): Fitted scaler object for inverse transformation.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df_scaled, scaler

def inverse_transform_minmax_column(scaled_values, scaler, columns_to_scale, target_column):
    """
    Reverse the MinMax scaling of a specific column.

    Parameters:
        scaled_values (array-like): Scaled values of the target column.
        scaler (MinMaxScaler): The fitted scaler used during scaling.
        columns_to_scale (list): Same list of columns passed to the scaler during scaling.
        target_column (str): The name of the column to inverse-transform.

    Returns:
        original_values (np.ndarray): Original scale values of the target column.
    """
    index = columns_to_scale.index(target_column)
    temp = np.zeros((len(scaled_values), len(columns_to_scale)))
    temp[:, index] = np.array(scaled_values).ravel()
    inverse = scaler.inverse_transform(temp)
    return inverse[:, index]


#######################################################################################################################
################################                          PLOTS                        ################################
#######################################################################################################################
def plot_time_series(df, metadata, save_plot=False, name_plot="timeseries", color_plot="#1bbbff", format_plot="eps", sample_plot=700):
    """
    Plots time series for multiple features in a dynamic grid layout, optimized for
    publication quality.

    Args:
        df (pd.DataFrame): DataFrame containing the data, with a "Time" column.
        metadata (dict): Dictionary with metadata including a list of "features".
        save_plot (bool): Whether to save the plot as a file.
        name_plot (str): Name of the saved plot file.
        color_plot (str): The color for the plot lines.
        format_plot (str): Format to save the plot (e.g., 'png', 'eps', 'pdf').
        sample_plot (int): Number of last observations to plot.

    Returns:
        None (plots are displayed).
    """
    import math
    import matplotlib.pyplot as plt

    # Filter features, excluding time columns
    features = [col for col in metadata["features"] if col not in metadata["time"]]
    n_features = len(features)

    # Dynamic grid with 3 columns
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    # Style optimized for papers
    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure size adjusted for the new grid
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(22, n_rows * 5))
    axes = axes.flatten()

    line_color = color_plot

    # Iterate over all features and create plots
    for i, feature in enumerate(features):
        ax = axes[i]

        subset_df = df.iloc[-sample_plot:]
        ax.plot(subset_df["Time"], subset_df[feature], color=line_color, linewidth=2)

        # Larger fonts
        ax.set_xlabel("Time", fontsize=23, labelpad=10)
        ax.set_ylabel(feature, fontsize=23, labelpad=10)
        ax.tick_params(axis='x', labelsize=23, rotation=30)
        ax.tick_params(axis='y', labelsize=23)

        ax.grid(True, which='major', linestyle='--', linewidth=0.7)

    # Hide any empty subplots that are not needed
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=4.0)

    if save_plot:
        plt.savefig(f"results/{name_plot}.{format_plot}", format=format_plot, dpi=300, bbox_inches='tight')

    plt.show()


def plot_boxplots_by_month(df, metadata, save_plot=False, name_plot="boxplot", color_plot="#1bbbff", format_plot="eps"):
    """
    Plots publication-quality boxplots for multiple features grouped by month in a
    dynamic grid layout.

    Args:
        df (pd.DataFrame): DataFrame with data and a "Month" column.
        metadata (dict): Dictionary with metadata, including a list of "features".
        save_plot (bool): Whether to save the plot as a file.
        name_plot (str): Name for the saved plot file.
        color_plot (str): Color for the boxplots.
        format_plot (str): Format for saving the plot (e.g., 'png', 'eps').

    Returns:
        None (plots are displayed).
    """
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Filter features, excluding time columns
    features = [col for col in metadata["features"] if col not in metadata["time"]]
    n_features = len(features)

    if "Month" not in df.columns:
        raise ValueError("The DataFrame must contain a 'Month' column to group the boxplots.")

    # --- MODIFIED: Dynamic grid with 3 columns ---
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    # --- MODIFIED: Style optimized for papers ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- MODIFIED: Figure size adjusted for the new grid ---
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(22, n_rows * 6))
    axes = axes.flatten()

    # Iterate over all features and create boxplots
    for i, feature in enumerate(features):
        ax = axes[i]

        # Use seaborn for a cleaner look with specified color
        sns.boxplot(x=df["Month"], y=df[feature], ax=ax, color=color_plot)

        # --- REMOVED: Redundant title ---

        # --- MODIFIED: Larger fonts for publication ---
        ax.set_xlabel("Month", fontsize=20, labelpad=10)
        ax.set_ylabel(feature, fontsize=20, labelpad=10)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=20)

        ax.grid(True, which='major', linestyle='--', linewidth=0.7)

    # Hide any empty subplots
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=4.0)

    if save_plot:
        # Save with high resolution for publication
        plt.savefig(f"results/{name_plot}.{format_plot}", format=format_plot, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)

def plot_acf_pacf_for_numerical_columns(df, lags=30, sample_plot=None, name_plot='CCF', save_plot=False, format_plot="eps"):
    """
    Plots a combined Autocorrelation Function (ACF) and Partial Autocorrelation
    Function (PACF) for each numerical column in a DataFrame. All plots are
    displayed side-by-side in a single figure with large, publication-quality text.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing time series data.
    lags : int, optional (default=30)
        Number of lags to include in the plots.
    sample_plot : int or None, optional (default=None)
        If specified, only the last `sample_plot` rows of the DataFrame will be used.
        If None, the full DataFrame is used.
    """

    df_num = df.select_dtypes(include='number')
    columns = df_num.columns
    n_plots = len(columns)
    df_used = df_num if sample_plot is None else df_num.tail(sample_plot)

    # Create a figure with side-by-side subplots, adjusted for larger fonts
    fig, axes = plt.subplots(1, n_plots, figsize=(11 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]  # Ensure axes is always iterable

    for i, column in enumerate(columns):
        ax = axes[i]
        series = df_used[column].dropna()
        n_obs = len(series)

        # Calculate ACF and PACF values (excluding lag 0)
        acf_vals = sm.tsa.stattools.acf(series, nlags=lags)[1:]
        pacf_vals = sm.tsa.stattools.pacf(series, nlags=lags, method='ywm')[1:]
        x_lags = np.arange(1, lags + 1)

        # Plot confidence intervals
        conf_level = 1.96 / np.sqrt(n_obs)
        ax.axhspan(-conf_level, conf_level, color='gray', alpha=0.2, zorder=1, label='95% Confidence Interval')

        # Plot ACF
        markerline_acf, stemlines_acf, _ = ax.stem(x_lags - 0.15, acf_vals, label='ACF', basefmt=" ")
        plt.setp(markerline_acf, 'color', '#1f77b4', 'markersize', 8)
        plt.setp(stemlines_acf, 'color', '#1f77b4', 'linewidth', 2.5)

        # Plot PACF
        markerline_pacf, stemlines_pacf, _ = ax.stem(x_lags + 0.15, pacf_vals, label='PACF', basefmt=" ")
        plt.setp(markerline_pacf, 'color', '#ff7f0e', 'markersize', 8)
        plt.setp(stemlines_pacf, 'color', '#ff7f0e', 'linewidth', 2.5)

        # --- Aesthetics for Publication (with larger fonts) ---
        # Add variable name inside the plot area
        ax.text(0.95, 0.95, column, transform=ax.transAxes, fontsize=25, fontweight='bold',
                ha='right', va='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

        ax.set_xlabel('Lag', fontsize=25)
        ax.set_ylabel('Correlation', fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=22) # Larger tick numbers

        ax.grid(linestyle='--', alpha=0.6, axis='y')
        ax.set_ylim(-0.6, 0.6)
        ax.set_xlim(0, lags + 1)
        ax.set_xticks(np.arange(0, lags + 1, 5))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='black', linewidth=0.8)

    # Create a single, shared legend with larger font
    handles, labels = axes[0].get_legend_handles_labels()
    order = [2, 0, 1]
    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=3, fontsize=26, frameon=False)

    if save_plot:
        # Save with high resolution for publication
        plt.savefig(f"results/{name_plot}.{format_plot}", format=format_plot, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)


def plot_ccf_for_targets_and_predictors(df, targets, lags=30, sample_plot=None):
    """
    Plots the Cross-Correlation Function (CCF) between each predictor and each target variable.

    The result is a grid of plots where each row corresponds to a predictor and each column to a target.
    Each cell shows the CCF between the predictor and the target over the specified number of lags.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numerical time series data.

    targets : list of str
        List of column names to use as target variables for cross-correlation.

    lags : int, optional (default=30)
        Number of lags to include in the CCF plots.

    sample_plot : int or None, optional (default=None)
        If specified, only the last `sample_plot` rows will be used.
        If None, the entire DataFrame will be used.

    Returns:
    -------
    None
        Displays the CCF plots between predictors and targets.
    """
    df_num = df.select_dtypes(include='number')
    df_used = df_num if sample_plot is None else df_num.tail(sample_plot)

    # Get predictors: all numeric columns except the targets
    predictors = [col for col in df_num.columns if col not in targets]

    n_rows = len(predictors)
    n_cols = len(targets)

    fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axarr = [[axarr]]
    elif n_rows == 1:
        axarr = [axarr]
    elif n_cols == 1:
        axarr = [[ax] for ax in axarr]

    for i, pred in enumerate(predictors):
        for j, target in enumerate(targets):
            ax = axarr[i][j]
            sm.graphics.tsa.plot_ccf(df_used[pred], df_used[target], lags=lags, ax=ax,
                                     color='purple', vlines_kwargs={"colors": 'purple'}, title=None)
            ax.set_title(f'CCF: {pred} vs {target}', fontsize=14)
            ax.set_xlabel('Lag', fontsize=12)
            ax.set_ylabel('Correlation', fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.show()


#######################################################################################################################
################################                     LAGGED DATASETS                   ################################
#######################################################################################################################
def create_specific_lags(df, lag_config, target_name):
    """
    Generates a lagged feature set based on a custom lag configuration for a given target variable.

    Parameters:
        df (pd.DataFrame): Original DataFrame containing time-series data.
        lag_config (dict): Dictionary where keys are variable names and values are lists of lag steps to include.
        target_name (str): Name of the target variable for modeling.

    Returns:
        pd.DataFrame: A new DataFrame with time-related columns and selected lagged features,
                      including the unlagged target variable. Rows with missing values due to lagging are dropped.
    """
    df_result = df[['Time', 'Month', 'Year']].copy()
    df_result[target_name] = df[target_name]  # Include current value of the target variable

    for var, lags in lag_config.items():
        for lag in lags:
            col_name = f"{var}_lag{lag}"
            df_result[col_name] = df[var].shift(lag)

    return df_result.dropna()