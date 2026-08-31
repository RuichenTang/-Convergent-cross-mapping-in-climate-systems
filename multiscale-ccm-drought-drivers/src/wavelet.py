#######################################################################################################################
################################            MATHEMATICAL DEVELOPMENT MODULE            ################################
#######################################################################################################################
# This section contains all classes and methods associated with Wavelet decomposition, functions involved in
# preprocessing the variables and covariates of the process. The structure is oriented to highlight 3 stages:
#  1. Wavelet Transform for the target
#  2. Wavelet Transform for the input
#  3. Correlation optimization
#######################################################################################################################

#######################################################################################################################
################################                       LIBRARIES                       ################################
#######################################################################################################################
import copy
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import pywt
from scipy.stats import spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from itertools import product
import pandas as pd
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

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


#######################################################################################################################
################################                    WAVELET TRANSFORM                  ################################
#######################################################################################################################
# Legacy function, used to compare against an approach with data leakage
def WT(index_list, wavefunc='db31', level=2):
    """
    Perform wavelet decomposition and partial reconstruction of a time series.

    Parameters:
    -----------
    index_list : array-like
        The input time series to decompose.
    wavefunc : str, optional (default='db31')
        The name of the wavelet function to use.
    lv : int, optional (default=2)
        The level of wavelet decomposition.

    Returns:
    --------
    list of numpy.ndarray
        A list where:
        - The first element is the original time series.
        - The following elements are reconstructed signals for each wavelet component.
    """
    coeff = pywt.wavedec(index_list, wavefunc, mode='sym', level=level)
    coeffs = {}

    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])

    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]

    return [np.array(index_list)] + coeff

def swt_mra_causal(x, wavefunc='db2', level=3, window=None, history=None):
    """
    Causal Stationary Wavelet Transform Multi-Resolution Analysis (SWT-MRA),
    looking only to the left (past data).

    For each time instant 't', this function decomposes a causal window of data.
    This window includes an optional 'history' segment (if provided) and the
    data from 'x' up to the current instant 't'. The window's length is adjusted
    to be a multiple of 2**level using padding exclusively on the left side.
    The function returns only the last reconstructed value of each component
    for the current instant 't'.

    Parameters
    ----------
    x : array-like
        The target signal (e.g., the test segment).
    wavefunc : str, optional
        The name of the wavelet to use (e.g., 'db2', 'sym4', 'coif1', etc.).
        Defaults to 'db2'.
    level : int, optional
        The number of SWT decomposition levels. Defaults to 3.
    window : int or None, optional
        The size of the causal window (in samples). If None, the entire
        available past (up to 't') is used as the window.
        A suggested minimum window size is approximately 2 * pywt.Wavelet(wavefunc).dec_len * 2**level.
    history : array-like or None, optional
        Prior historical data to 'x' (optional). This data is only used
        on the left side to extend the causal window.

    Returns
    -------
    list
        A list of NumPy arrays, all of length len(x):
        - x: A copy of the original input signal.
        - A_L: The approximation component at the coarsest level (L).
        - D_L: The detail component at the coarsest level (D_L).
        - D_{L-1}: The detail component at the next coarsest level.
        - ...
        - D_1: The detail component at the finest level (D_1).

    Notes
    -----
    - This function ensures that no information "to the right" (future data)
      of the current time instant 't' is used in the decomposition for 't'.
    - Padding to ensure the block length is a multiple of 2**level is applied
      ONLY to the left. It first attempts to use more real historical data
      (from 'history' or earlier parts of 'x'); if insufficient, it repeats
      the first available value of the segment (causal padding).
    - Computational cost: Performs an SWT for each time instant 't'. For long
      time series, consider reducing 'level' and/or restricting the 'window'
      size to manage performance.
    """
    x = np.asarray(x, float)
    n = len(x)

    # Convert history to NumPy array if provided
    hist = None if history is None else np.asarray(history, float)

    # Output accumulators for each component, initialized with empty arrays
    A_out = np.empty(n, dtype=float)
    Ds_out = [np.empty(n, dtype=float) for _ in range(level)]  # Indices: 0->D_L, ..., L-1->D_1 (filled later)

    # Determine window size if not specified
    if window is None:
        # Heuristic: ~ 2 * decomposition_length(wavefunc) * 2**level #incorporar a las propuestas
        dec_len = pywt.Wavelet(wavefunc).dec_len
        window = 2 * dec_len * (2 ** level)

    for t in range(n):
        # Create a causal window from 'x' ending at the current time 't'
        start = max(0, t - window + 1)
        seg_x = x[start:t+1]

        # Build the base block: [history | seg_x]
        if hist is not None and hist.size > 0:
            block = np.concatenate([hist, seg_x])
        else:
            block = seg_x

        # Ensure block length is a multiple of 2**level: pad ONLY to the left
        m = 2 ** level
        need = (-len(block)) % m
        if need:
            # Try to take more real historical data if available
            if hist is not None and hist.size >= need:
                pad_left = hist[-need:]
            else:
                # If insufficient history, repeat the first value of the block (causal padding)
                # Handle case where block might be empty if x and history are too short
                pad_val = block[0] if block.size > 0 else 0.0
                pad_left = np.full(need, pad_val, dtype=float)
            block = np.concatenate([pad_left, block])

        # Perform SWT on the causal block (coeffs: [(cA1,cD1), ... , (cAL,cDL)])
        coeffs = pywt.swt(block, wavelet=wavefunc, level=level)

        # Reconstruct Approximation A_L: zero out ALL detail coefficients (cD)
        A_blk = pywt.iswt([(cA, np.zeros_like(cD)) for (cA, cD) in coeffs], wavelet=wavefunc)

        # Reconstruct Detail D_j: zero out ALL approximation (cA) and ALL other detail coefficients (cD_k != cD_j)
        D_blks = []
        for j in range(level):  # j=0 -> D1 (finest), ..., j=level-1 -> D_L (coarsest)
            pairs = []
            for k, (cA, cD) in enumerate(coeffs):
                # Keep only the current detail coefficient cD_j, zeroing others
                pairs.append((np.zeros_like(cA), cD if k == j else np.zeros_like(cD)))
            D_blks.append(pywt.iswt(pairs, wavelet=wavefunc))  # Resulting list: [D1, D2, ..., D_L]

        # Store only the last value (current time instant 't') of each component
        A_out[t] = A_blk[-1]
        # D_blks is ordered [D1, D2, ..., D_L] (finest to coarsest)
        # We store them in coarsest to finest order in Ds_out: idx 0 = D_L, idx L-1 = D_1
        for j in range(level):
            Ds_out[level - 1 - j][t] = D_blks[j][-1]

    # Return in the consistent order: [original_x, A_L, D_L, ..., D_1]
    return [x.copy(), A_out] + Ds_out

# Used for testing purposes and later discarded


#######################################################################################################################
################################             ASSOCIATION BETWEEN VARIABLES             ################################
#######################################################################################################################
def correlacion_wavelets(predictor_series,
                         target_series,
                         wavelets_predictor,
                         wavelet_target=None,
                         level=None,
                         method_wave = swt_mra_causal):
    """
    Compute the Spearman correlation between wavelet components of a predictor time series
    and a target time series across multiple wavelet families.

    Parameters:
    -----------
    predictor_series : array-like
        The predictor time series to transform and correlate.
    target_series : array-like
        The target time series to transform using a fixed wavelet family.
    wavelets_predictor : list of str
        List of wavelet function names to use for the predictor series.
    wavelet_target : str, optional (default='db31')
        The wavelet function to use for decomposing the target series.
    level : int, optional (default=2)
        The level of decomposition to apply in both series.

    Returns:
    --------
    dict
        A dictionary where keys are wavelet names and values are correlation matrices
        (numpy.ndarray) of shape (n_components_predictor, n_components_target) showing
        correlations between corresponding wavelet components.
    """
    coeffs_target = method_wave(target_series, wavefunc=wavelet_target, level=level)

    resultado = {}

    for wavefunc in wavelets_predictor:
        coeffs_pred = method_wave(predictor_series, wavefunc=wavefunc, level=level)
        matriz_corr = np.zeros((len(coeffs_pred), len(coeffs_target)))

        for i in range(len(coeffs_pred)):
            for j in range(len(coeffs_target)):
                try:
                    corr, _ = spearmanr(coeffs_pred[i], coeffs_target[j])
                except:
                    corr = np.nan
                matriz_corr[i, j] = corr

        resultado[wavefunc] = matriz_corr

    return resultado

# Used for testing purposes and later discarded
def mutual_information_wavelets(predictor_series, target_series, wavelets_predictor, wavelet_target='bior3.1', level=2):
    """
    Compute mutual information between wavelet components of predictor and target series. This function applies wavelet transforms to both predictor and target time series, then calculates mutual information between all combinations of their decomposed components.
    Mutual information is better suited for capturing non-linear relationships compared to correlation.

    Parameters
    ----------
    predictor_series : array-like
        The predictor time series to be decomposed using various wavelets.
    target_series : array-like
        The target time series to be decomposed using a single wavelet.
    wavelets_predictor : list of str
        List of wavelet names to be tested for the predictor series decomposition.
    wavelet_target : str, default='bior3.1'
        Wavelet name to be used for target series decomposition.
    level : int, default=2
        Decomposition level for wavelet transform.

    Returns
    -------
    dict
        Dictionary where keys are wavelet names and values are 2D numpy arrays
        containing mutual information scores between predictor and target components.

    Notes
    -----
    The mutual information is computed using scikit-learn's mutual_info_regression
    with k-nearest neighbors estimation method. Values are interpreted as follows:
    - MI < 0.1: Component should be discarded (weak/no dependency)
    - MI 0.1-0.5: Component shows useful dependency for modeling
    - MI > 0.5: Component exhibits very strong dependency, highly important for prediction

    """
    # Decompose target series using specified wavelet
    coeffs_target = swt_mra_causal(target_series, wavefunc=wavelet_target, level=level)
    results = {}

    # Iterate through each wavelet for predictor decomposition
    for wavefunc in wavelets_predictor:
        # Decompose predictor series using current wavelet
        coeffs_pred = swt_mra_causal(predictor_series, wavefunc=wavefunc, level=level)
        # Initialize matrix to store mutual information scores
        matriz_mi = np.zeros((len(coeffs_pred), len(coeffs_target)))

        # Calculate mutual information between all component combinations
        for i in range(len(coeffs_pred)):
            for j in range(len(coeffs_target)):
                try:
                    # Compute mutual information using regression estimator
                    mi_score = mutual_info_regression(
                        coeffs_pred[i].reshape(-1, 1),
                        coeffs_target[j],
                        random_state=42
                    )[0]
                    matriz_mi[i, j] = mi_score
                except:
                    # Set to zero if computation fails
                    matriz_mi[i, j] = 0

        # Store results for current wavelet
        results[wavefunc] = matriz_mi

    return results


#######################################################################################################################
################################  PROCESS PREDICTOR - TARGET COMBINATIONS TO AN EXCEL  ################################
#######################################################################################################################
def process_wavelet_analysis(df_scaled,
                             target,
                             predictor_columns,
                             wavelets,
                             wavelet_target,
                             output_filename,
                             level=None,
                             method='mutual_info',
                             method_wave = swt_mra_causal):
    """
    Procesa las combinaciones predictor-objetivo utilizando una lista directa de columnas de predictores.

    Parámetros:
    ----------
    df_scaled : pd.DataFrame
        El dataset escalado que contiene todas las columnas necesarias.
    target : str
        El nombre de la variable objetivo (ej. 'SPEI_1M').
    predictor_columns : list
        Una lista de los nombres de las columnas de los predictores (ej. ['NAO_lag3', 'PDO_lag16']).
    wavelets : list
        Lista de funciones wavelet a probar.
    wavelet_target : str
        La wavelet específica a usar para la variable objetivo.
    output_filename : str
        El nombre base para el archivo Excel de salida.
    method : str
        El método de asociación a calcular ('mutual_info' o 'spearman').
    """
    with pd.ExcelWriter(f"results/{output_filename}.xlsx", engine='openpyxl') as writer:

        # Itera directamente sobre la lista de columnas de predictores
        for predictor_col in predictor_columns:

            # Omite la columna si no existe en el DataFrame
            if predictor_col not in df_scaled.columns:
                print(f"Advertencia: La columna {predictor_col} no se encontró en el DataFrame y será omitida.")
                continue

            # Elige el método y calcula la asociación
            if method == 'mutual_info':
                association_results = mutual_information_wavelets(
                    df_scaled[predictor_col].values, df_scaled[target].values, wavelets, wavelet_target, level=level)
            elif method == 'spearman':
                association_results = correlacion_wavelets(
                    df_scaled[predictor_col].values,
                    df_scaled[target].values,
                    wavelets, wavelet_target,
                    level=level, method_wave=method_wave)
            else:
                raise ValueError(f"Método '{method}' no soportado. Usa 'mutual_info' o 'spearman'.")

            # Convierte los resultados a un DataFrame
            rows = [[wavelet] + list(matrix_row) for wavelet, matrix in association_results.items() for matrix_row in matrix]
            df_result = pd.DataFrame(rows, columns=['wavelet', 'Original', 'A2', 'D2', 'D1'])

            # Guarda en una hoja de Excel, asegurando que el nombre no exceda el límite
            sheet_name = predictor_col[:31]
            df_result.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"Procesado ({method}): {target} vs {predictor_col}")

#######################################################################################################################
################################                     VISUALIZATION                     ################################
#######################################################################################################################

# Used for testing purposes and later discarded
def plot_spearman_dashboard_v2(df_results, target_name, figsize=(16, 15),plt_name="wave_cor"):
    """
    Create comprehensive visualizations for Spearman correlation results with optimal wavelets.

    This function generates publication-quality plots showing the distribution of optimal
    wavelet combinations, component analysis, and performance metrics for drought forecasting
    model development.

    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing Spearman correlation results with columns:
        'Variable', 'Series', 'Wavelet', 'Spearman'
    target_name : str
        Name of the target drought index ('SPEI' or 'SPI')
    figsize : tuple, default=(16, 10)
        Figure size for the visualization

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only optimal combinations (abs(Spearman) >= 0.1)
    """

    # Configure plotting style for publication quality
    font_size = 18
    cmap_color = "PuBu"

    plt.style.use('default')
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size
    })

    # Add Wavelet_Family column to the main dataframe
    df_results['Wavelet_Family'] = df_results['Wavelet'].apply(lambda x: re.match(r"([a-zA-Z]+)", x).group(1) if re.match(r"([a-zA-Z]+)", x) else 'unknown')

    # Filter results above threshold for summary stats
    df_filtered = df_results.copy() #df_results[abs(df_results['Spearman']) >= 0.15].copy()
    df_filtered['Spearman_abs'] = df_filtered['Spearman'].abs()

    if df_filtered.empty:
        print(f"No data to plot for {target_name} with Spearman correlation absolute value >= 0.1.")
        return df_filtered

    # Create subplot layout (2 rows)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3,
                          height_ratios=[1, 2.5],
                          hspace=0.1,
                          wspace=0.01,
                          top=0.92,
                          bottom=0.08,
                          left=0.06,
                          right=0.96)


    # 4. Wavelet Component Heatmap (Bottom-left and center) - Based on ALL data
    ax5 = fig.add_subplot(gs[1, 0:2])

    df_results['Spearman_abs'] = df_results['Spearman'].abs()
    idx = df_results.groupby(['Variable', 'Target'])['Spearman_abs'].idxmax()
    df_max_abs_corr = df_results.loc[idx]

    pivot_data = df_max_abs_corr.pivot_table(index='Variable',
                                             columns='Target',
                                             values='Spearman',
                                             fill_value=0)
    # ordenar variables por magnitud máxima de correlación
    component_order = "Original"
    order = pivot_data[component_order].abs().sort_values(ascending=False).index
    pivot_data = pivot_data.loc[order]
    # --- Orden deseado de componentes en el eje X ---
    comp_order = ["Original", "A2", "D1", "D2"]

    # deja solo las que existan (por si falta alguna)
    comp_order = [c for c in comp_order if c in pivot_data.columns]

    # reordenar columnas
    pivot_data = pivot_data[comp_order]

    vmax = pivot_data.abs().max().max()
    if vmax == 0: vmax = 1.0

    im = ax5.imshow(pivot_data.values,
                    cmap=cmap_color,
                    aspect='auto',
                    vmin=-vmax,
                    vmax=vmax)

    ax5.set_xticks(range(len(pivot_data.columns)))
    ax5.set_xticklabels(pivot_data.columns)
    ax5.set_yticks(range(len(pivot_data.index)))
    ax5.set_yticklabels(pivot_data.index, fontsize=font_size)
    plt.setp(ax5.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            if value != 0:
                color = 'white' if abs(value) > vmax * 0.6 else 'black'
                ax5.text(j, i,
                         f'{value:.2f}',
                         ha='center',
                         va='center',
                         color=color,
                         fontsize=font_size)

    ax5.set_title('Max Spearman Correlation by Variable and Component', fontweight='bold')
    ax5.set_xticks(range(len(comp_order)))
    ax5.set_xticklabels(comp_order)
    ax5.set_xlabel('Wavelet Component')
    ax5.set_ylabel('Variable')
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('Spearman Correlation', rotation=270, labelpad=25)
    ax5.axvline(x=0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.8)

    # 5. Summary Statistics (Bottom-right) - Based on filtered data
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    total_combinations = len(df_filtered)
    max_correlation = df_filtered['Spearman_abs'].max()
    unique_vars = df_results['Variable'].nunique()
   # unique_wavelets_filtered = df_filtered['Wavelet_Family'].nunique()

    summary_text = f"""SUMMARY STATISTICS
{"=" * 31}

Optimal Combinations: {total_combinations}

Max. Correlation: {max_correlation:.2f}

Variables Involved: {unique_vars}

Threshold: abs(Spearman) >= 0.15"""

#Unique Wavelet Families: {unique_wavelets_filtered}

    ax6.text(0.06, 0.95,
             summary_text,
             transform=ax6.transAxes,
             fontsize=font_size,
             verticalalignment='top',
             fontfamily='monospace',
             clip_on='monospace',
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor='lightgray',
                       alpha=0.8,
                       edgecolor='black'))

    plt.savefig(f"{plt_name}.pdf", bbox_inches="tight")
    plt.show()

    return df_filtered

def plot_spearman_dashboard_v3(df_results, target_name, figsize=(16, 15)):
    """
    Create comprehensive visualizations for Spearman correlation results with optimal wavelets.

    This function generates publication-quality plots showing the distribution of optimal
    wavelet combinations, component analysis, and performance metrics for drought forecasting
    model development.

    Parameters
    ----------
    df_results : pandas.DataFrame
        DataFrame containing Spearman correlation results with columns:
        'Variable', 'Series', 'Wavelet', 'Spearman'
    target_name : str
        Name of the target drought index ('SPEI' or 'SPI')
    figsize : tuple, default=(16, 10)
        Figure size for the visualization

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only optimal combinations (abs(Spearman) >= 0.1)
    """

    # Configure plotting style for publication quality
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 15,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'figure.titlesize': 16
    })

    # Add Wavelet_Family column to the main dataframe
    df_results['Wavelet_Family'] = df_results['Wavelet'].apply(lambda x: re.match(r"([a-zA-Z]+)", x).group(1) if re.match(r"([a-zA-Z]+)", x) else 'unknown')

    # Filter results above threshold for summary stats
    df_filtered = df_results.copy() #df_results[abs(df_results['Spearman']) >= 0.15].copy()
    df_filtered['Spearman_abs'] = df_filtered['Spearman'].abs()

    if df_filtered.empty:
        print(f"No data to plot for {target_name} with Spearman correlation absolute value >= 0.1.")
        return df_filtered

    # Create subplot layout (2 rows)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.5, wspace=0.35, top=0.92, bottom=0.08,
                          left=0.08, right=0.95)

    # 1. Component Distribution (Top-left) - Based on ALL data
    ax1 = fig.add_subplot(gs[0, 0])
    component_counts = df_results['Target'].value_counts()
    colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD']
    bars = ax1.bar(component_counts.index, component_counts.values,
                   color=colors[:len(component_counts)], alpha=0.8, edgecolor='black', linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    ax1.set_title('Distribution of Wavelet Components', fontweight='bold')
    ax1.set_xlabel('Wavelet Component')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Top Wavelet Families (Top-center) - Based on ALL data
    ax2 = fig.add_subplot(gs[0, 1])
    wavelet_family_counts = df_results['Wavelet_Family'].value_counts().head(8)
    bars2 = ax2.barh(range(len(wavelet_family_counts)), wavelet_family_counts.values,
                     color='#3498DB', alpha=0.8, edgecolor='black', linewidth=0.5)

    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height() / 2.,
                 f'{int(width)}', ha='left', va='center', fontweight='bold')

    ax2.set_yticks(range(len(wavelet_family_counts)))
    ax2.set_yticklabels(wavelet_family_counts.index)
    ax2.invert_yaxis()
    ax2.set_title('Most Frequent Wavelet Families', fontweight='bold')
    ax2.set_xlabel('Frequency')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Spearman Correlation Distribution (Top-right) - Based on ALL data
    ax3 = fig.add_subplot(gs[0, 2])
    spearman_values = df_results['Spearman']
    ax3.hist(spearman_values, bins=15, alpha=0.7, color='#27AE60', edgecolor='black', linewidth=0.5)

    mean_spearman = abs(spearman_values).mean()
    median_spearman = abs(spearman_values).median()
    ax3.axvline(mean_spearman, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_spearman:.3f}')
    ax3.axvline(median_spearman, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_spearman:.3f}')

    ax3.set_title('Distribution of Spearman Correlation', fontweight='bold')
    ax3.set_xlabel('Spearman Correlation')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Wavelet Component Heatmap (Bottom-left and center) - Based on ALL data
    ax5 = fig.add_subplot(gs[1, 0:2])

    df_results['Spearman_abs'] = df_results['Spearman'].abs()
    idx = df_results.groupby(['Variable', 'Target'])['Spearman_abs'].idxmax()
    df_max_abs_corr = df_results.loc[idx]

    pivot_data = df_max_abs_corr.pivot_table(index='Variable', columns='Target',
                                         values='Spearman', fill_value=0)

    vmax = pivot_data.abs().max().max()
    if vmax == 0: vmax = 1.0

    im = ax5.imshow(pivot_data.values, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)

    ax5.set_xticks(range(len(pivot_data.columns)))
    ax5.set_xticklabels(pivot_data.columns)
    ax5.set_yticks(range(len(pivot_data.index)))
    ax5.set_yticklabels(pivot_data.index, fontsize=12)
    plt.setp(ax5.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            if value != 0:
                color = 'white' if abs(value) > vmax * 0.6 else 'black'
                ax5.text(j, i, f'{value:.2f}', ha='center', va='center',
                         color=color, fontweight='bold', fontsize=8)

    ax5.set_title('Max Spearman Correlation by Variable and Component', fontweight='bold')
    ax5.set_xlabel('Wavelet Component')
    ax5.set_ylabel('Variable')
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('Spearman Correlation', rotation=270, labelpad=20)

    # 5. Summary Statistics (Bottom-right) - Based on filtered data
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    total_combinations = len(df_filtered)
    max_correlation = df_filtered['Spearman_abs'].max()
    unique_vars = df_results['Variable'].nunique()
    unique_wavelets_filtered = df_filtered['Wavelet_Family'].nunique()

    summary_text = f"""SUMMARY STATISTICS
{"=" * 30}

Optimal Combinations: {total_combinations}

Max. Correlation: {max_correlation:.3f}

Variables Involved: {unique_vars}

Unique Wavelet Families: {unique_wavelets_filtered}

Threshold: abs(Spearman) >= 0.15"""

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=15,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8, edgecolor='black'))

    plt.show()

    return df_filtered


#######################################################################################################################
################################                   MODELING DATASET                    ################################
#######################################################################################################################
def _decompose_raw(series: pd.Series, wavelet: str, level: int):
    """
    Llama a tu swt_mra_causal SIN tocar la escala ni interpolar.
    Devuelve dict con A3, D3, D2, D1 (misma longitud que la serie).
    Asume que la serie NO tiene NaN. Si los hay, la función levantará error (mejor así).
    """
    x = series.to_numpy(dtype=float)  # sin dropna, sin rellenos
    coeffs = swt_mra_causal(x, wavefunc=wavelet, level=level)  # [x, A3, D3, D2, D1]
    return {'A2': np.asarray(coeffs[1], float),
            'D2': np.asarray(coeffs[2], float),
            'D1': np.asarray(coeffs[3], float)}

def _looks_minmax_01(df: pd.DataFrame, cols: list, tol=1e-6):
    """
    Devuelve True si TODAS las columnas numéricas en 'cols' parecen estar en [0,1].
    Sirve para detectar bases ya escaladas por error.
    """
    if not cols:
        return False
    sub = df[cols].select_dtypes(include=[np.number])
    if sub.empty:
        return False
    min_ok = (sub.min(skipna=True) >= -tol).all()
    max_ok = (sub.max(skipna=True) <= 1.0 + tol).all()
    return bool(min_ok and max_ok)

def create_modeling_datasets(metadata, AGGREGATION, data_load, data_save, level=2, return_scaled=False):
    """
    Genera 10 archivos (SPEI/SPI x [Original, A3, D3, D2, D1]) en ESCALA ORIGINAL,
    usando EXACTAMENTE las wavelets indicadas en el JSON:
      - Target SIEMPRE se descompone con metadata['wavelet_SPI'/'wavelet_SPEI'] (una sola familia).
      - Cada predictor usa la wavelet indicada en el JSON para el componente correspondiente.
    """
    # 1) Config desde JSON
    df_spi  = pd.DataFrame(metadata['data_spi'])   # Variable, Predictor, Target, Wavelet, Spearman
    df_spei = pd.DataFrame(metadata['data_spei'])

    def predictor_list(df_wavelets, target_component):
        sub = df_wavelets[df_wavelets['Target'] == target_component]
        return [(r['Variable'], r['Wavelet'], r['Predictor']) for _, r in sub.iterrows()]

    components = ['Original', 'A2', 'D2', 'D1']
    spi_cfg  = {c: {'predictors': predictor_list(df_spi, c),
                    'component': c,
                    'target_wavelet': metadata['wavelet_SPI']}  for c in components}
    spei_cfg = {c: {'predictors': predictor_list(df_spei, c),
                    'component': c,
                    'target_wavelet': metadata['wavelet_SPEI']} for c in components}

    # 2) Cargar BASE (DEBE SER ESCALA ORIGINAL)
    df_base = pd.read_csv(f'data/processed/{data_load}_{AGGREGATION}.csv')
    df_base['Time'] = pd.to_datetime(df_base['Time'])

    # Chequeo: ¿la base parece estar minmax 0–1?
    numeric_cols = [c for c in df_base.columns if c != 'Time']
    if _looks_minmax_01(df_base, numeric_cols):
        raise ValueError(
            "La base que estás cargando parece estar escalada en 0–1. "
            "Apunta a un archivo en ESCALA ORIGINAL (con negativos si corresponde) "
            "y vuelve a ejecutar."
        )

    # Chequeo: NaN → paramos (así no alteramos tus series)
    if df_base[numeric_cols].isna().any().any():
        nan_cols = df_base.columns[df_base.isna().any()].tolist()
        raise ValueError(
            f"Hay NaNs en la base: {nan_cols}. "
            "Tu descomposición funciona perfecto sin NaN, así que limpia o imputa ANTES."
        )

    # 3) Constructor de datasets
    def build_dataset(target_name: str, cfg: dict) -> pd.DataFrame:
        out = pd.DataFrame({'Time': df_base['Time']})

        # TARGET
        if cfg['component'] == 'Original':
            out[target_name] = df_base[target_name].astype(float)
        else:
            comps_t = _decompose_raw(df_base[target_name], cfg['target_wavelet'], level)
            out[target_name] = comps_t[cfg['component']]

        # PREDICTORES
        for var, wv, pcomp in cfg['predictors']:
            if var not in df_base.columns:
                continue
            col = f"{var}_{pcomp}_{wv}"
            if pcomp == 'Original':
                out[col] = df_base[var].astype(float)
            else:
                comps_p = _decompose_raw(df_base[var], wv, level)
                out[col] = comps_p[pcomp]

        return out  # sin dropna, respeta 1:1 con Time y escala original

    # 4) Generar y guardar en ESCALA ORIGINAL
    made = {}

    for comp, cfg in spi_cfg.items():
        name = f"SPI_{AGGREGATION}_{comp}"
        df_o = build_dataset(metadata['target'][1], cfg)
        df_o.to_csv(f"data/processed/{data_save}_{name}.csv", index=False)
        made[name] = df_o

    for comp, cfg in spei_cfg.items():
        name = f"SPEI_{AGGREGATION}_{comp}"
        df_o = build_dataset(metadata['target'][0], cfg)
        df_o.to_csv(f"data/processed/{data_save}_{name}.csv", index=False)
        made[name] = df_o

    print(f"Guardados {len(made)} archivos en ESCALA ORIGINAL.")

    # 5) (Opcional) versiones escaladas SOLO en memoria
    if not return_scaled:
        return made

    from sklearn.preprocessing import MinMaxScaler
    feat_cols = sorted(set().union(*[set(df.columns) for df in made.values()]) - {'Time'})
    big = pd.concat(made.values(), ignore_index=True, sort=False)[feat_cols].copy()

    scaler = MinMaxScaler()
    scaler.fit(big.fillna(0.0).values)

    scaled = {}
    for name, df_o in made.items():
        X = df_o.reindex(columns=feat_cols).fillna(0.0).values
        Xs = scaler.transform(X)
        df_s = pd.DataFrame(Xs, columns=feat_cols, index=df_o.index)
        df_s.insert(0, 'Time', df_o['Time'])
        keep = ['Time'] + [c for c in feat_cols if c in df_o.columns]
        scaled[name] = df_s[keep]

    return scaled


# --------------------------------------------------------------
# WAVELET ANALYSIS FUNCTION
# --------------------------------------------------------------

def analyze_wavelet_energy(target, wavelet, level=2):
    """
    Execute the swt_mra_causal function for a specified wavelet family and
     compute the corresponding energies and proportions.
    """
    try:
        out = swt_mra_causal(target, wavefunc=wavelet, level=level)
        x, A2, D2, D1 = out  # level=2

        var_tot = np.var(target)

        E_A2 = np.var(A2) / var_tot
        E_D2 = np.var(D2) / var_tot
        E_D1 = np.var(D1) / var_tot

        E_total = E_A2 + E_D2 + E_D1

        return {
            "wavelet": wavelet,
            "E_total": E_total,
            "E_A2": E_A2,
            "E_D2": E_D2,
            "E_D1": E_D1
        }
    except Exception as e:
        return {
            "wavelet": wavelet,
            "E_total": np.nan,
            "E_A2": np.nan,
            "E_D2": np.nan,
            "E_D1": np.nan,
        }
# --------------------------------------------------------------
# PRINCIPAL LOOP FOR FAMILY EVALUATION
# --------------------------------------------------------------

def wavelet_energy_ranking(target, families, level=2):
    results = []
    for w in tqdm(families):
        info = analyze_wavelet_energy(target, w, level)
        results.append(info)

    df = pd.DataFrame(results)

    # Ranking for total energy
    df = df.sort_values("E_total", ascending=False).reset_index(drop=True)
    return df