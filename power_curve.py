import glob
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TCX_PATTERN = "./data/*.tcx"
DURATIONS = [1,2,5,10,20,30,60,120,300,600,1200,1800, 3600, 5400, 7200, 5*3600] 
WEIGHT = 70

# ---------------------------------------
# Parse TCX
# ---------------------------------------
def parse_tcx_file(path):
    ns = {
        'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
        'ext': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'
    }

    tree = ET.parse(path)
    root = tree.getroot()

    rows = []
    for tp in root.findall('.//tcx:Trackpoint', ns):
        t_el = tp.find('tcx:Time', ns)
        if t_el is None: 
            continue
        t = datetime.fromisoformat(t_el.text.replace("Z","+00:00")).astimezone(timezone.utc)
        watts_el = tp.find('.//ext:Watts', ns)
        if watts_el is None:
            continue
        try:
            watts = float(watts_el.text)
        except:
            continue
        rows.append({"time": pd.Timestamp(t), "Power": watts})

    return pd.DataFrame(rows)


# ---------------------------------------
# Load, resample safely (no infinite ffill)
# ---------------------------------------
def load_all_tcx(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No TCX found")

    dfs = []
    for i, f in enumerate(files):
        print("Reading", f)
        df = parse_tcx_file(f)
        if df.empty:
            continue
        df = df.sort_values("time")
        try:
            df = df.set_index("time").resample("1s").ffill()
            # ----- FIX: stop ffill across large gaps -----
            df["gap"] = df.index.to_series().diff().dt.total_seconds()
            df.loc[df["gap"] > 5, "Power"] = np.nan
            df = df.drop(columns=["gap"]).dropna()
            df["session"] = i

            dfs.append(df)
        except ValueError:
            continue



    if not dfs:
        raise ValueError("No power data found")

    all_data = pd.concat(dfs)
    all_data = all_data.reset_index().rename(columns={"index": "time"})
    return all_data

coggan_profiles_wkg = {
# estimation based on Allen, H., Coggan, A., & McGregor, S. (2019). Training and Racing with a Power Meter
    "Diamond": {
        5:21.3,
        60:10.3,
        300:6.55,
        1200:5.5
    },
    "Titanium": {
        5:19.5,
        60:9.6,
        300:5.8,
        1200:5
    },
    "Platinum": {
        5:18.06,
        60:	8.97,
        300:5.33,
        1200:4.44
    },

    "Gold": {
        5: 16.15,
        60: 8.17,
        300: 4.6,
        1200: 3.82
    },
    "Silver": {         
        5: 14.52,
        60: 7.48,
        300: 3.98,
        1200: 3.29
    },
    "Bronze": {          
        5: 12.89,
        60: 6.79,
        300: 3.36,
        1200: 2.75
    },
    "Iron": {     
        5: 11.26,
        60: 6.1,
        300: 2.74,
        1200: 2.22
    },
    "Wood": {     
        5: 8.3,
        60: 5,
        300: 1.6,
        1200: 1.2
    },
}

coggan_profiles_watts = {
    level: {dur: wkg * WEIGHT for dur, wkg in durs.items()}
    for level, durs in coggan_profiles_wkg.items()
}

def morton_3p(t, CP, Wp, k):
    return CP + Wp / (t + k)

def morton_model(t, CP, Wp, tau):
    return CP + (Wp / t) * (1 - np.exp(-t / tau))

def fit_morton_model(profile_dict, durations):
    # convert dict to arrays
    t_ref = np.array(sorted(profile_dict.keys()), dtype=float)
    p_ref = np.array([profile_dict[t] for t in t_ref], dtype=float)

    # initial guesses
    p0 = [p_ref[-1],  # CP approx FTP
          20000,      # W' guess
          10]         # k guess

    params, _ = curve_fit(morton_3p, t_ref, p_ref, p0=p0, maxfev=20000)
    
    CP, Wp, k = params
    
    t_target = np.array(durations, dtype=float)
    smoothed = morton_3p(t_target, CP, Wp, k)
    return smoothed, params

def interpolate_profile(profile_dict, durations):
    # key durations (Coggan reference)
    ref_durs = np.array(sorted(profile_dict.keys()))
    ref_vals = np.array([profile_dict[d] for d in ref_durs])

    # log-log interpolation
    log_ref_durs = np.log(ref_durs)
    log_ref_vals = np.log(ref_vals)
    log_target = np.log(np.array(durations))

    smoothed = np.exp(np.interp(log_target, log_ref_durs, log_ref_vals))
    return smoothed

def compute_model_based_ftp(power_curve_df):
    durations = power_curve_df["duration_s"].values
    power = power_curve_df["best_avg_power"].values

    valid = np.isfinite(power) & (durations > 0)
    t = durations[valid]
    p = power[valid]

    # initial guesses
    p0 = [np.percentile(p, 25), 20000, 300]

    params, _ = curve_fit(
        morton_model,
        t,
        p,
        p0=p0,
        maxfev=10000
    )

    CP, Wp, tau = params

    # FTP ≈ model power at 60 min
    ftp = morton_model(3600.0, CP, Wp, tau)
    return float(ftp)


# ---------------------------------------
# Power curve
# ---------------------------------------
def compute_best_average_power(power_series, duration_seconds):
    win = int(duration_seconds)
    if len(power_series) < win:
        return np.nan
    roll = power_series.rolling(win, min_periods=win).mean()
    return float(roll.max())

def compute_power_curve(df, durations):
    p = df["Power"].reset_index(drop=True)
    return pd.DataFrame({
        "duration_s": durations,
        "best_avg_power": [compute_best_average_power(p, d) for d in durations]
    })

def compute_ftp_from_curve(power_curve_df):
    """
    Compute FTP from a power curve DataFrame.
    """
    row = power_curve_df.loc[
        power_curve_df["duration_s"] == 20 * 60,
        "best_avg_power"
    ]

    if row.empty or np.isnan(row.iloc[0]):
        return np.nan

    return 0.95 * float(row.iloc[0])


def score_templates(y_obs, y_pred,
                    close_tol = 10,
                    max_diff = 30):

    diff = np.abs(y_pred - y_obs)

    valid = diff <= max_diff
    if not np.any(valid):
        return 0
    score = np.sum(diff[valid] <= close_tol)
    score = score / len(y_obs)
    return score

def plot_ftp_progression(df, ax):
    session_dates = []
    session_ftps = []
    
    if "session" not in df.columns:
        return
        
    for sess_id, group in df.groupby("session"):
        duration = (group["time"].max() - group["time"].min()).total_seconds()
        if duration > 40 * 60:
            curve = compute_power_curve(group, DURATIONS)
            ftp = compute_ftp_from_curve(curve)
            if not np.isnan(ftp):
                session_dates.append(group["time"].min())
                session_ftps.append(ftp)
                
    if session_ftps:
        sorted_pairs = sorted(zip(session_dates, session_ftps))
        dates = [p[0] for p in sorted_pairs]
        ftps = np.array([p[1] for p in sorted_pairs])

        # ---- plot raw points ----
        ax.plot(dates, ftps, marker='o', linestyle='none', color='purple', label="FTP")

        # ---- smoothing spline ----
        if len(ftps) > 3:  # spline needs enough points
            x_numeric = mdates.date2num(dates)

            # smoothing factor: adjust if too wiggly or too flat
            spline = UnivariateSpline(x_numeric, ftps, s=len(ftps)*1000)

            x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 300)
            y_smooth = spline(x_smooth)

            ax.plot(
                mdates.num2date(x_smooth),
                y_smooth,
                color='purple',
                linewidth=2,
                alpha=0.8,
                label="Smoothed trend"
            )

        ax.set_title("FTP Progression")
        ax.set_ylabel("FTP (W)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.4)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    df = load_all_tcx(TCX_PATTERN)

    # all time
    curve_all = compute_power_curve(df, DURATIONS)

    # last 7 days
    last_week = df[df["time"] > df["time"].max() - pd.Timedelta(days=7)]
    curve_week = compute_power_curve(last_week, DURATIONS)

    ftp_all = np.floor(compute_ftp_from_curve(curve_all))
    ftp_all_model = np.floor(compute_model_based_ftp(curve_all))
    ftp_week = np.floor(compute_ftp_from_curve(curve_week))
    ftp_week_model = np.floor(compute_model_based_ftp(curve_week))
    

    print("All-time power curve:")
    print("FTP:", ftp_all)
    print("FTP (model):", ftp_all_model)
    print("\nLast-week power curve:")
    print("FTP:", ftp_week)
    print("FTP (model):", ftp_week_model)

    # plot both
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.fill_between(curve_all["duration_s"], curve_all["best_avg_power"], color = "lightblue",
             label="All time")
    ax1.fill_between(curve_week["duration_s"], curve_week["best_avg_power"], color = "orange",
             alpha = 0.2, label="Last 7 days")

    best_score = -np.inf
    best_level = None
    best_curve = None

    for level, pdata in coggan_profiles_wkg.items():
        smooth_vals, _ = fit_morton_model(pdata, curve_all["duration_s"])
        score = score_templates(curve_all["best_avg_power"], smooth_vals*WEIGHT)
        ax1.plot(
            curve_all["duration_s"],
            smooth_vals*WEIGHT,
            lw=1,
            alpha=score,
            color = "grey"
        )
        ax1.text(
            curve_all["duration_s"][0],
            smooth_vals[0]*WEIGHT,
            str(level),
            fontsize=9,
            alpha = score,
            ha="left",
            va="center",
            color="black"
        )
        if score > best_score:
            best_score = score
            best_level = level
            best_curve = smooth_vals

    if best_curve is not None:
        x0 = curve_all["duration_s"][0]
        y0 = best_curve[0]*WEIGHT

        ax1.text(
            x0,
            y0,
            str(best_level),
            fontsize=9,
            ha="left",
            va="center",
            color="black"
        )
    

    ax1.set_xscale("log")
    ax1.set_xlabel("Duration (s)")
    ax1.set_ylabel("Best avg power (W)")
    ax1.set_title("Power-duration curve (all-time vs last-week)")
    ax1.grid(True, which="both", alpha=0.4)
    ax1.legend()

    plot_ftp_progression(df, ax2)

    plt.tight_layout()
    plt.show()