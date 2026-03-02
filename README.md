# Cycle-Wattslab 🚴‍♂️⚡

A Python-based tool for analyzing cycling power profiles using TCX data. It calculates power-duration curves, estimates FTP (Functional Threshold Power), and tracks historical performance trends.

## Features

-   **Power-Duration Curve**: Automatically calculates your best average power for various durations (e.g., 1s, 5s, 1min, 5min, 20min, etc.).
-   **FTP Estimation**:
    -   **Classic**: 95% of your 20-minute best average power.
    -   **Model-Based**: Morton's 3-parameter model (extrapolated power for 60 minutes).
-   **Performance Benchmarking**: Compares your power-to-weight ratio against Coggan power profiles (e.g., "Good", "Excellent", "Pro").
-   **FTP Progression**: Visualizes how your estimated FTP has changed over time across multiple training sessions.
-   **Time-Based Comparison**: Compare your current week's performance against your all-time records.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/HeymansAdrien/Cycle-Wattslab.git
    cd Cycle-Wattslab
    ```

2.  **Install dependencies**:
    You'll need Python installed. Install the required libraries using pip:
    ```bash
    pip install pandas numpy scipy matplotlib
    ```

## Usage

1.  **Prepare your data**:
    Collect your `.tcx` files from your cycling head unit or training platform (e.g., Garmin Connect, Kinomap, etc.). Place them in the `data/` folder in the root of this project.

2.  **Configure weight**:
    The script uses Body Weight (kg) for benchmarks. Open `power_curve.py` and update the `WEIGHT` variable (line 13) with your own weight in kilograms:
    ```python
    WEIGHT = 70  # Replace with your weight in kilograms
    ```

3.  **Run the script**:
    Execute the script from your terminal:
    ```bash
    python power_curve.py
    ```

## Understanding the Output

### Terminal
The script outputs your Estimated FTP (Functional Threshold Power) for:
-   **All-time**: Based on all `.tcx` files in the `data/` folder.
-   **Last 7 days**: Based on sessions within the last week.

### Visualizations
A window will open with two subplots:
1.  **Power-Duration Curve (Left)**:
    -   **Blue fill**: Represents your all-time personal records for each duration.
    -   **Orange overlay**: Represents your best efforts from the last 7 days.
    -   **Grey lines/labels**: Coggan performance levels used as benchmarks for your power-to-weight ratio.
2.  **FTP Progression (Right)**:
    -   **Purple dots**: Estimated FTP for individual sessions (minimum 40 minutes duration).
    -   **Solid line**: A smoothed trend showing your FTP progression over time.

## Project Structure

-   `power_curve.py`: The core analysis and visualization script.
-   `data/`: Place your training files here. Support for `.tcx` format.
-   `LICENSE`: MIT License.

## Acknowledgments
This project implements performance profiling concepts and Morton's equation from:
-   [Allen, H., Coggan, A., & McGregor, S. (2019). Training and Racing with a Power Meter 3rd edition. Velopress](https://books.google.be/books?id=X62SDwAAQBAJ)
-   [Morton, R. H. (2006). The critical power and related whole-body bioenergetic models. European journal of applied physiology, 96(4), 339-354.](10.1007/s00421-005-0088-2)