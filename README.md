# Driving Style Analyzer

A Python-based tool for analyzing driving behavior using public datasets and simulated data. This project processes driving data to identify driving modes, calculate trip statistics, and visualize driving patterns.

## Features

- **Data Loading**: Load and preprocess driving data from CSV files
- **Driving Mode Detection**: Identify stop, acceleration, deceleration, and cruise modes based on speed and acceleration thresholds
- **Trip Analysis**: Segment driving data into individual trips and calculate statistics (duration, distance, average speed, etc.)
- **Visualization**: Create comprehensive plots showing speed profiles, acceleration patterns, trip statistics, and driving mode distributions
- **Modular Architecture**: Well-structured codebase with separate modules for data loading, mode detection, trip analysis, and visualization

## Project Structure

```
Driving_style_analyzer/
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── src/                     # Source code
│   ├── data_loader.py      # Load and preprocess data
│   ├── mode_detector.py    # Detect driving modes
│   ├── trip_analyzer.py    # Identify trips and calculate statistics
│   ├── visualizer.py       # Create visualizations
│   └── main.py             # Main analysis pipeline
├── Mock/                    # Mock data and analysis
│   ├── 01_Mock_Analysis.py # Simulated driving data analysis
│   └── 01_Mock_Analysis.ipynb
├── data/                    # Data files
│   └── udds.csv            # UDDS driving cycle data
├── output/                  # Generated outputs
│   ├── udds_metric_analysis.png
│   └── udds_detailed_statistics.png
└── venv/                   # Python virtual environment (ignored)
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Driving_style_analyzer.git
   cd Driving_style_analyzer
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   Option 1: Install individual packages:
   ```bash
   pip install pandas numpy matplotlib
   ```

   Option 2: Install from requirements file:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Main Analysis Pipeline

Execute the main analysis script to process the UDDS driving cycle data:

```bash
cd src
python main.py
```

This will:
1. Load the UDDS driving data from `data/udds.csv`
2. Detect driving modes (stop, cruise, acceleration, deceleration)
3. Identify individual trips and calculate statistics
4. Generate visualizations saved to `output/` directory
5. Display key metrics in the terminal

### Running Mock Analysis

For a quick demonstration with simulated data:

```bash
cd Mock
python 01_Mock_Analysis.py
```

### Using Individual Modules

You can also import and use individual modules in your own Python scripts:

```python
from src.data_loader import load_data
from src.mode_detector import detect_driving_modes
from src.trip_analyzer import identify_trips, calculate_trip_statistics
from src.visualizer import plot_main_analysis

# Load data
df = load_data("../data/udds.csv")

# Detect driving modes
df = detect_driving_modes(df)

# Identify trips
df = identify_trips(df)
trips_df, summary = calculate_trip_statistics(df)

# Create visualization
fig = plot_main_analysis(df, trips_df, save_path="analysis.png")
```

## Output Examples

The analysis generates two main visualizations:

1. **Main Analysis Plot** (`udds_metric_analysis.png`):
   - Speed profile with driving modes color-coded
   - Acceleration profile with threshold lines
   - Cumulative distance over time

2. **Detailed Statistics Plot** (`udds_detailed_statistics.png`):
   - Speed and acceleration distributions
   - Trip durations and distances
   - Driving mode statistics

## Technology Stack

| Category | Tools & Resources |
| :--- | :--- |
| **Language** | Python 3.x |
| **Data Analysis** | pandas, numpy |
| **Visualization** | matplotlib |
| **Version Control** | Git & GitHub |
| **Datasets** | UDDS (EPA Standard Driving Cycle) |
| **Development** | Virtual Environments, Jupyter Notebooks |

## Key Metrics Analyzed

- **Speed Analysis**: Maximum, minimum, and average speed
- **Acceleration Analysis**: Hard acceleration (>3 km/h/s) and hard braking (<-3 km/h/s) events
- **Trip Statistics**: Number of trips, total distance, average trip duration
- **Driving Mode Distribution**: Percentage of time spent in stop, cruise, acceleration, and deceleration modes
- **Aggressiveness Score**: Metric based on frequency of hard acceleration/braking events

## Current Status

- [x] Create repository and set up Python environment
- [x] Implement mock data analysis with simulated driving data
- [x] Develop modular analysis pipeline (data loading, mode detection, trip analysis, visualization)
- [x] Analyze real UDDS driving cycle data
- [x] Generate comprehensive visualizations
- [ ] Add support for additional driving datasets (FTP-75, etc.)
- [ ] Implement machine learning for driving style classification
- [ ] Add real-time OBD-II data integration

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
