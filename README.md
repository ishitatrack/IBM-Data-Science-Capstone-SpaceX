# IBM Data Science Capstone: SpaceX

This repository contains an end-to-end data science project exploring SpaceX
Falcon 9 launches and predicting first-stage landing outcomes. It covers data
collection and preparation, exploratory analysis, SQL queries, interactive
mapping, a Dash dashboard, and supervised machine-learning models.

## 1. Project Overview & Objective

The objective is to analyze Falcon 9 launch and first-stage recovery data, then
build classification models that predict whether a first stage lands
successfully. Launch outcomes can help assess launch costs and support
data-driven comparisons between launch sites, payloads, and mission profiles.

The analysis uses the SpaceX REST API (`https://api.spacexdata.com/v4`) for
launches, payloads, and launchpads. Some workflows also read a processed CSV
dataset supplied separately.

## 2. Repository Structure

```text
IBM-Data-Science-Capstone-SpaceX/
├── README.md
├── spacex_launch_eda.py          # API data preparation, EDA, plots, SQLite queries, Folium map
├── spacex_launch_dashboard.py    # Dash app and interactive callbacks
├── spacex_dash_app.py            # Requested command-line entry point for the Dash app
├── spacex_landing_prediction.py  # Feature preparation, model tuning, evaluation
├── dataset_part_2.csv            # Processed dataset expected by default by the ML script
├── spacex_launch_eda.csv         # Generated cleaned launch data
├── payload_mass_by_launch_site.png
├── success_rate_by_launch_site.png
├── spacex_launch_sites_map.html
└── spacex_confusion_matrices.png # Generated model evaluation plot
```

The Python scripts are included in this repository. Notebooks and processed or
generated data files are not currently checked in; the data/output filenames
above describe the inputs expected or files generated when the corresponding
scripts are run. Add the processed CSV before running the machine-learning
script. The EDA and dashboard scripts fetch current data from the SpaceX API.

## 3. Methodology Summary

### Data Collection

The EDA and dashboard scripts request launch, payload, and launchpad records
from the SpaceX v4 REST API using `requests`. HTTP errors and unexpected
response shapes are raised rather than silently ignored.

### Data Wrangling

Nested launch records are flattened into launch-level rows. Payload IDs are
joined to payload masses, launchpad IDs are mapped to launch-site names, dates
are parsed as UTC timestamps, and unavailable values are handled as missing.
Landing outcomes are represented by a binary `Class`/`class` target.

### Exploratory Data Analysis

`spacex_launch_eda.py` creates:

- A scatter plot of payload mass by launch site, colored by landing outcome.
- A bar chart of landing success rate by launch site.
- A cleaned launch CSV for downstream analysis.

### SQL

The EDA script loads launch sites, core landing attempts, and payload/customer
records into an in-memory SQLite database. It queries unique launch sites,
successful landings per site, and average payload mass for NASA (CRS) payloads.

### Interactive Mapping

The EDA script uses Folium to create an HTML map centered on the launch region.
It marks CCAFS SLC-40, KSC LC-39A, and VAFB SLC-4E, with separate green
successful-landing and red failed-landing layers.

### Dash Dashboard

`spacex_dash_app.py` runs the Dash application implemented in
`spacex_launch_dashboard.py`. Its launch-site dropdown is populated from the
fetched data, and its payload-mass range slider filters the visualizations.
Selecting “All Sites” displays a launch outcome pie chart; selecting a
particular site displays a success/failure scatter plot against payload mass.

### Machine Learning

`spacex_landing_prediction.py` loads a processed dataset, one-hot encodes
categorical launch features such as `LaunchSite`, `Orbit`/`Orbits`, and
`Serial`, then uses a stratified 80/20 train-test split. `GridSearchCV` tunes
Logistic Regression, Support Vector Classifier, Decision Tree, and
K-Nearest Neighbors models. Test accuracy is reported for each tuned model,
and confusion matrices are saved for comparison.

## 4. Key Results & Model Performance

The project reports **83.3% accuracy across the classification models**.
Model performance can vary with the processed dataset and train/test split.
The training script prints the measured test accuracy and best parameters for
each model, and saves a confusion-matrix figure to help compare predictions
and errors.

## 5. Run the Dash App Locally

Use Python 3.9 or newer. From the repository root, install the required
packages:

```bash
python -m pip install dash plotly pandas requests
```

Then start the dashboard:

```bash
python spacex_dash_app.py
```

Open the local address printed in the terminal (typically
`http://127.0.0.1:8050`). The app fetches its data from the SpaceX API on
startup, so an internet connection and an available API are required.

### Run the other analyses

Install their additional dependencies as needed:

```bash
python -m pip install matplotlib seaborn folium scikit-learn
```

Generate the EDA plots, CSV, SQLite query output, and interactive map:

```bash
python spacex_launch_eda.py
```

Train the classifiers using the default processed CSV filename:

```bash
python spacex_landing_prediction.py
```

Or provide the dataset path explicitly:

```bash
python spacex_landing_prediction.py path/to/processed_dataset.csv
```
