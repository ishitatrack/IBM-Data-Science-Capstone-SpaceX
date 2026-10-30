"""Interactive SpaceX launch dashboard with site and payload-mass filters."""

from typing import Any

import dash
import pandas as pd
import plotly.express as px
import requests
from dash.dependencies import Input, Output

try:
    import dash_core_components as dcc
    import dash_html_components as html
except ModuleNotFoundError:
    # Dash 2+ provides these components directly; keep compatibility with older Dash.
    from dash import dcc, html


API_BASE = "https://api.spacexdata.com/v4"
REQUEST_TIMEOUT = 30


def fetch_collection(endpoint: str) -> list[dict[str, Any]]:
    """Fetch a SpaceX API collection, raising errors for failed or invalid responses."""
    response = requests.get(f"{API_BASE}/{endpoint}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in the response from /{endpoint}.")
    return data


def build_launch_dataframe(
    launches: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    launchpads: list[dict[str, Any]],
) -> pd.DataFrame:
    """Flatten launch, payload, and launchpad records for dashboard plotting."""
    payload_mass_by_id = {
        payload.get("id"): payload.get("mass_kg")
        for payload in payloads
        if payload.get("id")
    }
    site_name_by_id = {
        launchpad.get("id"): launchpad.get("name")
        for launchpad in launchpads
        if launchpad.get("id")
    }

    records = []
    for launch in launches:
        masses = [
            payload_mass_by_id.get(payload_id)
            for payload_id in launch.get("payloads") or []
            if payload_mass_by_id.get(payload_id) is not None
        ]
        cores = [
            core for core in (launch.get("cores") or [])
            if isinstance(core, dict)
        ]
        successful = any(
            core.get("landing_success") is True
            and core.get("landing_type") in {"RTLS", "ASDS"}
            for core in cores
        )

        launchpad_id = launch.get("launchpad")
        records.append(
            {
                "FlightNumber": launch.get("flight_number"),
                "Date": launch.get("date_utc"),
                "LaunchSite": site_name_by_id.get(launchpad_id) or launchpad_id,
                "PayloadMass": sum(masses) if masses else None,
                "class": int(successful),
            }
        )

    df = pd.DataFrame.from_records(
        records,
        columns=["FlightNumber", "Date", "LaunchSite", "PayloadMass", "class"],
    )
    if df.empty:
        raise ValueError("The SpaceX API returned no launch records.")

    df["FlightNumber"] = pd.to_numeric(df["FlightNumber"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["PayloadMass"] = pd.to_numeric(df["PayloadMass"], errors="coerce")
    df["LaunchSite"] = df["LaunchSite"].replace("", pd.NA)
    df["class"] = df["class"].astype("int8")
    return df


def load_launch_data() -> pd.DataFrame:
    """Fetch the collections needed to populate the dashboard."""
    return build_launch_dataframe(
        launches=fetch_collection("launches"),
        payloads=fetch_collection("payloads"),
        launchpads=fetch_collection("launchpads"),
    )


df = load_launch_data()
valid_masses = df["PayloadMass"].dropna()
if valid_masses.empty:
    raise ValueError("No payload mass values were returned; cannot configure the slider.")

slider_min = float(valid_masses.min())
slider_max = float(valid_masses.max())
if slider_min == slider_max:
    slider_max = slider_min + 1

launch_sites = sorted(df["LaunchSite"].dropna().astype(str).unique())
site_options = [{"label": "All Sites", "value": "ALL"}] + [
    {"label": site, "value": site} for site in launch_sites
]

app = dash.Dash(__name__)
app.title = "SpaceX Launch Records Dashboard"
app.layout = html.Div(
    children=[
        html.H1("SpaceX Launch Records Dashboard"),
        html.Label("Select a launch site:"),
        dcc.Dropdown(
            id="site-dropdown",
            options=site_options,
            value="ALL",
            clearable=False,
        ),
        html.Label("Payload mass range (kg):"),
        dcc.RangeSlider(
            id="payload-slider",
            min=slider_min,
            max=slider_max,
            step=max((slider_max - slider_min) / 100, 1),
            value=[slider_min, slider_max],
            marks={
                int(slider_min): f"{slider_min:,.0f}",
                int(slider_max): f"{slider_max:,.0f}",
            },
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        dcc.Graph(id="success-pie-chart"),
    ],
    style={"maxWidth": "1100px", "margin": "auto", "padding": "24px"},
)


@app.callback(
    Output("success-pie-chart", "figure"),
    Input("site-dropdown", "value"),
    Input("payload-slider", "value"),
)
def update_graph(selected_site: str, payload_range: list[float]):
    """Update the all-site outcomes pie or selected-site mass scatter plot."""
    low, high = payload_range
    filtered = df[
        df["PayloadMass"].between(low, high, inclusive="both")
    ].copy()

    if selected_site == "ALL":
        outcome_counts = (
            filtered["class"]
            .map({1: "Successful", 0: "Unsuccessful"})
            .value_counts()
            .reindex(["Successful", "Unsuccessful"], fill_value=0)
            .rename_axis("Outcome")
            .reset_index(name="Launches")
        )
        return px.pie(
            outcome_counts,
            names="Outcome",
            values="Launches",
            title="Launch Outcomes for All Sites",
            color="Outcome",
            color_discrete_map={
                "Successful": "green",
                "Unsuccessful": "red",
            },
        )

    site_data = filtered[filtered["LaunchSite"] == selected_site]
    plot_data = site_data.dropna(subset=["PayloadMass"]).copy()
    plot_data["Outcome"] = plot_data["class"].map(
        {1: "Successful", 0: "Unsuccessful"}
    )
    return px.scatter(
        plot_data,
        x="PayloadMass",
        y="class",
        color="Outcome",
        hover_data=["FlightNumber", "Date"],
        category_orders={"Outcome": ["Unsuccessful", "Successful"]},
        color_discrete_map={"Successful": "green", "Unsuccessful": "red"},
        title=f"Launch Outcomes by Payload Mass at {selected_site}",
        labels={
            "PayloadMass": "Payload Mass (kg)",
            "class": "Landing Success (0 = No, 1 = Yes)",
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
