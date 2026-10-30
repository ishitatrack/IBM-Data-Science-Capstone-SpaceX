"""Fetch SpaceX launch data, create a cleaned DataFrame, and save EDA outputs."""

from pathlib import Path
import html
import sqlite3
from typing import Any

import folium
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

API_BASE = "https://api.spacexdata.com/v4"
OUTPUT_DIR = Path(__file__).resolve().parent
TIMEOUT_SECONDS = 30
SUCCESSFUL_LANDING_TYPES = {"ASDS", "RTLS"}
LAUNCH_SITE_LOCATIONS = {
    "CCAFS SLC-40": (28.5618571, -80.577366),
    "KSC LC-39A": (28.6080585, -80.604046),
    "VAFB SLC-4E": (34.632834, -120.610745),
}
LAUNCH_SITE_ALIASES = {
    "CCAFSSLC40": "CCAFS SLC-40",
    "CCSFSSLC40": "CCAFS SLC-40",
    "CCSFSLC40": "CCAFS SLC-40",
    "KSCLC39A": "KSC LC-39A",
    "VAFBSLC4E": "VAFB SLC-4E",
}


def fetch_collection(endpoint: str) -> list[dict[str, Any]]:
    """Fetch one SpaceX API collection and verify the response shape."""
    response = requests.get(
        f"{API_BASE}/{endpoint}",
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Expected a list from the {endpoint} endpoint.")
    return data


def landing_succeeded(core: dict[str, Any]) -> bool:
    """Match successful RTLS/ASDS landings, including legacy outcome strings."""
    if core.get("landing_success") is True:
        landing_type = core.get("landing_type")
        return isinstance(landing_type, str) and landing_type.upper() in SUCCESSFUL_LANDING_TYPES

    outcome = core.get("outcome")
    if isinstance(outcome, str):
        normalized = outcome.strip().upper()
        return normalized == "TRUE" or any(
            normalized == f"TRUE {landing_type}"
            for landing_type in SUCCESSFUL_LANDING_TYPES
        )
    return False


def build_launch_dataframe(
    launches: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    launchpads: list[dict[str, Any]],
) -> pd.DataFrame:
    """Flatten API records into one analysis row per launch."""
    payload_mass_by_id = {
        payload.get("id"): payload.get("mass_kg")
        for payload in payloads
        if payload.get("id")
    }
    launch_site_by_id = {
        launchpad.get("id"): launchpad.get("name")
        for launchpad in launchpads
        if launchpad.get("id")
    }

    rows = []
    for launch in launches:
        payload_ids = launch.get("payloads") or []
        masses = [
            payload_mass_by_id.get(payload_id)
            for payload_id in payload_ids
            if payload_mass_by_id.get(payload_id) is not None
        ]
        cores = [core for core in (launch.get("cores") or []) if isinstance(core, dict)]

        rows.append(
            {
                "FlightNumber": launch.get("flight_number"),
                "Date": launch.get("date_utc"),
                "RocketID": launch.get("rocket"),
                "PayloadIDs": payload_ids or pd.NA,
                "PayloadMass": sum(masses) if masses else pd.NA,
                "LaunchSite": launch_site_by_id.get(launch.get("launchpad"))
                or launch.get("launchpad")
                or pd.NA,
                "CoreLandingOutcomes": [
                    {
                        "core_id": core.get("core"),
                        "landing_attempt": core.get("landing_attempt"),
                        "landing_success": core.get("landing_success"),
                        "landing_type": core.get("landing_type"),
                        "landpad_id": core.get("landpad"),
                        "outcome": core.get("outcome"),
                    }
                    for core in cores
                ]
                or pd.NA,
                "Class": int(any(landing_succeeded(core) for core in cores)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("The launches endpoint returned no launch records.")

    df["FlightNumber"] = pd.to_numeric(df["FlightNumber"], errors="coerce").astype("Int64")
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["PayloadMass"] = pd.to_numeric(df["PayloadMass"], errors="coerce")
    df["Class"] = df["Class"].astype("int8")

    for column in ("RocketID", "LaunchSite"):
        df[column] = df[column].replace("", pd.NA)

    return df


def create_eda_outputs(df: pd.DataFrame) -> None:
    """Save the cleaned launch data and the requested visualizations."""
    sns.set_theme(style="whitegrid")

    scatter_data = df.dropna(subset=["PayloadMass", "LaunchSite"]).copy()
    scatter_data["Class"] = scatter_data["Class"].astype(str)
    ax = sns.scatterplot(
        data=scatter_data,
        x="LaunchSite",
        y="PayloadMass",
        hue="Class",
        hue_order=["0", "1"],
        palette={"0": "tomato", "1": "seagreen"},
        alpha=0.75,
    )
    ax.set(
        title="Payload Mass by Launch Site and Landing Outcome",
        xlabel="Launch Site",
        ylabel="Payload Mass (kg)",
    )
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "payload_mass_by_launch_site.png", dpi=150)
    plt.close()

    success_rates = (
        df.dropna(subset=["LaunchSite"])
        .groupby("LaunchSite", as_index=False)
        .agg(success_rate=("Class", "mean"))
        .sort_values("success_rate", ascending=False)
    )
    ax = sns.barplot(
        data=success_rates,
        x="LaunchSite",
        y="success_rate",
        color="steelblue",
    )
    ax.set(
        title="Landing Success Rate by Launch Site",
        xlabel="Launch Site",
        ylabel="Success Rate",
        ylim=(0, 1),
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "success_rate_by_launch_site.png", dpi=150)
    plt.close()

    df.to_csv(OUTPUT_DIR / "spacex_launch_eda.csv", index=False)
    print(f"Saved {len(df)} launch rows and EDA charts to {OUTPUT_DIR}")


def create_launch_map(df: pd.DataFrame) -> Path:
    """Create an interactive map with launch sites and success/failure markers."""
    class_column = next(
        (column for column in df.columns if column.casefold() == "class"),
        None,
    )
    if class_column is None or "LaunchSite" not in df.columns:
        raise ValueError("The launch DataFrame must contain Class and LaunchSite columns.")

    launch_map = folium.Map(location=[29.559684, -95.083097], zoom_start=5)

    for site_name, coordinates in LAUNCH_SITE_LOCATIONS.items():
        folium.map.Marker(
            location=coordinates,
            popup=site_name,
            tooltip=site_name,
            icon=folium.Icon(color="blue", icon="rocket", prefix="fa"),
        ).add_to(launch_map)
        folium.Circle(
            location=coordinates,
            radius=1000,
            color="blue",
            fill=True,
            fill_opacity=0.15,
            tooltip=f"{site_name} launch area",
        ).add_to(launch_map)

    success_group = folium.FeatureGroup(name="Successful landings (green)")
    failure_group = folium.FeatureGroup(name="Failed landings (red)")
    site_coordinates = {
        normalized: LAUNCH_SITE_LOCATIONS[canonical]
        for normalized, canonical in LAUNCH_SITE_ALIASES.items()
    }

    for row in df.to_dict(orient="records"):
        site = row.get("LaunchSite")
        if not isinstance(site, str):
            continue
        normalized_site = "".join(character for character in site.upper() if character.isalnum())
        coordinates = site_coordinates.get(normalized_site)
        if coordinates is None:
            continue

        outcome = pd.to_numeric(pd.Series([row.get(class_column)]), errors="coerce").iloc[0]
        if pd.isna(outcome):
            continue
        successful = bool(outcome)
        group = success_group if successful else failure_group
        color = "green" if successful else "red"
        flight_number = html.escape(str(row.get("FlightNumber", "unknown")))
        site_label = html.escape(site)
        folium.map.Marker(
            location=coordinates,
            popup=f"Flight {flight_number}<br>Site: {site_label}<br>"
            f"Outcome: {'Success' if successful else 'Failure'}",
            tooltip=f"Flight {flight_number}: {'success' if successful else 'failure'}",
            icon=folium.Icon(color=color),
        ).add_to(group)

    success_group.add_to(launch_map)
    failure_group.add_to(launch_map)
    folium.LayerControl().add_to(launch_map)

    map_path = OUTPUT_DIR / "spacex_launch_sites_map.html"
    launch_map.save(map_path)
    print(f"Saved interactive launch map to {map_path}")
    return map_path


def run_sql_queries(
    df: pd.DataFrame,
    launches: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> None:
    """Load launch, landing, and customer payload data into SQLite and query it."""
    payload_by_id = {
        payload.get("id"): payload
        for payload in payloads
        if payload.get("id")
    }

    landing_rows = []
    for row in df.itertuples(index=False):
        core_outcomes = row.CoreLandingOutcomes
        if not isinstance(core_outcomes, list):
            continue
        for core in core_outcomes:
            landing_rows.append(
                {
                    "LaunchSite": row.LaunchSite,
                    "LandingSuccess": int(landing_succeeded(core)),
                }
            )

    payload_rows = []
    customer_rows = []
    for launch in launches:
        for payload_id in launch.get("payloads") or []:
            payload = payload_by_id.get(payload_id)
            if payload is None or payload.get("mass_kg") is None:
                continue
            payload_rows.append(
                {
                    "PayloadID": payload_id,
                    "PayloadMass": payload["mass_kg"],
                }
            )
            customers = payload.get("customers") or []
            if isinstance(customers, str):
                customers = [customers]
            for customer in customers:
                if isinstance(customer, str):
                    customer_rows.append(
                        {"PayloadID": payload_id, "Customer": customer}
                    )

    launch_sites = df[["LaunchSite"]].dropna().drop_duplicates()
    landing_df = pd.DataFrame(
        landing_rows,
        columns=["LaunchSite", "LandingSuccess"],
    ).dropna(subset=["LaunchSite"])
    payload_df = pd.DataFrame(payload_rows, columns=["PayloadID", "PayloadMass"])
    payload_df = payload_df.drop_duplicates(subset=["PayloadID"])
    customers_df = pd.DataFrame(
        customer_rows,
        columns=["PayloadID", "Customer"],
    ).drop_duplicates()

    with sqlite3.connect(":memory:") as connection:
        launch_sites.to_sql("launch_sites", connection, index=False)
        landing_df.to_sql("landing_attempts", connection, index=False)
        payload_df.to_sql("payloads", connection, index=False)
        customers_df.to_sql("payload_customers", connection, index=False)

        unique_sites = pd.read_sql_query(
            """
            SELECT DISTINCT LaunchSite
            FROM launch_sites
            ORDER BY LaunchSite
            """,
            connection,
        )
        successful_landings = pd.read_sql_query(
            """
            SELECT s.LaunchSite,
                   COALESCE(SUM(l.LandingSuccess), 0) AS SuccessfulLandings
            FROM launch_sites AS s
            LEFT JOIN landing_attempts AS l ON l.LaunchSite = s.LaunchSite
            GROUP BY s.LaunchSite
            ORDER BY s.LaunchSite
            """,
            connection,
        )
        nasa_crs_average = pd.read_sql_query(
            """
            SELECT AVG(p.PayloadMass) AS AveragePayloadMassKg
            FROM payloads AS p
            INNER JOIN payload_customers AS c ON c.PayloadID = p.PayloadID
            WHERE c.Customer = 'NASA (CRS)'
            """,
            connection,
        )

    print("\nUnique launch sites:")
    print(unique_sites.to_string(index=False))
    print("\nSuccessful landings by launch site:")
    print(successful_landings.to_string(index=False))
    print("\nAverage payload mass for NASA (CRS) payloads:")
    print(nasa_crs_average.to_string(index=False))


def main() -> None:
    launches = fetch_collection("launches")
    payloads = fetch_collection("payloads")
    launchpads = fetch_collection("launchpads")
    df = build_launch_dataframe(launches, payloads, launchpads)
    create_eda_outputs(df)
    create_launch_map(df)

    print("\nSuccess rate by launch site:")
    print(
        df.groupby("LaunchSite", dropna=False)["Class"]
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )
    run_sql_queries(df, launches, payloads)


if __name__ == "__main__":
    main()
