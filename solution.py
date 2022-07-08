from datetime import date
from time import time

import cudf
import dash
import dash_daq as daq
import dash_html_components as html
import dash_core_components as dcc
import dask_cudf
from jupyter_dash import JupyterDash
import numpy as np
import pandas as pd
import plotly.graph_objects as go


verbose = True

# ---- Read Data ----
start_time = time()
if verbose:
    print("Loading data...")
    
# TODO: Update to read all CSVs
df = dask_cudf.read_csv(
    "data/*.csv",
    usecols=["STATION", "LATITUDE", "LONGITUDE", "DlySum", "DATE"],
    dtype={
        "STATION": "object",
        "LATITUDE": np.float32,
        "LONGITUDE": np.float32,
        "DlySum": np.uint32,
        "DATE": str,
    },
    na_values=["-9999"],
)
df = df.dropna()
if verbose:
    print(df.head())

# Define columns
df = df.compute()
seconds_elapsed = time() - start_time
if verbose:
    print(seconds_elapsed, "seconds")
start_time = time()
if verbose:
    print("Data loaded. Defining columns...")
df["Inches"] = df["DlySum"] / 100
df["TEXT"] = df["STATION"] + " - " + df["Inches"].astype(str) + " inches"

# Set index for faster querying
date_min = df["DATE"].min()
date_max = df["DATE"].max()
df = df.set_index("DATE")

# Calculate color across all dataframes each day has same max
df["COLOR"] = np.log10(df["Inches"] + 1)
cmax = df["COLOR"].max()
 # TODO: Delete this, and add one to each figure method

# ---- Define Dash App ----
seconds_elapsed = time() - start_time
if verbose:
    print(seconds_elapsed, "seconds")
start_time = time()
if verbose:
    print("Data computed. Defining Dash App...")

bgcolor = "rgb(25, 26, 26)"
ftcolor = "rgb(200, 200, 200)"
paper_bgcolor = "rgb(52, 51, 50)"
subunitcolor = "rgb(73, 73, 73)"

initial_date = date(2021, 1, 1)

app = JupyterDash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.Span(
                    [
                        dcc.DatePickerSingle(
                            id="my-date-picker-single",
                            min_date_allowed=date_min,
                            max_date_allowed=date_max,
                            initial_visible_month=initial_date,
                            date=initial_date,
                        ),
                        daq.BooleanSwitch(
                            id="show-zeros",
                            on=True,
                            label="Show Zeros",
                            labelPosition="bottom",
                            style={"margin": "auto"},
                        ),
                    ],
                    style={
                        "width": "132px",
                        "display": "flex",
                        "flex-direction": "column",
                    },
                ),
                dcc.Markdown(
                    """
            # USA Precipitation Dashboard
            Welcome to the Preciptiation Dashboard based on NOAA's Hourly Precipitation Data.
            * Click a station on the map below to see its precipitation history from 1940 to 2020.
            * Click the date in the top left in order to change it. This can similarly be done by clicking a day
            on the time series below.
            * There are many days without any preciptiation. Click the toggle on the left to remove them from
            the map.
            """,
                    style={"padding": "0 35px 20px 20px"},
                ),
            ],
            style={"display": "flex", "flex-direction": "row"},
        ),
        dcc.Graph(
            id="precipitation-map",
            config={"modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        ),
        dcc.Graph(id="time-series"),
    ],
    style={"color": ftcolor, "background-color": paper_bgcolor},
)

# ---- Define Callbacks ----
seconds_elapsed = time() - start_time
if verbose:
    print(seconds_elapsed, "seconds")
start_time = time()
if verbose:
    print("App defined. Creating Callbacks...")

plot_height = 250
plot_margin = {"r": 50, "t": 37, "l": 50, "b": 8}


def update_map(date_value, show_zeros):
    """Updates the precipitation map.

    Args:
        date_value: as selected by "my-date-picker-single". Map renders
            precipitation for the selected day.
        show_zeros(bool): as selected by "show-zeros". If TRUE show zero
            precipitation locations.

    Returns:
        A Figure - A map of the US with precipitation.
    """
    ddf = df[df.index == date_value]
    ddf = ddf if show_zeros else ddf[ddf["Inches"] != 0]
    ddf = ddf.to_pandas()
    fig = go.Figure(
        [
            go.Scattergeo(
                lon=ddf["LONGITUDE"],
                lat=ddf["LATITUDE"],
                customdata=ddf["STATION"],
                mode="markers",
                marker_color=ddf["Inches"],
                marker=dict(
                    size=4,
                    reversescale=True,
                    autocolorscale=False,
                    colorscale="Blues",
                    cmin=0,
                    color=ddf["COLOR"],
                    cmax=cmax,
                    colorbar=dict(
                        title="Inches",
                        tickvals=[0.1, 0.4, 0.699, 1, 1.3],
                        ticktext=["0", ".25", ".5", "1", "2"],
                        x=0,
                    ),
                ),
                text=ddf["TEXT"],
            )
        ]
    )

    fig.update_layout(
        title="USA Precipitation for " + str(date_value),
        font_color=ftcolor,
        paper_bgcolor=paper_bgcolor,
        geo=dict(
            scope="north america",
            bgcolor=bgcolor,
            landcolor="rgb(37, 37, 37)",
            showsubunits=True,
            subunitcolor=subunitcolor,
            lakecolor=bgcolor,
            lonaxis_range=[-165.0, -55.0],
            lataxis_range=[25.0, 45.0],
        ),
        height=plot_height,
        margin=plot_margin,
    )

    return fig


def update_series(click_data):
    """Updates the precipitation time series for the selected station.

    Args:
        click_data: information relating to the station the user clicked
            on the precipitation map (rendered in update_map)
    Returns:
        A Figure - A time series of the preciptiation for the selected station.
    """
    
    station_name = "AQC00914902"
    if click_data:  # click_data is None on app startup
        station_name = click_data["points"][0]["customdata"]
    # TODO: Filter df with station_name and convert to ploty readable format
    ddf = df[df["STATION"]==station_name]
    #ddf = df  # Delete this line
    ddf = ddf.to_pandas()
    fig = go.Figure(
        [
            go.Scatter(
                x=ddf.index,
                y=ddf["Inches"],
                customdata=ddf.index,
                text=ddf["TEXT"],
                line=dict(color="rgb(98, 168, 211)"),
            )
        ]
    )

    fig.update_layout(
        title="Precipitation for Station " + station_name,
        font_color=ftcolor,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=bgcolor,
        height=plot_height,
        margin=plot_margin,
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=subunitcolor)

    return fig

print(df.describe())
@app.callback(
    dash.dependencies.Output("precipitation-map", "figure"),
    [
        dash.dependencies.Input("my-date-picker-single", "date"),
        dash.dependencies.Input("show-zeros", "on"),
    ],
)
def wrapped_update_map(date_value, show_zeros):
    return update_map(date_value, show_zeros)


@app.callback(
    dash.dependencies.Output("time-series", "figure"),
    [dash.dependencies.Input("precipitation-map", "clickData")],
)
def wrapped_update_series(click_data):
    return update_series(click_data)


@app.callback(
    dash.dependencies.Output("my-date-picker-single", "date"),
    [dash.dependencies.Input("time-series", "clickData")],
)
def update_date(click_data):
    """Updates the date selector when a date is clicked on in the time series.

    Args:
        click_data: the data point the user clicked on in the time series
    Returns:
        A date - the date the user clicked on.
    """
    if click_data:  # click_data is None on app startup
        return click_data["points"][0]["customdata"]
    return initial_date


seconds_elapsed = time() - start_time
if verbose:
    print(seconds_elapsed, "seconds")
if verbose:
    print("Callbacks created. App finished building.")


if __name__ == "_main_":
    # debug causes code to run twice. Feel free to turn on.
    app.run_server(host="0.0.0.0", debug=False)
