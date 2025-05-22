import faicons as fa
import plotly.express as px
import numpy as np
import pandas as pd
from mda_assignment.shared import app_dir, data
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly


ICONS = {
    "euro": fa.icon_svg("euro-sign"),
    "wallet": fa.icon_svg("wallet"),
    "contract": fa.icon_svg("file-contract"),
    "ellipsis": fa.icon_svg("frog"),
}

year_rng = (
    int(data['ecSignatureDate'].dt.year.min()),
    int(data['ecSignatureDate'].dt.year.max())
)

app_ui = ui.page_fillable(
    ui.input_dark_mode(),
    ui.navset_pill(
        ui.nav_panel(
            "Dashboard",
            ui.card(
                ui.card_header("Filters"),
                ui.layout_columns(
                    ui.card(
                        ui.input_slider(
                            "signature_year",
                            "Signature Year Range",
                            min=year_rng[0],
                            max=year_rng[1],
                            value=year_rng,
                        )
                    ),
                    ui.card(
                        ui.input_checkbox_group(
                            "topic",
                            "Topic",
                            ["natural sciences", "engineering and technology", "medical and health sciences", "social sciences", "humanities", "agricultural sciences", "not available"],
                            selected=["natural sciences", "engineering and technology", "medical and health sciences", "social sciences", "humanities", "agricultural sciences", "not available"],
                            inline=True,
                        )
                    ),
                ),
                ui.input_action_button("apply_filters", "Apply Filters"),
                open="desktop",
            ),
            ui.layout_columns(
                ui.value_box(
                    "Total Funding",
                    ui.output_ui("total_funding"),
                    showcase=ICONS["euro"],
                ),
                ui.value_box(
                    "Average Funding per Project",
                    ui.output_ui("average_funding"),
                    showcase=ICONS["wallet"],
                ),
                ui.value_box(
                    "Amount of Projects",
                    ui.output_ui("projectcount"),
                    showcase=ICONS["contract"],
                ),
                fill=False,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Funding data FILLER"),
                    ui.output_data_frame("table"),
                ),
                ui.card(
                    ui.card_header(
                        "Scatterplot FILLER",
                        ui.popover(
                            ICONS["ellipsis"],
                            ui.input_radio_buttons(
                                "scatter_color",
                                None,
                                ["none", "sex", "smoker", "day", "time"],
                                inline=True,
                            ),
                            title="Add a color variable",
                            placement="top",
                        ),
                        class_="d-flex justify-content-between align-items-center",
                    ),
                    output_widget("scatterplot"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header(
                        "Quarterly Funding",
                    ),
                    output_widget("time_contribution"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header(
                        "Country Funding",
                        ui.popover(
                            ICONS["ellipsis"],
                            ui.input_select(
                                "map_topic",
                                "Select Topic",
                                {
                                    "all": "All Topics",
                                    "natural sciences": "Natural Sciences",
                                    "engineering and technology": "Engineering & Tech",
                                    "medical and health sciences": "Medical & Health",
                                    "social sciences": "Social Sciences",
                                    "humanities": "Humanities",
                                    "agricultural sciences": "Agricultural Sciences",
                                    "not available": "Not Available"
                                },
                                selected="all"
                            ),
                            title="Filter by topic",
                            placement="top"
                        ),
                        class_="d-flex justify-content-between align-items-center",
                    ),
                    output_widget("country_map"),
                    full_screen=True
                ),
                col_widths=[6, 6, 12],
            ),
        ),
        ui.nav_panel("Predictions", 
                     ui.layout_columns(
                         ui.card(
                             ui.card_header("Input"),
                             ui.card(
                                  ui.input_numeric("budget", "Budget in euro", 1, min=1)),
                            ui.card(
                                 ui.input_radio_buttons( "radio", "Topic", {"natural sciences": "Natural Sciences",
                                    "engineering and technology": "Engineering & Tech",
                                    "medical and health sciences": "Medical & Health",
                                    "social sciences": "Social Sciences",
                                    "humanities": "Humanities",
                                    "agricultural sciences": "Agricultural Sciences",
                                    "not available": "Not Available"})),
                         ),
                         ui.card(
                             ui.value_box(
                                 "Predicted Funding",
                                ui.output_ui("predict"),
                                showcase=ICONS["ellipsis"], 
                             ) 
                         )                    

                     ),
        )
    ),
    ui.include_css(app_dir / "styles.css"),
    fillable=True,
)


# Function to format large numbers properly
def format_number(num):
    """
    Format a number with k, m, b, t suffixes for thousands, millions, billions, trillions.
    """
    if num < 1000:
        return str(num)

    magnitude = 0
    suffixes = ['', 'k', 'm', 'b', 't']

    while abs(num) >= 1000 and magnitude < len(suffixes) - 1:
        magnitude += 1
        num /= 1000.0

    formatted = f"{num:.2f}".rstrip('0').rstrip('.') if num % 1 else f"{int(num)}"
    return f"{formatted}{suffixes[magnitude]}"


def server(input, output, session):
    @reactive.event(input.apply_filters, ignore_none=False)
    def filtered_data():
        year_range = input.signature_year()
        idx1 = data['ecSignatureDate'].dt.year.between(year_range[0], year_range[1])
        idx2 = data['topic'].isin(input.topic())
        return data[idx1 & idx2]

    @render.ui
    def total_funding():
        total = format_number(filtered_data()['ecMaxContribution'].sum())
        return f"€{total}"

    @render.ui
    def average_funding():
        avg = filtered_data()['ecMaxContribution'].mean()
        average = format_number(avg)
        return f"€{average}"

    @render.ui
    def projectcount():
        count = format_number(filtered_data()['projectID'].nunique())
        return f"{count}"

    @render.data_frame
    def table():
        return render.DataGrid(filtered_data())

    @render_plotly
    def scatterplot():
        color = input.scatter_color()
        return px.scatter(
            filtered_data(),
            x="ecMaxContribution",
            y="ecContribution_sum",
            color=None if color == "none" else color,
            trendline="lowess",
        )

    @render_plotly
    def time_contribution():
        clean_data = filtered_data().dropna(subset=['ecSignatureDate', 'ecMaxContribution'])
        clean_data['quarter'] = clean_data['ecSignatureDate'].dt.to_period('Q').astype(str)
        clean_data['quarter'] = clean_data['quarter'].str.replace('Q', ' Q')

        # Group by quarter and calculate the sum of ecMaxContribution
        quarterly_data = clean_data.groupby('quarter')['ecMaxContribution'].sum().reset_index()

        # Sort by quarter for chronological display
        quarterly_data = quarterly_data.sort_values('quarter')

        # Create plot
        fig = px.bar(
            quarterly_data,
            x='quarter',
            y='ecMaxContribution',
            labels={
                'quarter': 'Quarter',
                'ecMaxContribution': 'EC Contribution'
            },
            color='ecMaxContribution',
            color_continuous_scale='Blues'
        )

        # Layout
        fig.update_layout(
            xaxis_title='Quarter of Signature Date',
            yaxis_title='EC Contribution',
            template='plotly_white',
            xaxis={'categoryorder': 'array', 'categoryarray': quarterly_data['quarter'].tolist()},
            coloraxis_showscale=False
        )

        fig.update_traces(
            texttemplate='%{y:.2s}',
            textposition='outside'
        )
        return fig

    @reactive.calc
    def map_data():
        # Filter by selected topic
        df = filtered_data()
        if input.map_topic() != "all":
            df = df[df["topic"] == input.map_topic()]
        
        # Get European country codes (2-letter to 3-letter mapping)
        country_mapping = {
            'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'HR': 'HRV', 'CY': 'CYP',
            'CZ': 'CZE', 'DK': 'DNK', 'EE': 'EST', 'FI': 'FIN', 'FR': 'FRA',
            'DE': 'DEU', 'GR': 'GRC', 'HU': 'HUN', 'IE': 'IRL', 'IT': 'ITA',
            'LV': 'LVA', 'LT': 'LTU', 'LU': 'LUX', 'MT': 'MLT', 'NL': 'NLD',
            'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'SK': 'SVK', 'SI': 'SVN',
            'ES': 'ESP', 'SE': 'SWE', 'GB': 'GBR', 'NO': 'NOR', 'CH': 'CHE'
        }
        
        country_names = {
            'AT': 'Austria', 
            'BE': 'Belgium', 
            'BG': 'Bulgaria', 
            'HR': 'Croatia', 
            'CY': 'Cyprus',
            'CZ': 'Czech Republic', 
            'DK': 'Denmark', 
            'EE': 'Estonia', 
            'FI': 'Finland', 
            'FR': 'France',
            'DE': 'Germany', 
            'GR': 'Greece', 
            'HU': 'Hungary', 
            'IE': 'Ireland', 
            'IT': 'Italy',
            'LV': 'Latvia', 
            'LT': 'Lithuania', 
            'LU': 'Luxembourg', 
            'MT': 'Malta', 
            'NL': 'Netherlands',
            'PL': 'Poland', 
            'PT': 'Portugal', 
            'RO': 'Romania', 
            'SK': 'Slovakia', 
            'SI': 'Slovenia',
            'ES': 'Spain', 
            'SE': 'Sweden', 
            'GB': 'United Kingdom', 
            'NO': 'Norway', 
            'CH': 'Switzerland'
        }

        # Process data
        europe_df = df[df["country"].isin(country_mapping.keys())]
        if europe_df.empty:
            return pd.DataFrame()
        
        # Calculate average funding and top topics
        agg_df = europe_df.groupby("country").agg(
            average_funding=("ecMaxContribution", "mean"),
            total_projects=("projectID", "count")
        ).reset_index()
        
        # Get top 3 topics per country
        top_topics = {}
        for country in agg_df["country"]:
            country_data = europe_df[europe_df["country"] == country]
            topics = country_data["topic"].value_counts().head(3).index.tolist()
            top_topics[country] = topics
        
        agg_df["top_topics"] = agg_df["country"].map(top_topics)
        
        # Add 3-letter country codes
        agg_df["iso_alpha"] = agg_df["country"].map(country_mapping)
        agg_df["country_name"] = agg_df["country"].map(country_names)

        # Format the funding value for display
        agg_df["funding_display"] = agg_df["average_funding"].apply(
            lambda x: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.1f}k" if x >= 1e3 else f"{x:.0f}"
        )
        
        return agg_df

    @render_plotly
    def country_map():
        df = map_data()
        if df.empty:
            return px.choropleth(title="No data available for selected filters")
        
        df["formatted_funding"] = df["average_funding"].apply(
            lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else 
                    f"€{x/1e3:.1f}k" if x >= 1e3 else 
                    f"€{x:.0f}"
        )
        
        df["top_topics_str"] = df["top_topics"].apply(
            lambda x: "\n• ".join([
                topic if i == 0 else f"• {topic}" 
                for i, topic in enumerate(x) 
                if topic != "not available"
            ]) if isinstance(x, list) and any(topic != "not available" for topic in x) 
            else "No specific topic"
        )
        
        fig = px.choropleth(
            df,
            locations="iso_alpha",
            color="average_funding",
            scope="europe",
            hover_name="country_name",
            hover_data={
                "formatted_funding": True,
                "total_projects": True,
                "top_topics_str": True,
                "average_funding": False,
                "iso_alpha": False,
                "country": False
            },
            labels={
                "formatted_funding": "Average Funding",
                "total_projects": "Projects",
                "top_topics_str": "Top Topics"
            },
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(
                title="Avg Funding",
                tickprefix="€",
                tickformat=",.0f"
            )
        )
        
        return fig
    
    @render.ui
    def predict():
        total = format_number(filtered_data()['ecMaxContribution'].sum())
        return f"€{total}"
    
app = App(app_ui, server)