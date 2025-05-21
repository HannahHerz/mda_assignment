import faicons as fa
import plotly.express as px
import numpy as np
from mda_assignment.shared import app_dir, data
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly


ICONS = {
    "euro": fa.icon_svg("euro-sign"),
    "wallet": fa.icon_svg("wallet"),
    "contract": fa.icon_svg("file-contract"),
    "ellipsis": fa.icon_svg("frog"),
}

year_rng = (int(data['ecSignatureDate'].dt.year.min()), int(data['ecSignatureDate'].dt.year.max()))

app_ui = ui.page_fillable(
    ui.input_dark_mode(),
    ui.navset_pill(  
        ui.nav_panel(
            "Dashboard",
       ui.card(ui.card_header("Filters"), ui.layout_columns(
           ui.card(ui.input_slider(
            "signature_year",
            "Signature Year Range",
            min=year_rng[0],
            max=year_rng[1],
            value=year_rng,
        )),
        ui.card(ui.input_checkbox_group(
            "time",
            "Food service",
            ["Lunch", "Dinner"],
            selected=["Lunch", "Dinner"],
            inline=True,
        ))),
        ui.input_action_button("reset", "Reset filter"),
        open="desktop",),

            ui.layout_columns(
                ui.value_box(
                    "Total Funding", 
                    ui.output_ui("total_funding"), 
                    showcase=ICONS["euro"]
                ),
                ui.value_box(
                    "Average Funding per Project", 
                    ui.output_ui("average_funding"), 
                    showcase=ICONS["wallet"]
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
                    ui.card_header("Funding data"), 
                    ui.output_data_frame("table")
                ),
                ui.card(
                    ui.card_header(
                        "Scatterplot",
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
                        ui.popover(
                            ICONS["ellipsis"],
                            ui.input_radio_buttons(
                                "time_contribution_y",
                                "Split by:",
                                ["sex", "smoker", "day", "time"],
                                selected="day",
                                inline=True,
                            ),
                            title="Add a color variable",
                        ),
                        class_="d-flex justify-content-between align-items-center",
                    ),
                    output_widget("time_contribution"),
                    full_screen=True,
                ),
                col_widths=[6, 6, 12],
            )
        ),
        ui.nav_panel("Predictions",
                    "test")
    ),
    ui.include_css(app_dir / "styles.css"),
    fillable=True,
)

# Function to format large numbers
def format_number(num):
    """
    Format a number with k, m, b, t suffixes for thousands, millions, billions, trillions
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
    @reactive.calc
    def filtered_data():
        year_range = input.signature_year()
        idx1 = data['ecSignatureDate'].dt.year.between(year_range[0], year_range[1])
        return data[idx1]
    
        
    @render.ui
    def total_funding():
        total = format_number(filtered_data()['ecMaxContribution'].sum())
        return f"€{total} ↑"
        
    @render.ui
    def average_funding():
        avg = filtered_data()['ecMaxContribution'].mean()
        average = format_number(avg)
        return f"€{average}"

    @render.ui
    def projectcount():
        row_count = format_number(len(filtered_data()))
        return f"{row_count}"

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
            color_continuous_scale='Viridis'
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

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        ui.update_slider("signature_year", value=year_rng)
        ui.update_checkbox_group("time", selected=["Lunch", "Dinner"])


app = App(app_ui, server)