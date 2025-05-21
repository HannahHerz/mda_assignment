
import faicons as fa
import plotly.express as px

# Load data and compute static values
from mda_assignment.shared import app_dir, tips, data
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly

bill_rng = (min(tips.total_bill), max(tips.total_bill))


ICONS = {
    "euro": fa.icon_svg("euro-sign"),
    "wallet": fa.icon_svg("wallet"),
    "contract": fa.icon_svg("file-contract"),
    "ellipsis": fa.icon_svg("frog"),
}

# Add page title and sidebar
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_slider(
            "total_bill",
            "Bill amount",
            min=bill_rng[0],
            max=bill_rng[1],
            value=bill_rng,
            pre="$",
        ),
        ui.input_checkbox_group(
            "time",
            "Food service",
            ["Lunch", "Dinner"],
            selected=["Lunch", "Dinner"],
            inline=True,
        ),
        ui.input_action_button("reset", "Reset filter"),
        open="desktop",
    ),
     
    ui.layout_columns(
        ui.value_box(
            "Total Funding", ui.output_ui("total_funding"), showcase=ICONS["euro"]
        ),
        ui.value_box(
            "Average Funding per Project", ui.output_ui("average_funding"), showcase=ICONS["wallet"]
        ),
        ui.value_box(
            "Amount of Projects",
            ui.output_ui("average_bill"),
            showcase=ICONS["contract"],
        ),
        fill=False,
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Tips data"), ui.output_data_frame("table")
        ),
        ui.card(
            ui.card_header(
                "Total bill vs tip",
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
                        "tip_perc_y",
                        "Split by:",
                        ["sex", "smoker", "day", "time"],
                        selected="day",
                        inline=True,
                    ),
                    title="Add a color variable",
                ),
                class_="d-flex justify-content-between align-items-center",
            ),
            output_widget("tip_perc"),
            full_screen=True,
        ),
        col_widths=[6, 6, 12],
    ),
    ui.include_css(app_dir / "styles.css"),
    title="Restaurant tipping",
    fillable=True,
)


def server(input, output, session):
    @reactive.calc
    def tips_data():
        bill = input.total_bill()
        idx1 = tips.total_bill.between(bill[0], bill[1])
        idx2 = tips.time.isin(input.time())
        return tips[idx1 & idx2]
        

    @render.ui
    def total_tippers():
        return f"€{tips_data().shape[0]}:.1"
    
    @render.ui
    def total_funding():
        return f"€{data['ecMaxContribution'].sum():.2f} ↑"

    @render.ui
    def average_funding():
            return f"€{data['ecMaxContribution'].mean():.2f} "

    @render.ui
    def average_bill():
        d = tips_data()
        if d.shape[0] > 0:
            bill = d.total_bill.mean()
            return f"{bill:.0f}"

    @render.data_frame
    def table():
        return render.DataGrid(tips_data())

    @render_plotly
    def scatterplot():
        color = input.scatter_color()
        return px.scatter(
            tips_data(),
            x="total_bill",
            y="tip",
            color=None if color == "none" else color,
            trendline="lowess",
        )

    @render_plotly
    def tip_perc():
        from ridgeplot import ridgeplot

        dat = tips_data()
        dat["percent"] = dat.tip / dat.total_bill
        yvar = input.tip_perc_y()
        uvals = dat[yvar].unique()

        samples = [[dat.percent[dat[yvar] == val]] for val in uvals]

        plt = ridgeplot(
            samples=samples,
            labels=uvals,
            bandwidth=0.01,
            colorscale="viridis",
            colormode="row-index",
        )

        plt.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            )
        )

        return plt

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        ui.update_slider("total_bill", value=bill_rng)
        ui.update_checkbox_group("time", selected=["Lunch", "Dinner"])


app = App(app_ui, server)