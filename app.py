import faicons as fa
import plotly.express as px
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud
from mda_assignment.shared import app_dir, data
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly, render_widget

ICONS = {
    "euro": fa.icon_svg("euro-sign"),
    "wallet": fa.icon_svg("wallet"),
    "contract": fa.icon_svg("file-contract"),
    "ellipsis": fa.icon_svg("ellipsis"),
}

year_rng = (
    int(data['ecSignatureDate'].dt.year.min()),
    int(data['ecSignatureDate'].dt.year.max())
)

topics = {"natural sciences": "Natural Sciences",
          "engineering and technology": "Engineering & Tech",
          "medical and health sciences": "Medical & Health",
          "social sciences": "Social Sciences",
          "humanities": "Humanities",
          "agricultural sciences": "Agricultural Sciences",
          "not available": "Not Available"
          }

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
                            topics,
                            selected=["natural sciences", "engineering and technology", "medical and health sciences", "social sciences", "humanities", "agricultural sciences", "not available"],
                            inline=True,
                        )
                    ),
                ),
                ui.input_action_button("apply_filters", "Apply filters"),
                ui.input_action_button("reset", "Reset filters (press apply to confirm reset)"),
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
                    ui.card_header(
                        "Distribution of Funding per Topic",
                    ),
                    output_widget("pie_funding"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("Distribution of Amount of Projects per Topic"),
                    output_widget("pie_projects"),
                    full_screen=True
                ),
            ),
                ui.card(
                    ui.card_header(
                        "Quarterly Funding",
                    ),
                    ui.popover(
                        ICONS["ellipsis"],
                        ui.input_select(
                            "graph_color",
                            "Select coloring logic",
                            {
                                "funds": "Coloring based on funds",
                                "topics": "Coloring based on topics"
                            },
                            selected="funds"
                        ),
                        title="Filter by topic",
                        placement="top"
                    ),
                    output_widget("time_contribution"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header(
                        "Quarterly Topic",
                    ),
                    output_widget("time_topics"),
                    full_screen=True,
                ),

            ui.layout_columns(
                ui.card(
                    ui.card_header(
                        "Avg country Funding",
                    ),
                    output_widget("avg_country_map"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header(
                        "Total country funding",
                    ),
                    output_widget("total_country_map"),
                    full_screen=True
                )
            ),

                ui.card(
                    ui.card_header(
                        "Word cloud of objectives"
                    ),
                    ui.output_plot("wordcloud")
                )
                
        ),
        ui.nav_panel(
            "Predictions",
            ui.card(
                    ui.value_box(
                        "Predicted Funding",
                        ui.output_ui("predict"),
                        showcase=ICONS["wallet"],
                    )
                ),
                ui.card(
                    ui.card_header("Input"),
                    ui.layout_columns(
                    ui.card(ui.input_numeric("budget", "Budget in euro", 0, min=1)),
                    ui.card(
                        ui.input_radio_buttons(
                            "objectives", 
                            "Objective", 
                            {
                                "obj_1": "Advanced Engery Storage Materials",
                                "obj_2": "Academic Researcher Training",
                                "obj_3": "EU Climate Policy Data",
                                "obj_4": "Molecular Synthetic Biology",
                                "obj_5": "Clinical Cancer Cell Biology",
                                "obj_6": "Industrial Sustainable Energy",
                                "obj_7": "Global Environmental Change",
                                "obj_8": "Social and Cultural Studies",
                                "obj_9": "Theoretical Quantum Physics",
                                "obj_10": "Digital Health and AI"
                            }
                        )
                    ),
                    ui.card(
                        ui.input_checkbox_group(
                            "regionselection", 
                            "Region", 
                            {
                                "northern europe": "Northern Europe",
                                "eastern europe": "Eastern Europe",
                                "southern europe": "Southern Europe",
                                "western europe": "Western Europe",
                                "africa": "Africa",
                                "americas": "North and South America",
                                "asia": "Asia",
                                "oceania": "Oceania"
                            }
                        ),
                        ui.input_numeric("numcountries", "Number of Participating Countries", 1, min=1)
                    ),
                    ui.card(
                    ui.input_numeric("numsme", "Number of Small or Medium Entreprises", 0, min=0),
                    ui.input_numeric("numpartner", "Number of Partners", 0, min=0),
                    ui.input_numeric("numthirdparties", "Number of Third Parties", 0, min=0),
                    ui.input_numeric("numasspartners", "Number of Associated Partners", 0, min=0)),
                    ui.card(ui.input_date("startdate", "Start of project"),
                    ui.input_date("enddate", "Expected end of project")),
                ),
                ui.input_action_button("predict_button", "Calculate predicted funding")  
            ),
        ),
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

#Format topics for mapping
def format_topics(topics_list):
    if not isinstance(topics_list, list):
        return "No specific topic"
    
    #Filter out "not available"
    valid_topics = [topic for topic in topics_list if topic != "not available"]
    
    if not valid_topics:
        return "No specific topic"
    
    #Format topics
    formatted_topics = []
    for i, topic in enumerate(valid_topics):
        if i == 0:
            formatted_topics.append(topic)  # First topic without bullet
        else:
            formatted_topics.append(f"• {topic}")  # Subsequent topics with bullet
    
    return "\n".join(formatted_topics)


def server(input, output, session):
    @reactive.event(input.apply_filters, ignore_none=False)
    def filtered_data():
        year_range = input.signature_year()
        idx1 = data['ecSignatureDate'].dt.year.between(year_range[0], year_range[1])
        idx2 = data['topic'].isin(input.topic())
        return data[idx1 & idx2]

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        ui.update_slider("signature_year", value=year_rng)
        ui.update_checkbox_group("topic", selected=["natural sciences", "engineering and technology", "medical and health sciences", "social sciences", "humanities", "agricultural sciences", "not available"])

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

    @reactive.calc
    def colormap():
       return {"natural sciences": px.colors.qualitative.Pastel[4],
                "engineering and technology": px.colors.qualitative.Pastel[2],
                "medical and health sciences": px.colors.qualitative.Pastel[3],
                "social sciences": px.colors.qualitative.Pastel[5],
                "humanities": px.colors.qualitative.Pastel[0],
                "agricultural sciences": px.colors.qualitative.Pastel[1],
                "not available": px.colors.qualitative.Pastel[10]}
    
    @reactive.calc 
    def catorder():
        return {"topic": ["natural sciences", "engineering and technology", "medical and health sciences", "social sciences", "humanities", "agricultural sciences", "not available"]}


    @render_plotly
    def time_contribution():
        clean_data = filtered_data().dropna(subset=['ecSignatureDate', 'ecMaxContribution', 'topic'])
        clean_data['quarter'] = clean_data['ecSignatureDate'].dt.to_period('Q').astype(str)
        clean_data['quarter'] = clean_data['quarter'].str.replace('Q', ' Q')

        # For the first chart (by quarter only)
        quarterly_data = clean_data.groupby('quarter')['ecMaxContribution'].sum().reset_index()
        quarterly_data = quarterly_data.sort_values('quarter')

        # For the second chart (by quarter and topic)
        quarterly_topic_data = clean_data.groupby(['quarter', 'topic'])['ecMaxContribution'].sum().reset_index()
        quarterly_topic_data = quarterly_topic_data.sort_values('quarter')
        
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
        
        fig2 = px.bar(
            quarterly_topic_data,
            x='quarter',
            y='ecMaxContribution',
            labels={
                'quarter': 'Quarter',
                'ecMaxContribution': 'EC Contribution'
            },
            color='topic',
            color_discrete_map= colormap(),
            category_orders= catorder()
        )

        fig2.update_layout(
            xaxis_title='Quarter of Signature Date',
            yaxis_title='EC Contribution',
            template='plotly_white',
            xaxis={'categoryorder': 'array', 'categoryarray': quarterly_topic_data['quarter'].unique().tolist()},
        )

        fig2.update_traces(
            texttemplate='%{y:.2s}',
            textposition='outside'
        )
        
        if input.graph_color() == "funds":
            return fig
        if input.graph_color() == "topics":
            return fig2

    @render_plotly
    def time_topics():
        clean_data = filtered_data().dropna(subset=['ecSignatureDate', 'topic'])
        clean_data['quarter'] = clean_data['ecSignatureDate'].dt.to_period('Q').astype(str)
        clean_data['quarter'] = clean_data['quarter'].str.replace('Q', ' Q')

        quarterly_topic_data = clean_data.groupby(['quarter', 'topic']).size().reset_index(name='count')
        quarterly_topic_data = quarterly_topic_data.sort_values('quarter')
        
        fig = px.bar(
            quarterly_topic_data,
            x='quarter',
            y='count',
            labels={
                'quarter': 'Quarter',
                'count': 'Number of Projects'
            },
            color='topic',
            color_discrete_map= colormap(),
            category_orders= catorder()
        )

        fig.update_layout(
            xaxis_title='Quarter of Signature Date',
            yaxis_title='Number of Projects',
            template='plotly_white',
            xaxis={'categoryorder': 'array', 'categoryarray': quarterly_topic_data['quarter'].unique().tolist()},
        )

        fig.update_traces(
            texttemplate='%{y}',
            textposition='outside'
        )

        return fig
        
      
    @reactive.calc
    def map_data():
        df = filtered_data()
        
        #Country mapping & names
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

        europe_df = df[df["country"].isin(country_mapping.keys())]
        if europe_df.empty:
            return pd.DataFrame()
        
        #Avg funding & total projects per country
        agg_df = europe_df.groupby("country").agg(
            average_funding=("ecMaxContribution", "mean"),
            total_projects=("projectID", "count")
        ).reset_index()
        
        #Top 3 topics per country
        top_topics = {}
        for country in agg_df["country"]:
            country_data = europe_df[europe_df["country"] == country]
            topics = country_data["topic"].value_counts().head(3).index.tolist()
            top_topics[country] = topics
        
        agg_df["top_topics"] = agg_df["country"].map(top_topics)
        
        #Country mapping
        agg_df["iso_alpha"] = agg_df["country"].map(country_mapping)
        agg_df["country_name"] = agg_df["country"].map(country_names)

        #Funding format
        agg_df["funding_display"] = agg_df["average_funding"].apply(
            lambda x: f"€{format_number(x)}"
        )
        
        return agg_df

    @render_plotly
    def avg_country_map():
        df = map_data()
        if df.empty:
            return px.choropleth(title="No data available for selected filters")
        
        df["formatted_avg_funding"] = df["average_funding"].apply(
            lambda x: f"€{format_number(x)}"
        )
        
        df["top_topics_str"] = df["top_topics"].apply(format_topics)
        
        max_avg_funding = df["average_funding"].max()

        map = px.choropleth(
            df,
            locations="iso_alpha",
            color="average_funding",
            scope="europe",
            hover_name="country_name",
            hover_data={
                "formatted_avg_funding": True,
                "total_projects": True,
                "top_topics_str": True,
                "average_funding": False,
                "iso_alpha": False,
                "country": False
            },
            labels={
                "formatted_avg_funding": "Avg Funding",
                "total_projects": "Projects",
                "top_topics_str": "Top Topics"
            },
            color_continuous_scale=px.colors.sequential.Blues,
            range_color=[0, max_avg_funding]
        )
        
        num_ticks = 6
        tick_avg_values = [i * (max_avg_funding / (num_ticks - 1)) for i in range(num_ticks)]
        tick_avg_labels = [f"€{format_number(val)}" for val in tick_avg_values]

        map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(
                tickvals=tick_avg_values,
                ticktext=tick_avg_labels
            )
        )
        
        return map

    @render_plotly
    def total_country_map():
        df = map_data()
        if df.empty:
            return px.choropleth(title="No data available for selected filters")
        
        # Calculate total funding per country
        df["total_funding"] = df["average_funding"] * df["total_projects"]
        
        df["formatted_total_funding"] = df["total_funding"].apply(
            lambda x: f"€{format_number(x)}"
        )
        
        df["top_topics_str"] = df["top_topics"].apply(format_topics)
        
        max_total_funding = df["total_funding"].max()

        map = px.choropleth(
            df,
            locations="iso_alpha",
            color="total_funding",
            scope="europe",
            hover_name="country_name",
            hover_data={
                "formatted_total_funding": True,
                "total_projects": True,
                "top_topics_str": True,
                "total_funding": False,
                "iso_alpha": False,
                "country": False
            },
            labels={
                "formatted_total_funding": "Total Funding",
                "total_projects": "Projects",
                "top_topics_str": "Top Topics"
            },
            color_continuous_scale=px.colors.sequential.Blues,
            range_color=[0, max_total_funding]
        )
        
        num_ticks = 6
        tick_total_values = [i * (max_total_funding / (num_ticks - 1)) for i in range(num_ticks)]
        tick_total_labels = [f"€{format_number(val)}" for val in tick_total_values]

        map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(
                tickvals=tick_total_values,
                ticktext=tick_total_labels
            )
        )
        
        return map
    
    @render_plotly
    def pie_funding():
        data = filtered_data()
        fig = px.pie(data, values='ecMaxContribution', names='topic', color='topic',
                     color_discrete_map= colormap(),
                     category_orders= catorder()
                     )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    @render_plotly
    def pie_projects():
        data = filtered_data()
        topic_counts = data['topic'].value_counts().reset_index()
        topic_counts.columns = ['topic', 'count']
        fig = px.pie(topic_counts, values='count', names='topic', color='topic',
                     color_discrete_map= colormap(),
                     category_orders= catorder()
                     )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    @render.ui
    def predict():
        total = format_number(filtered_data()['ecMaxContribution'].sum())
        return f"€{total}"

app = App(app_ui, server)