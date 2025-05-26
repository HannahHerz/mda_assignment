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
    "magnifying-glass": fa.icon_svg("magnifying-glass"),
    "calculator": fa.icon_svg("calculator"),
    "github": fa.icon_svg("github")
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

url = 'https://github.com/HannahHerz/mda_assignment'

app_ui = ui.page_fillable(
    ui.input_dark_mode(),
    ui.tags.a(ICONS['github'], href=url, target="_blank", style="font-size: 30px; text-align: right;"),
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
                            sep=''
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
                        ICONS["magnifying-glass"],
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
                        "Average Funding per Country",
                    ),
                    output_widget("avg_country_map"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header(
                        "Total Funding per Country",
                    ),
                    output_widget("total_country_map"),
                    full_screen=True
                )
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header(
                        "Word cloud of objectives"
                    ),
                    ui.output_plot("wordcloud")
                ),
                ui.card(
                    ui.card_header("EU vs Total Funding Distribution"),
                    output_widget("funding_donut_diagram"),
                    full_screen=True
                )
            )
        ),
        ui.nav_panel(
            "Predictions",
            ui.card( ui.output_text_verbatim("predict")),
            ui.card(
                    ui.value_box(
                        "Predicted Funding",
                        ui.output_ui("predict2"),
                        showcase=ICONS["calculator"],
                    )
                ),
                ui.card(
                    ui.card_header("Input"),
                    ui.layout_columns(
                    ui.card(
                        ui.layout_columns(
                        ui.input_radio_buttons(
                            "euroSciVoxTopic", 
                            "Topic", 
                            topics                        
                        ),
                        ui.input_radio_buttons(
                            "objective", 
                            "Objective", 
                            {
                                "obj_advanced_energy_storage_materials": "Advanced Engery Storage Materials",
                                "obj_researcher_academic_training": "Academic Researcher Training",
                                "obj_eu_climate_policy_data": "EU Climate Policy Data",
                                "obj_molecular_synthetic_biology": "Molecular Synthetic Biology",
                                "obj_clinical_cell_cancer_biology": "Clinical Cancer Cell Biology",
                                "obj_industrial_sustainable_energy": "Industrial Sustainable Energy",
                                "obj_global_environmental_change": "Global Environmental Change",
                                "obj_social_cultural_studies": "Social and Cultural Studies",
                                "obj_quantum_theoretical_physics": "Theoretical Quantum Physics",
                                "obj_digital_health_ai": "Digital Health and AI"
                            }                        
                        )),
                        ui.input_select("fundingScheme", "Funding Scheme",
                                        {
                                        'HORIZON-TMA-MSCA-PF-EF': 'HORIZON-TMA-MSCA-PF-EF',
                                        'HORIZON-ERC': 'HORIZON-ERC',
                                        'HORIZON-RIA': 'HORIZON-RIA',
                                        'HORIZON-CSA': 'HORIZON-CSA',
                                        'HORIZON-IA': 'HORIZON-IA',
                                        'HORIZON-ERC-POC': 'HORIZON-ERC-POC',
                                        'HORIZON-EIC': 'HORIZON-EIC',
                                        'HORIZON-EIC-ACC-BF': 'HORIZON-EIC-ACC-BF',
                                        'HORIZON-TMA-MSCA-PF-GF': 'HORIZON-TMA-MSCA-PF-GF',
                                        'HORIZON-TMA-MSCA-DN': 'HORIZON-TMA-MSCA-DN',
                                        'HORIZON-JU-RIA': 'HORIZON-JU-RIA',
                                        'HORIZON-TMA-MSCA-SE': 'HORIZON-TMA-MSCA-SE',
                                        'HORIZON-JU-IA': 'HORIZON-JU-IA',
                                        'HORIZON-ERC-SYG': 'HORIZON-ERC-SYG',
                                        'HORIZON-AG': 'HORIZON-AG',
                                        'HORIZON-EIC-ACC': 'HORIZON-EIC-ACC',
                                        'ERC': 'ERC',
                                        'HORIZON-TMA-MSCA-Cofund-P': 'HORIZON-TMA-MSCA-Cofund-P',
                                        'HORIZON-TMA-MSCA-DN-ID': 'HORIZON-TMA-MSCA-DN-ID',
                                        'HORIZON-TMA-MSCA-Cofund-D': 'HORIZON-TMA-MSCA-Cofund-D',
                                        'HORIZON-JU-CSA': 'HORIZON-JU-CSA',
                                        'HORIZON-TMA-MSCA-DN-JD': 'HORIZON-TMA-MSCA-DN-JD',
                                        'EURATOM-RIA': 'EURATOM-RIA',
                                        'HORIZON-COFUND': 'HORIZON-COFUND',
                                        'MSCA-PF': 'MSCA-PF',
                                        'HORIZON-AG-UN': 'HORIZON-AG-UN',
                                        'HORIZON-EIT-KIC': 'HORIZON-EIT-KIC',
                                        'EURATOM-CSA': 'EURATOM-CSA',
                                        'EURATOM-IA': 'EURATOM-IA',
                                        'ERC-POC': 'ERC-POC',
                                        'HORIZON-AG-LS': 'HORIZON-AG-LS',
                                        'EIC-ACC': 'EIC-ACC',
                                        'CSA': 'CSA',
                                        'HORIZON-PCP': 'HORIZON-PCP',
                                        'EURATOM-COFUND': 'EURATOM-COFUND',
                                        'RIA': 'RIA',
                                        'EIC': 'EIC',
                                        'IA': 'IA'
   
                                        })
                    ),
                    ui.card(
                        ui.card_header("Region"),
                        ui.layout_columns(
                        ui.input_numeric("Northern_Europe_count", "Number of Participating Countries from Northern Europe", 0, min=0),
                        ui.input_numeric("Eastern_Europe_count", "Number of Participating Countries from Eastern Europe", 0, min=0),
                        ui.input_numeric("Southern_Europe_count", "Number of Participating Countries from Southern Europe", 0, min=0)),
                        ui.layout_columns(
                        ui.input_numeric("Western_Europe_count", "Number of Participating Countries from Western Europe", 0, min=0),
                        ui.input_numeric("Africa_count", "Number of Participating Countries from Africa", 0, min=0),
                        ui.input_numeric("Americas_count", "Number of Participating Countries from North and South America", 0, min=0)),
                        ui.layout_columns(
                        ui.input_numeric("Asia_count", "Number of Participating Countries from Asia", 0, min=0),
                        ui.input_numeric("Oceania_count", "Number of Participating Countries from Oceania", 0, min=0),
                        ui.input_numeric("num_countries", "Number of Total Participating Countries", 1, min=1))
                    )),
                    ui.layout_columns(
                    ui.card(
                    ui.layout_columns(
                    ui.input_text("organisationID", "Organization ID"),
                    ui.input_numeric("n_participant", "Number of Participants", 1, min=1),
                    ui.input_numeric("num_organisations", "Number of Organizations involved", 1, min=1)),
                    ui.layout_columns(
                    ui.input_numeric("num_sme", "Number of Small or Medium Entreprises", 0, min=0),
                    ui.input_numeric("n_thirdParty", "Number of Third Parties", 0, min=0),
                    ui.input_numeric("n_associatedPartner", "Number of Associated Partners", 0, min=0))),
                    ),

                ui.card(ui.layout_columns(ui.input_date("startDate", "Start of project"),
                ui.input_date("endDate", "Expected end of project"))),
                
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
            formatted_topics.append(topic)
        else:
            formatted_topics.append(f"• {topic}")
    
    return "\n".join(formatted_topics)


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

    @reactive.calc
    def colormap():
       topic_keys = list(topics.keys())
       return {topic_keys[0]: px.colors.qualitative.D3[2],
               topic_keys[1]: px.colors.qualitative.D3[1],
               topic_keys[2]: px.colors.qualitative.D3[3],
               topic_keys[3]: px.colors.qualitative.D3[0],
               topic_keys[4]: px.colors.qualitative.G10[4],
               topic_keys[5]: px.colors.qualitative.D3[5],
               topic_keys[6]: px.colors.qualitative.D3[7]}
    
    @reactive.calc 
    def catorder():
        return {"topic": ["natural sciences", "engineering and technology", "medical and health sciences", "social sciences", "humanities", "agricultural sciences", "not available"]}

    @render_plotly
    def pie_funding():
        data = filtered_data().copy()
        data['topic_display'] = data['topic'].map(topics)
        fig = px.pie(data, values='ecMaxContribution', names='topic_display', color='topic',
                     color_discrete_map= colormap(),
                     category_orders= {"topic_display": list(topics.values())}
                     )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    @render_plotly
    def pie_projects():
        data = filtered_data().copy()
        data['topic_display'] = data['topic'].map(topics)
        topic_counts = data['topic_display'].value_counts().reset_index()
        topic_counts.columns = ['topic_display', 'count']
        fig = px.pie(topic_counts, values='count', names='topic_display', color='topic_display',
                     color_discrete_map= {topics[i]: c for i, c in colormap().items()},
                     category_orders= {"topic_display": list(topics.values())}
                     )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

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
        
        quarterly_data['formatted_text'] = quarterly_data['ecMaxContribution'].apply(lambda x: f"€{format_number(x)}")

        fig = px.bar(
            quarterly_data,
            x='quarter',
            y='ecMaxContribution',
            labels={
                'quarter': 'Quarter',
                'ecMaxContribution': 'EC Contribution'
            },
            color_discrete_sequence=['#003399']
        )

        fig.update_layout(
            xaxis_title='Quarter of Signature Date',
            yaxis_title='EC Contribution',
            template='plotly_white',
            xaxis={'categoryorder': 'array', 'categoryarray': quarterly_data['quarter'].tolist()}
        )

        fig.update_traces(
            text=quarterly_data['formatted_text'],
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
        clean_data = filtered_data().dropna(subset=['ecSignatureDate', 'topic']).copy()
        clean_data['quarter'] = clean_data['ecSignatureDate'].dt.to_period('Q').astype(str)
        clean_data['quarter'] = clean_data['quarter'].str.replace('Q', ' Q')

        clean_data['topic_display'] = clean_data['topic'].map(topics)

        quarterly_topic_data = clean_data.groupby(['quarter', 'topic_display']).size().reset_index(name='count')
        quarterly_topic_data = quarterly_topic_data.sort_values('quarter')
        
        fig = px.bar(
            quarterly_topic_data,
            x='quarter',
            y='count',
            labels={
                'quarter': 'Quarter',
                'count': 'Number of Projects'
            },
            color='topic_display',
            color_discrete_map= {topics[i]: c for i, c in colormap().items()},
            category_orders= {"topic_display": list(topics.values())}
        )

        fig.update_layout(
            xaxis_title='Quarter of Signature Date',
            yaxis_title='Number of Projects',
            template='plotly_white',
            xaxis={'categoryorder': 'array', 'categoryarray': quarterly_topic_data['quarter'].unique().tolist()},
            legend_title_text=""
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
        mapdata = map_data()
        if mapdata.empty:
            return px.choropleth(title="No data available for selected filters")
        
        mapdata["formatted_avg_funding"] = mapdata["average_funding"].apply(
            lambda x: f"€{format_number(x)}"
        )
        
        mapdata["top_topics_str"] = mapdata["top_topics"].apply(format_topics)
        
        max_avg_funding = mapdata["average_funding"].max()

        map = px.choropleth(
            mapdata,
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
                "top_topics_str": "Top Topics",
                "average_funding": "Average Funding"
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
                title="",
                tickvals=tick_avg_values,
                ticktext=tick_avg_labels
            )
        )
        
        return map

    @render_plotly
    def total_country_map():
        mapdata = map_data()
        if mapdata.empty:
            return px.choropleth(title="No data available for selected filters")
        
        # Calculate total funding per country
        mapdata["total_funding"] = mapdata["average_funding"] * mapdata["total_projects"]
        
        mapdata["formatted_total_funding"] = mapdata["total_funding"].apply(
            lambda x: f"€{format_number(x)}"
        )
        
        mapdata["top_topics_str"] = mapdata["top_topics"].apply(format_topics)
        
        max_total_funding = mapdata["total_funding"].max()

        map = px.choropleth(
            mapdata,
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
                "top_topics_str": "Top Topics",
                "total_funding": "Total Funding"
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
                title="",
                tickvals=tick_total_values,
                ticktext=tick_total_labels
            )
        )
        
        return map
    
    @render.plot
    def wordcloud():
        wc = filtered_data()
        text = ' '.join(wc['objective'].astype(str).tolist())

        text = re.sub(r'[^A-Za-z\s]', '', text)

        text = text.lower()

        stopwords = set(STOPWORDS)
        custom_stopwords = {'project', 'will'}
        stopwords.update(custom_stopwords)    
        text = ' '.join(word for word in text.split() if word not in stopwords)
        wordcloud = WordCloud(background_color='white', max_words=75).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        return plt.gcf()
   
    

    @render_plotly
    def funding_donut_diagram():
        dd = filtered_data()
        
        #Avg per project
        avg_eu_funding = dd['ecMaxContribution'].mean()
        avg_total_cost = dd['totalCost'].mean()
        
        avg_non_eu_funding = avg_total_cost - avg_eu_funding
        
        funding_data = pd.DataFrame({
            'source': ['EU Contribution', 'Other Sources'],
            'amount': [avg_eu_funding, avg_non_eu_funding],
            'percentage': [
                (avg_eu_funding / avg_total_cost) * 100,
                (avg_non_eu_funding / avg_total_cost) * 100
            ]
        })
        
        #Donut diagram
        fig = px.pie(
            funding_data,
            values='amount',
            names='source',
            color='source',
            color_discrete_map={
                'EU Contribution': '#003399',
                'Other Sources': '#FFD700'
            },
            hole=0.6
        )
        
        fig.update_traces(hoverinfo='skip', hovertemplate=None)

        fig.update_traces(
            textposition='inside',
            textinfo='percent',
            textfont_size=12,
            marker=dict(line=dict(color='white', width=2))
        )
        
        #Middle text
        fig.add_annotation(
            text=f"Average per project<br>€{format_number(avg_total_cost)}",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False,
            font_color="black"
        )
        
        #Title & legend
        fig.update_layout(
            title={
                'text': f"Average EU funding vs other sources per project<br>EU: {funding_data.iloc[0]['percentage']:.1f}% of average project cost",
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
        margin=dict(t=100, b=60, l=20, r=20)
        )
        
        return fig
    

    @reactive.calc
    def funding_data():
        df = pd.DataFrame(["euroSciVoxTopic", "objective", "fundingScheme", "Northern_Europe_count", "Eastern_Europe_count", "Southern_Europe_count", "Western_Europe_count", "Africa_count", "Americas_count", "Asia_count", "Oceania_count", "num_countries", "organisationID", "n_participant", "num_organisations", "num_sme", "n_thirdParty", "n_associatedPartner", "startDate", "endDate"])
        return df

    @reactive.event(input.predict_button)
    def predict():
        from mda_assignment.model import loaded_model
        y = loaded_model.predict(funding_data)
        return f"€{y[1]}"



app = App(app_ui, server)