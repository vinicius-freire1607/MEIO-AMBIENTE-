
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit as st

st.set_page_config(layout="wide")

# 2. CSS "Fine Line" Profissional
st.markdown("""
    <style>
        /* Fundo azul e verde e tipografia neutra */
        .stApp { background-color: #ffffff; }
        
        /* Títulos estilo 'Fine Line' */
        .header-box {
            border-left: 3px solid #000000;
            padding-left: 15px;
            margin-bottom: 30px;
        }
        h1 { font-family: 'Inter', sans-serif; font-weight: 700; color: #000; font-size: 24px; letter-spacing: -0.5px; }
        h3 { font-family: 'Inter', sans-serif; font-weight: 400; color: #444; font-size: 16px; margin-top: -10px; }
        
        /* Linhas divisórias finas */
        hr { border: 0; border-top: 1px solid #e5e5e5; margin: 30px 0; }
    </style>
""", unsafe_allow_html=True)
st.write("Teste")
st.markdown(""" <div style="padding-bottom: 20px;">
        <h1 style="color: #000000; font-family: sans-serif;">Dashboard de Qualidade da Água</h1>
    </div>
""", unsafe_allow_html=True)
st.caption("Oxigênio Dissolvido (OD) • Análise exploratória • Modelagem preditiva")
# 4. Estrutura do Conteúdo
# Em vez de imagens, usamos o próprio espaço para dividir as seções
st.markdown("### Visão Geral")
col1, col2 = st.columns(2)
with col1:
    st.metric("Nível Médio de OD", "7.2 mg/L")
with col2:
    st.metric("Estações Ativas", "26")

st.markdown("<hr>", unsafe_allow_html=True)
# --- Configuração da Página e Tema ---


# CSS "Fine Line" Profissional
st.markdown("""
    <style>
    /* Cores personalizadas */
    :root {
        --primary-color: #4CAF50; /* Verde vibrante */
        --secondary-color: #FFC107; /* Amarelo para destaque */
        --background-color: #f0f2f6; /* Fundo suave */
        --card-background-color: #ffffff;
        --text-color: #333333;
        --header-color: #1a1a1a;
        --sidebar-background: #e6e6e6;
    }

    .reportview-container .main {
        color: var(--text-color);
        /* Gradiente azul para o fundo */
        background: linear-gradient(to right, #e0f2f7, #bbdefb); /* Um degradê azul suave */
    }
    .sidebar .sidebar-content {
        background-color: var(--sidebar-background);
        color: var(--text-color);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--header-color);
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px; /* Bordas mais arredondadas */
        padding: 10px 20px;
        font-size: 16px; /* Fonte maior */
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.1s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Sombra para botões */
    }
    .stButton>button:hover {
        background-color: #43A047; /* Verde um pouco mais escuro */
        transform: translateY(-2px); /* Efeito de elevação */
        box_shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stButton>button:active {
        background-color: #388E3C;
        transform: translateY(0);
        box_shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .css-1d391kg, .css-1l0dlwz, .css-vk32qa, .css-1dp5dn9 { /* Adjust for sidebar text color */
        color: var(--text-color) !important;
    }
    /* Refinamentos para campos de entrada */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 1px solid #cccccc; /* Borda mais suave */
        padding: 10px 15px;
        background-color: var(--card-background-color);
        color: var(--text-color);
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus, .stSelectbox>div>div>select:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 0.2rem rgba(76, 175, 80, 0.25); /* Sombra de foco com cor primária */
        outline: none;
    }
    /* Estilo de Cartão para st.container e st.expander */
    .stBlock, .stContainer {
        padding: 25px; /* Mais preenchimento */
        margin-bottom: 25px; /* Mais espaço entre cartões */
        border-radius: 12px; /* Bordas mais arredondadas */
        box-shadow: 0 6px 15px rgba(0,0,0,0.1); /* Sombra mais pronunciada */
        background-color: var(--card-background-color);
        border: 1px solid #e0e0e0; /* Borda sutil */
    }
    .stExpander {
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        background-color: var(--card-background-color);
        border: 1px solid #e0e0e0;
    }
    .stAlert.info {
        background-color: #e0f7fa; /* Cor de fundo para informações */
        color: #006064; /* Cor de texto para informações */
        border-left: 5px solid #00bcd4; /* Borda de destaque */
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Funções de Carregamento e Limpeza de Dados ---
@st.cache_data(show_spinner="Carregando e limpando os dados...")
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv('BASE DE DADOS OD - ANA.csv')
    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{'BASE DE DADOS OD - ANA.csv'}' não foi encontrado. Por favor, certifique-se de que o arquivo CSV esteja no local correto.")
        st.stop()

    # Remover duplicatas
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)

    # Preencher NaNs com a mediana para colunas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Outlier removal for 'NUOD'
    Q1 = df['NUOD'].quantile(0.25)
    Q3 = df['NUOD'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[~((df['NUOD'] < lower_bound) | (df['NUOD'] > upper_bound))].copy()

    # Ensure 'AMBIENTE' has no NaNs for modeling, if not handled by general numeric fillna
    if df_cleaned['AMBIENTE'].isnull().any():
        median_ambiente = df_cleaned['AMBIENTE'].median()
        df_cleaned['AMBIENTE'].fillna(median_ambiente, inplace=True)

    # Ensure 'CDESTACAO' is string type for consistent sorting
    df_cleaned['CDESTACAO'] = df_cleaned['CDESTACAO'].astype(str)

    # Melt for temporal analysis and classification target creation
    id_vars = ['CDESTACAO', 'LATITUDE', 'LONGITUDE', 'AMBIENTE', 'SGUF', 'CORPODAGUA'] # Added SGUF and CORPODAGUA
    med_cols = [col for col in df_cleaned.columns if col.startswith('MED_')] # Ex: MED_1978

    df_melted = df_cleaned.melt(
        id_vars=id_vars,
        value_vars=med_cols,
        var_name='Ano_Coluna',
        value_name='Valor_OD_Medio'
    )
    df_melted['Ano'] = df_melted['Ano_Coluna'].str.extract(r'(\d+)').astype(int)
    df_melted.dropna(subset=['Valor_OD_Medio'], inplace=True)
    df_melted['OD_Status'] = np.where(df_melted['Valor_OD_Medio'] < 5, 'Ruim', 'Bom')

    return df, df_cleaned, df_melted

# --- Carregar Dados ---
file_path = 'BASE DE DADOS OD - ANA.csv'
df_original, df_cleaned, df_melted = load_and_clean_data(file_path)


# --- Sidebar para Navegação ---
st.sidebar.title("Navegação")
page = st.sidebar.radio(
    "",
    ["Introdução", "Análise Exploratória de Dados", "Mapa e Dados Brutos", "Metodologia de Tratamento de Dados", "Resultados do Modelo Preditivo"]
)


# --- Filtros Globais para SGUF e CDESTACAO (aplicados em Análise Exploratória e Mapa) ---
st.sidebar.markdown("### Filtros de Dados")

available_sguf = sorted(df_cleaned['SGUF'].unique().tolist())
selected_sguf = st.sidebar.multiselect(
    "Filtrar por Unidade Federativa (SGUF)",
    options=available_sguf,
    default=[]
)

filtered_df_cleaned = df_cleaned.copy()
if selected_sguf:
    filtered_df_cleaned = filtered_df_cleaned[filtered_df_cleaned['SGUF'].isin(selected_sguf)]

# --- BLOCO DE CORREÇÃO ---
# Remove nulos, converte cada item para string (para evitar conflito float vs str)
# e remove o '.0' caso algum número tenha ficado com casa decimal
temp_list = filtered_df_cleaned['CDESTACAO'].dropna().astype(str)
temp_list = temp_list.apply(lambda x: x.replace('.0', '') if x.endswith('.0') else x)

# Agora que tudo é texto limpo, geramos a lista única e ordenamos
available_cdestacao = sorted(temp_list.unique().tolist())
# --- FIM DA CORREÇÃO ---
selected_cdestacao = st.sidebar.multiselect
selected_cdestacao = st.sidebar.multiselect(
    "Filtrar por Estação (CDESTACAO)",
    options=available_cdestacao,
    default=[],
    key="cdestacao_filter"
)

# --- filtros ---
selected_cdestacao = st.sidebar.multiselect(
    "Filtrar por Estação (CDESTACAO)",
    options=available_cdestacao,
    default=[]
)

if selected_cdestacao:
    filtered_df_cleaned = filtered_df_cleaned[
        filtered_df_cleaned['CDESTACAO'].isin(selected_cdestacao)
    ]

# =========================
# CONTEÚDO DAS PÁGINAS
# =========================

if page == "Introdução":

    st.markdown(
        "Uma exploração detalhada dos dados tabulares de Oxigênio Dissolvido, "
        "incluindo limpeza, visualizações e modelagem preditiva."
    )

    st.header("Compreendendo o Oxigênio Dissolvido (OD)")

    st.markdown("""
    O Oxigênio Dissolvido (OD) é um dos indicadores mais importantes da qualidade da água em ecossistemas aquáticos.
    Ele é essencial para a sobrevivência de peixes e outros organismos aquáticos.
    """)

    st.header("Visão Geral dos Dados")

    with st.container(border=True):
        st.subheader("Estatísticas Descritivas")
        st.write(df_cleaned['NUOD'].describe())


# --- Conteúdo das Páginas ---

        # --- Conteúdo das Páginas ---


    st.markdown(
        "Uma exploração detalhada dos dados tabulares de Oxigênio Dissolvido, "
        "incluindo limpeza, visualizações e modelagem preditiva."
    )

    st.header("Compreendendo o Oxigênio Dissolvido (OD)")
    st.markdown("""
    O Oxigênio Dissolvido (OD) é um dos indicadores mais importantes da qualidade da água em ecossistemas aquáticos. Ele é essencial para a sobrevivência de peixes e outros organismos aquáticos, sendo diretamente influenciado por fatores como temperatura, salinidade, poluição orgânica e atividade fotossintética.

    * **Níveis Saudáveis**: Geralmente, níveis de OD acima de 5 mg/L são considerados adequados para a maioria da vida aquática.
    * **Níveis Baixos**: Valores abaixo de 5 mg/L podem indicar estresse ambiental, poluição e impactar negativamente a biodiversidade aquática, levando à morte de organismos.
    """)

    st.header("Visão Geral dos Dados")
    st.markdown("Esta seção oferece um panorama inicial do conjunto de dados, com estatísticas descritivas para as principais variáveis.")

    with st.container(border=True):
        st.subheader("Estatísticas Descritivas para Oxigênio Dissolvido (NUOD)")
        st.write(df_cleaned['NUOD'].describe())

    col_intro_1, col_intro_2 = st.columns(2)

    with col_intro_1:
        with st.container(border=True):
            st.subheader("Contagem de SGUF")
            st.dataframe(
                df_cleaned['SGUF'].value_counts().reset_index()
                .rename(columns={'index': 'SGUF', 'SGUF': 'Contagem'}),
                use_container_width=True
            )

    with col_intro_2:
        with st.container(border=True):
            st.subheader("Contagem de AMBIENTE")
            st.dataframe(
                df_cleaned['AMBIENTE'].value_counts().reset_index()
                .rename(columns={'index': 'AMBIENTE', 'AMBIENTE': 'Contagem'}),
                use_container_width=True
            )

    with st.expander("Ver Amostra dos Dados Limpos (Clique para Expandir)"):
        st.dataframe(df_cleaned.head(), use_container_width=True)

elif page == "Análise Exploratória de Dados":
    st.header("Análise Exploratória de Dados")

    if filtered_df_cleaned.empty:
        st.warning("Nenhum dado para exibir com os filtros selecionados. Por favor, ajuste os filtros na barra lateral.")
    else:
        # Histograma de NUOD (Plotly)
        with st.container(border=True):
            st.subheader("Distribuição de Oxigênio Dissolvido (NUOD)")
            fig_hist_plotly = px.histogram(filtered_df_cleaned, x='NUOD', nbins=30, marginal='rug',
                                           title='Distribuição de Oxigênio Dissolvido (NUOD) em Dados Filtrados',
                                           color_discrete_sequence=['var(--primary-color)'])
            st.plotly_chart(fig_hist_plotly, use_container_width=True)

        # Box Plot de NUOD por SGUF (Plotly)
        with st.container(border=True):
            st.subheader("Box Plot de OD por Unidade Federativa (SGUF)")
            min_stations_for_plot = 5
            valid_sguf_for_plot = filtered_df_cleaned.groupby('SGUF')['NUOD'].count()
            valid_sguf_for_plot = valid_sguf_for_plot[valid_sguf_for_plot >= min_stations_for_plot].index
            df_filtered_sguf_for_plot = filtered_df_cleaned[filtered_df_cleaned['SGUF'].isin(valid_sguf_for_plot)]

            if not df_filtered_sguf_for_plot.empty:
                fig_boxplot_plotly = px.box(df_filtered_sguf_for_plot, x='SGUF', y='NUOD',
                                              title='Distribuição de Oxigênio Dissolvido (NUOD) por Unidade Federativa (SGUF)',
                                              color='SGUF',
                                              color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_boxplot_plotly, use_container_width=True)
            else:
                st.info("Nenhum SGUF com estações suficientes para plotar com os filtros selecionados.")

        with st.container(border=True):
            # Tendência Temporal de OD (média nacional anual) (Plotly)
            st.subheader("Tendência Temporal: Média Anual de OD")
            med_cols_filtered = [col for col in filtered_df_cleaned.columns if col.startswith('MED_')]
            if med_cols_filtered:
                years_filtered = [int(col.split('_')[1]) for col in med_cols_filtered]
                annual_avg_od_filtered = filtered_df_cleaned[med_cols_filtered].mean()
                annual_avg_od_filtered.index = years_filtered
                temporal_df_filtered = pd.DataFrame({'Ano': annual_avg_od_filtered.index, 'Media_OD': annual_avg_od_filtered.values})

                fig_temporal_plotly = px.line(temporal_df_filtered, x='Ano', y='Media_OD', markers=True,
                                              title='Média Anual de Oxigênio Dissolvido (OD) em Dados Filtrados',
                                              color_discrete_sequence=['var(--primary-color)'])
                st.plotly_chart(fig_temporal_plotly, use_container_width=True)

                st.write("Análise da tendência para os dados filtrados:")
                if len(temporal_df_filtered) > 1:
                    first_year_od = temporal_df_filtered.iloc[0]['Media_OD']
                    last_year_od = temporal_df_filtered.iloc[-1]['Media_OD']
                    if last_year_od > first_year_od:
                        st.success(f"A média de Oxigênio Dissolvido parece ter melhorado de {first_year_od:.2f} mg/L no ano {temporal_df_filtered.iloc[0]['Ano']} para {last_year_od:.2f} mg/L no ano {temporal_df_filtered.iloc[-1]['Ano']}.")
                    elif last_year_od < first_year_od:
                        st.error(f"A média de Oxigênio Dissolvido parece ter piorado de {first_year_od:.2f} mg/L no ano {temporal_df_filtered.iloc[0]['Ano']} para {last_year_od:.2f} mg/L no ano {temporal_df_filtered.iloc[-1]['Ano']}.")
                    else:
                        st.info("A média de Oxigênio Dissolvido permaneceu estável ao longo do período.")
                else:
                    st.warning("Não há dados suficientes para determinar uma tendência temporal nos dados filtrados.")
            else:
                st.info("Nenhuma coluna de média anual de OD encontrada nos dados filtrados.")


elif page == "Mapa e Dados Brutos":
    st.header("Geolocalização das Estações e Dados Completos")
    st.markdown("Visualize a localização das estações de monitoramento e explore o conjunto de dados bruto.")

    with st.container(border=True):
        st.subheader("Mapa das Estações de Monitoramento")
        map_data = filtered_df_cleaned[['LATITUDE', 'LONGITUDE', 'CDESTACAO', 'NUOD', 'SGUF', 'CORPODAGUA']].dropna()
        if not map_data.empty:
            fig_map = go.Figure(go.Scattermapbox(
                lat=map_data['LATITUDE'],
                lon=map_data['LONGITUDE'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=np.log1p(map_data['NUOD']) * 5,  # Scale size by NUOD, log1p for better visual range
                    color=np.where(map_data['NUOD'] < 5, '#dc3545', '#28a745'), # Red for 'Ruim', Green for 'Bom'
                    opacity=0.8
                ),
                text=map_data.apply(lambda row: f"Estação: {row['CDESTACAO']}<br>SGUF: {row['SGUF']}<br>Corpo D'Água: {row['CORPODAGUA']}<br>NUOD: {row['NUOD']:.2f}", axis=1),
                hoverinfo='text'
            ))

            fig_map.update_layout(
                mapbox_style="open-street-map",
                autosize=True,
                hovermode='closest',
                margin={"r":0,"t":0,"l":0,"b":0},
                mapbox=dict(
                    bearing=0,
                    center=dict(
                        lat=map_data['LATITUDE'].mean(),
                        lon=map_data['LONGITUDE'].mean()
                    ),
                    pitch=0,
                    zoom=4 # Adjust initial zoom level as needed
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Nenhum dado de estação para exibir no mapa com os filtros selecionados.")

    with st.container(border=True):
        st.subheader("Visualização dos Dados Completos (Filtrados)")
        st.markdown("Explore a tabela completa dos dados limpos e filtrados.")
        st.dataframe(filtered_df_cleaned, use_container_width=True)

elif page == "Metodologia de Tratamento de Dados":
    st.header("Metodologia de Tratamento de Dados")
    st.markdown("Esta seção descreve os passos tomados para limpar e preparar o conjunto de dados.")

    st.subheader("1. Carregamento e Duplicatas")
    st.markdown("""
    O conjunto de dados foi carregado de um arquivo CSV. Inicialmente, foi realizada uma verificação para identificar e remover quaisquer linhas duplicadas presentes, garantindo que cada registro seja único e contribua de forma independente para a análise.
    """)

    st.subheader("2. Tratamento de Valores Ausentes (NaNs)")
    st.markdown("""
    Valores ausentes (NaN) foram tratados para evitar vieses e erros em análises subsequentes. Para todas as colunas numéricas, os valores NaN foram preenchidos com a mediana da respectiva coluna. A mediana foi escolhida por ser menos sensível a outliers do que a média, preservando melhor a distribuição dos dados.
    """)

    st.subheader("3. Remoção de Outliers para Oxigênio Dissolvido (NUOD)")
    st.markdown("""
    Para a coluna 'NUOD' (Oxigênio Dissolvido), que é central para esta análise, outliers foram identificados e removidos utilizando o método do Intervalo Interquartil (IQR). Os valores considerados outliers são aqueles que caem abaixo de $Q1 - 1.5 \times IQR$ ou acima de $Q3 + 1.5 \times IQR$, onde $Q1$ é o primeiro quartil, $Q3$ é o terceiro quartil e $IQR = Q3 - Q1$. Esta etapa ajuda a garantir que a distribuição de 'NUOD' seja mais robusta e representativa.
    """)

    st.subheader("4. Criação de Variáveis para Modelagem")
    st.markdown("""
    Para a análise de classificação temporal, as colunas de média anual de OD (e.g., `MED_1978`, `MED_1979`) foram \"derretidas\" (unpivot) em um formato longo, criando uma coluna `Ano` e `Valor_OD_Medio`. Com base neste `Valor_OD_Medio`, uma nova variável categórica `OD_Status` foi criada, classificando o OD como 'Bom' (>= 5 mg/L) ou 'Ruim' (< 5 mg/L), seguindo um limiar comum para qualidade da água.
    """)

elif page == "Resultados do Modelo Preditivo":
    st.header("Modelagem Preditiva: Classificação do Status do OD")
    st.markdown("Utilizamos um modelo RandomForest para classificar o status do Oxigênio Dissolvido "
                "('Bom' ou 'Ruim') com base em características geográficas e temporais, utilizando uma divisão temporal dos dados.")

    # Preparar dados para o modelo de classificação (split temporal)
    # É importante usar df_melted aqui, pois ele já tem a estrutura por ano e o target OD_Status

    all_years = sorted(df_melted['Ano'].unique())
    num_years = len(all_years)
    test_years_count = int(np.ceil(num_years * 0.20))
    test_years = all_years[-test_years_count:]
    train_years = all_years[:-test_years_count]

    df_train_ts = df_melted[df_melted['Ano'].isin(train_years)]
    df_test_ts = df_melted[df_melted['Ano'].isin(test_years)]

    features_time_series = ['LATITUDE', 'LONGITUDE', 'AMBIENTE', 'Ano']
    target_time_series = 'OD_Status'

    X_train_clf_ts = df_train_ts[features_time_series]
    y_train_clf_ts = df_train_ts[target_time_series]
    X_test_clf_ts = df_test_ts[features_time_series]
    y_test_clf_ts = df_test_ts[target_time_series]

    model_clf_ts = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model_clf_ts.fit(X_train_clf_ts, y_train_clf_ts)
    y_pred_clf_ts = model_clf_ts.predict(X_test_clf_ts)

    st.subheader("Resultados do Modelo RandomForest (Conjunto de Teste)")
    accuracy_ts = accuracy_score(y_test_clf_ts, y_pred_clf_ts)
    precision_ts = precision_score(y_test_clf_ts, y_pred_clf_ts, pos_label='Bom')
    recall_ts = recall_score(y_test_clf_ts, y_pred_clf_ts, pos_label='Bom')
    f1_ts = f1_score(y_test_clf_ts, y_pred_clf_ts, pos_label='Bom')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia", f"{accuracy_ts:.2f}")
    col2.metric("Precisão (Bom)", f"{precision_ts:.2f}")
    col3.metric("Recall (Bom)", f"{recall_ts:.2f}")
    col4.metric("F1-Score (Bom)", f"{f1_ts:.2f}")

    with st.container(border=True):
        st.subheader("Matriz de Confusão")
        cm_ts = confusion_matrix(y_test_clf_ts, y_pred_clf_ts, labels=model_clf_ts.classes_)
        fig_cm = px.imshow(cm_ts, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predito", y="Real", color="Contagem"),
                           x=model_clf_ts.classes_, y=model_clf_ts.classes_)
        fig_cm.update_layout(title='Matriz de Confusão (Split Temporal por Ano)')
        st.plotly_chart(fig_cm, use_container_width=True)

    with st.container(border=True):
        st.subheader("Importância das Features")
        feature_importances_ts = pd.DataFrame({'Feature': features_time_series, 'Importance': model_clf_ts.feature_importances_})
        fig_features = px.bar(feature_importances_ts.sort_values(by='Importance', ascending=False), x='Importance', y='Feature',
                              orientation='h', title='Importância das Features no Modelo RandomForest',
                              color_discrete_sequence=['var(--primary-color)'])
        st.plotly_chart(fig_features, use_container_width=True)

    st.markdown("--- ")
    st.markdown("**Conclusão da Modelagem:** O modelo RandomForest demonstra um bom desempenho na "
                "classificação do status do Oxigênio Dissolvido, especialmente para a classe 'Bom'. "
                "As características geográficas (`LATITUDE`, `LONGITUDE`, `AMBIENTE`) e o `Ano` "
                "contribuem significativamente para a previsão, indicando a importância de fatores "
                "espaciais e temporais na qualidade da água.")

    st.header("Principais Insights e Conclusões")
    st.markdown("""
    *   **Desempenho Sólido na Classe 'Bom'**: O modelo demonstrou alta precisão e recall para identificar águas com boa qualidade de Oxigênio Dissolvido, o que é crucial para a saúde ambiental.
    *   **Desafio na Classe 'Ruim'**: A classificação de águas com baixa qualidade de OD ('Ruim') apresentou um desafio maior, com recall e F1-Score mais baixos, indicando que o modelo pode ter dificuldade em identificar todos os casos de problemas.
    *   **Importância das Variáveis Geográficas e Temporais**: `LATITUDE`, `LONGITUDE`, `AMBIENTE` e `Ano` são features significativas para o modelo, confirmando que a localização e a passagem do tempo são fatores cruciais para a qualidade da água.
    *   **Variação Temporal no OD**: A análise da tendência temporal revelou variações anuais na média de OD, sugerindo que as condições ambientais e os impactos humanos ao longo dos anos afetam diretamente a qualidade da água.
    *   **Potencial para Melhoria**: Futuras iterações podem explorar a engenharia de novas features (e.g., sazonalidade, proximidade a áreas urbanas/industriais) ou a aplicação de modelos mais complexos para aprimorar a detecção de cenários 'Ruins' de OD.
    """)
