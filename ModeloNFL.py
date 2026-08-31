import streamlit as st
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from scipy.stats import norm

st.set_page_config(page_title="NFL Analytics Pro", page_icon="🏈", layout="wide")

# --- 1. EXTRACCIÓN Y PROCESAMIENTO DE DATOS ---
@st.cache_data
def cargar_datos_completos(anio):
    games = nfl.import_schedules([anio])
    played = games[games['home_score'].notna()].copy()
    unplayed = games[games['home_score'].isna()].copy()
    
    semana_actual = unplayed['week'].min() if not unplayed.empty else None
    
    try:
        weekly = nfl.import_weekly_data([anio])
    except:
        weekly = pd.DataFrame()
        
    if not played.empty:
        home_df = played[['week', 'home_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'home_score': 'pts_scored', 'away_score': 'pts_allowed'})
        away_df = played[['week', 'away_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'away_score': 'pts_scored', 'home_score': 'pts_allowed'})
        all_games = pd.concat([home_df, away_df]).sort_values(by=['team', 'week'])
        
        team_stats = all_games.groupby('team').agg(Off_Pts=('pts_scored', 'mean'), Def_Pts=('pts_allowed', 'mean'), Std_Dev=('pts_scored', 'std')).reset_index()
        momentum = all_games.groupby('team').tail(4).groupby('team').agg(Mom_Off=('pts_scored', 'mean'), Mom_Def=('pts_allowed', 'mean')).reset_index()
        team_stats = pd.merge(team_stats, momentum, on='team')
    else:
        team_stats = pd.DataFrame()
        
    if not weekly.empty:
        to_data = weekly.groupby('recent_team').agg(
            Ints=('interceptions', 'sum'),
            Fumbles=('fumbles_lost', 'sum')
        ).reset_index().rename(columns={'recent_team': 'team'})
        to_data['Turnovers_Per_Game'] = (to_data['Ints'] + to_data['Fumbles']) / (played['week'].max() if not played.empty else 1)
        
        if not team_stats.empty:
            team_stats = pd.merge(team_stats, to_data, on='team', how='left')
    
    rest_dict = {}
    if not played.empty and 'gameday' in played.columns:
        played['gameday'] = pd.to_datetime(played['gameday'])
        for team in team_stats['team'].unique() if not team_stats.empty else []:
            team_games = played[(played['home_team'] == team) | (played['away_team'] == team)].sort_values('gameday')
            if not team_games.empty:
                rest_dict[team] = team_games['gameday'].iloc[-1]
            
    lista_equipos = sorted(team_stats['team'].unique()) if not team_stats.empty else []
    team_dict = team_stats.set_index('team').to_dict('index') if not team_stats.empty else {}
    league_avg = all_games['pts_scored'].mean() if not team_stats.empty else 21.5

    return team_dict, league_avg, lista_equipos, unplayed, semana_actual, weekly, rest_dict

@st.cache_data
def obtener_historial():
    historial = nfl.import_schedules(list(range(2015, 2027)))
    return historial[historial['home_score'].notna()]

# --- 2. INTERFAZ GRÁFICA Y PESTAÑAS ---
st.title("🏈 Modelo Predictivo Avanzado y Player Props")

with st.sidebar:
    st.header("⚙️ Configuración General")
    anio_seleccionado = st.selectbox("Temporada:", [2026, 2025, 2024, 2023], index=1)
    hfa = st.slider("Ventaja de Localía (Pts)", 0.0, 5.0, 2.0, 0.5)

with st.spinner('Procesando estadísticas avanzadas, play-by-play y Player Props...'):
    team_dict, league_avg, lista_equipos, unplayed, semana_actual, weekly_data, rest_dict = cargar_datos_completos(anio_seleccionado)
    historial_df = obtener_historial()

if not team_dict:
    st.warning("No hay suficientes datos procesados para esta temporada aún. Intenta con un año anterior.")
    st.stop()

tab1, tab2 = st.tabs(["🏆 Predicción del Partido", "📈 Player Props (Probabilidades)"])

# ==========================================
# PESTAÑA 1: PREDICCIÓN DE PARTIDO
# ==========================================
with tab1:
    modo_seleccion = st.radio("Método de Selección de Partido:", ["📅 Próximos Partidos (Semana Actual)", "🔍 Selección Manual"], horizontal=True)
    
    eq_local, eq_vis = None, None
    
    if modo_seleccion == "📅 Próximos Partidos (Semana Actual)":
        partidos_semana = unplayed[unplayed['week'] == semana_actual] if semana_actual else pd.DataFrame()
        if not partidos_semana.empty:
            st.subheader(f"Cartelera: Semana {int(semana_actual)}")
            opciones = [f"{row['away_team']} @ {row['home_team']}" for _, row in partidos_semana.iterrows()]
            partido_elegido = st.selectbox("Selecciona el encuentro:", opciones)
            
            partes = partido_elegido.split(" @ ")
            eq_vis, eq_local = partes[0], partes[1]
        else:
            st.info("No hay partidos pendientes detectados. Cambiando a selección manual.")
            modo_seleccion = "🔍 Selección Manual"

    if modo_seleccion == "🔍 Selección Manual":
        col1, col2 = st.columns(2)
        with col1:
            eq_local = st.selectbox("Equipo Local:", lista_equipos, index=lista_equipos.index("KC") if "KC" in lista_equipos else 0)
        with col2:
            eq_vis = st.selectbox("Equipo Visitante:", lista_equipos, index=lista_equipos.index("BAL") if "BAL" in lista_equipos else 1)

    if eq_local and eq_vis:
        # Checkboxes de lesiones dinámicos según los equipos seleccionados
        st.markdown("---")
        st.write("🚑 **Reporte de Lesiones (Opcional):**")
        col_inj1, col_inj2 = st.columns(2)
        with col_inj1:
            qb_local_out = st.checkbox(f"QB Titular de {eq_local} NO jugará")
        with col_inj2:
            qb_vis_out = st.checkbox(f"QB Titular de {eq_vis} NO jugará")

        if st.button("🚀 Ejecutar Modelo de Simulación", type="primary"):
            if eq_local == eq_vis:
                st.error("Selecciona equipos distintos.")
            else:
                off_local = ((team_dict[eq_local]["Off_Pts"] + team_dict[eq_local]["Mom_Off"]) / 2) / league_avg
                def_vis = ((team_dict[eq_vis]["Def_Pts"] + team_dict[eq_vis]["Mom_Def"]) / 2) / league_avg
                
                off_vis = ((team_dict[eq_vis]["Off_Pts"] + team_dict[eq_vis]["Mom_Off"]) / 2) / league_avg
                def_local = ((team_dict[eq_local]["Def_Pts"] + team_dict[eq_local]["Mom_Def"]) / 2) / league_avg

                exp_local = (league_avg * off_local * def_vis) + hfa
                exp_vis = (league_avg * off_vis * def_local)

                matchups = historial_df[((historial_df['home_team'] == eq_local) & (historial_df['away_team'] == eq_vis)) | ((historial_df['home_team'] == eq_vis) & (historial_df['away_team'] == eq_local))]
                w_loc = sum((matchups['home_team'] == eq_local) & (matchups['home_score'] > matchups['away_score'])) + sum((matchups['away_team'] == eq_local) & (matchups['away_score'] > matchups['home_score']))
                w_vis = sum((matchups['home_team'] == eq_vis) & (matchups['home_score'] > matchups['away_score'])) + sum((matchups['away_team'] == eq_vis) & (matchups['away_score'] > matchups['home_score']))
                
                dif_h2h = w_loc - w_vis
                exp_local += (dif_h2h * 0.25) if dif_h2h > 0 else 0
                exp_vis += (abs(dif_h2h) * 0.25) if dif_h2h < 0 else 0

                to_margin = team_dict[eq_vis].get('Turnovers_Per_Game', 1.5) - team_dict[eq_local].get('Turnovers_Per_Game', 1.5)
                exp_local += (to_margin * 0.5) 
                
                # Penalización Analítica por Lesión
                std_local = np.nan_to_num(team_dict[eq_local]["Std_Dev"], nan=7.0)
                std_vis = np.nan_to_num(team_dict[eq_vis]["Std_Dev"], nan=7.0)

                penalidad_calculada_local = std_local * 0.7 if qb_local_out else 0
                penalidad_calculada_vis = std_vis * 0.7 if qb_vis_out else 0

                exp_local -= penalidad_calculada_local
                exp_vis -= penalidad_calculada_vis
                
                exp_local = max(exp_local, 0)
                exp_vis = max(exp_vis, 0)

                sim_local = np.where(np.round(np.random.normal(exp_local, std_local, 10000)) < 0, 0, np.round(np.random.normal(exp_local, std_local, 10000)))
                sim_vis = np.where(np.round(np.random.normal(exp_vis, std_vis, 10000)) < 0, 0, np.round(np.random.normal(exp_vis, std_vis, 10000)))

                st.markdown("### 📊 Resultados de Simulación Avanzada")
                
                if qb_local_out or qb_vis_out:
                    st.info(f"**Ajuste por Lesiones Aplicado:** El modelo restó automáticamente {penalidad_calculada_local:.1f} pts a {eq_local} y/o {penalidad_calculada_vis:.1f} pts a {eq_vis} por ausencia de QB, basado en la volatilidad de su ofensiva.")

                c1, c2, c3 = st.columns(3)
                c1.metric(f"Prob. {eq_local}", f"{(np.sum(sim_local > sim_vis) / 10000) * 100:.1f}%", f"{exp_local:.1f} Pts Esperados")
                c2.metric("Empate", f"{(np.sum(sim_local == sim_vis) / 10000) * 100:.1f}%")
                c3.metric(f"Prob. {eq_vis}", f"{(np.sum(sim_vis > sim_local) / 10000) * 100:.1f}%", f"{exp_vis:.1f} Pts Esperados")
                
                c4, c5 = st.columns(2)
                c4.metric("Over/Under Proyectado", f"{np.median(sim_local + sim_vis):.1f} pts")
                spread = exp_vis - exp_local
                c5.metric("Línea de Spread Proyectada", f"{eq_local} {spread:.1f}" if spread < 0 else f"{eq_vis} -{spread:.1f}")

# ==========================================
# PESTAÑA 2: PLAYER PROPS
# ==========================================
with tab2:
    if weekly_data.empty:
        st.warning("No hay datos de jugadores disponibles para esta temporada.")
    else:
        col_t, col_p = st.columns(2)
        with col_t:
            equipo_prop = st.selectbox("Selecciona Equipo:", lista_equipos)
            
        roster = weekly_data[weekly_data['recent_team'] == equipo_prop]
        jugadores = sorted(roster['player_display_name'].dropna().unique())
        
        with col_p:
            jugador = st.selectbox("Selecciona Jugador:", jugadores)
            
        player_stats = roster[roster['player_display_name'] == jugador]
        
        if not player_stats.empty:
            posicion = player_stats['position'].iloc[0]
            st.markdown(f"### 📈 Líneas de Probabilidad para {jugador} ({posicion})")
            
            metricas = []
            if posicion == 'QB':
                metricas = [('passing_yards', 'Yardas por Pase'), ('attempts', 'Intentos de Pase')]
            elif posicion in ['WR', 'TE']:
                metricas = [('receiving_yards', 'Yardas de Recepción'), ('receptions', 'Recepciones')]
            elif posicion == 'RB':
                metricas = [('rushing_yards', 'Yardas Terrestres'), ('carries', 'Acarreos (Intentos)')]
            else:
                metricas = [('receiving_yards', 'Yardas Totales'), ('receptions', 'Recepciones')]

            probabilidades = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
            etiquetas_prob = ["99% (Muy Seguro)", "90%", "80%", "70%", "60%", "50% (Promedio)", "40%", "30%", "20%", "10% (Arriesgado)"]

            df_props = pd.DataFrame({"Probabilidad (OVER)": etiquetas_prob})

            for col_stat, nombre_stat in metricas:
                data_stat = player_stats[col_stat].dropna()
                if len(data_stat) < 2:
                    st.info(f"No hay suficientes juegos registrados para calcular {nombre_stat}.")
                    continue
                
                mu = data_stat.mean()
                sigma = data_stat.std()
                if sigma == 0: sigma = 0.1
                
                lineas_calculadas = [max(0, np.round(norm.ppf(1 - p, loc=mu, scale=sigma), 1)) for p in probabilidades]
                df_props[f"Línea de {nombre_stat}"] = lineas_calculadas
                
                st.write(f"**{nombre_stat}:** Promedio {mu:.1f} | Desviación Estándar {sigma:.1f}")

            st.dataframe(df_props, use_container_width=True, hide_index=True)