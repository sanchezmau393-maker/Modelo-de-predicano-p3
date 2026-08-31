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
    
    anio_stats = anio
    if played.empty:
        anio_stats = anio - 1
        games_prev = nfl.import_schedules([anio_stats])
        played_stats = games_prev[games_prev['home_score'].notna()].copy()
    else:
        played_stats = played.copy()

    try:
        weekly = nfl.import_weekly_data([anio_stats])
    except:
        weekly = pd.DataFrame()
        
    if not played_stats.empty:
        home_df = played_stats[['week', 'home_team', 'home_score', 'away_score']].rename(columns={'home_team': 'team', 'home_score': 'pts_scored', 'away_score': 'pts_allowed'})
        away_df = played_stats[['week', 'away_team', 'away_score', 'home_score']].rename(columns={'away_team': 'team', 'away_score': 'pts_scored', 'home_score': 'pts_allowed'})
        all_games = pd.concat([home_df, away_df]).sort_values(by=['team', 'week'])
        
        team_stats = all_games.groupby('team').agg(Off_Pts=('pts_scored', 'mean'), Def_Pts=('pts_allowed', 'mean'), Std_Dev=('pts_scored', 'std')).reset_index()
        momentum = all_games.groupby('team').tail(4).groupby('team').agg(Mom_Off=('pts_scored', 'mean'), Mom_Def=('pts_allowed', 'mean')).reset_index()
        team_stats = pd.merge(team_stats, momentum, on='team')
    else:
        team_stats = pd.DataFrame()
        
    if not weekly.empty:
        to_data = weekly.groupby('recent_team').agg(Ints=('interceptions', 'sum'), Fumbles=('fumbles_lost', 'sum')).reset_index().rename(columns={'recent_team': 'team'})
        to_data['Turnovers_Per_Game'] = (to_data['Ints'] + to_data['Fumbles']) / (played_stats['week'].max() if not played_stats.empty else 1)
        if not team_stats.empty:
            team_stats = pd.merge(team_stats, to_data, on='team', how='left')
    
    lista_equipos = sorted(team_stats['team'].unique()) if not team_stats.empty else []
    team_dict = team_stats.set_index('team').to_dict('index') if not team_stats.empty else {}
    league_avg = all_games['pts_scored'].mean() if not team_stats.empty else 21.5

    return team_dict, league_avg, lista_equipos, unplayed, semana_actual, weekly, anio_stats

@st.cache_data
def obtener_historial_y_qbs():
    historial = nfl.import_schedules(list(range(2015, 2027)))
    historial_jugado = historial[historial['home_score'].notna()]
    
    try:
        # Cargar historial de jugadores de los últimos 4 años para evaluar QBs contra rivales
        weekly_history = nfl.import_weekly_data([2023, 2024, 2025, 2026])
    except:
        weekly_history = pd.DataFrame()
        
    return historial_jugado, weekly_history

def obtener_qb_titular(equipo, df_weekly):
    if df_weekly.empty: return None
    qbs = df_weekly[(df_weekly['recent_team'] == equipo) & (df_weekly['position'] == 'QB')]
    if not qbs.empty:
        return qbs.groupby('player_display_name')['attempts'].sum().idxmax()
    return None

# --- 2. INTERFAZ GRÁFICA Y PESTAÑAS ---
st.title("🏈 Modelo Predictivo Avanzado y Player Props")

with st.sidebar:
    st.header("⚙️ Configuración General")
    anio_seleccionado = st.selectbox("Temporada a visualizar:", [2026, 2025, 2024, 2023], index=0)
    hfa = st.slider("Ventaja de Localía (Pts)", 0.0, 5.0, 2.0, 0.5)

with st.spinner('Procesando estadísticas, rosters y analítica de QBs (esto toma unos segundos)...'):
    team_dict, league_avg, lista_equipos, unplayed, semana_actual, weekly_data, anio_usado = cargar_datos_completos(anio_seleccionado)
    historial_df, weekly_history = obtener_historial_y_qbs()

if anio_seleccionado != anio_usado:
    st.info(f"📅 **Aviso de Temporada Nueva:** Evaluando fuerzas de los equipos con datos y rosters consolidados de {anio_usado}.")

if not team_dict:
    st.warning("No hay suficientes datos procesados. Intenta con un año anterior.")
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
        st.markdown("---")
        
        # --- MÓDULO DINÁMICO DE LESIONES ---
        st.write("🚑 **Reporte de Lesiones (Multiselección)**")
        col_inj1, col_inj2 = st.columns(2)
        
        roster_local = sorted(weekly_data[weekly_data['recent_team'] == eq_local]['player_display_name'].dropna().unique()) if not weekly_data.empty else []
        roster_vis = sorted(weekly_data[weekly_data['recent_team'] == eq_vis]['player_display_name'].dropna().unique()) if not weekly_data.empty else []
        
        with col_inj1:
            lesionados_local = st.multiselect(f"Lesionados en {eq_local}:", roster_local, placeholder="Selecciona jugadores descartados...")
        with col_inj2:
            lesionados_vis = st.multiselect(f"Lesionados en {eq_vis}:", roster_vis, placeholder="Selecciona jugadores descartados...")

        # --- LÍNEA DE VEGAS ---
        st.write("📊 **Análisis Over/Under**")
        linea_vegas = st.number_input("Ingresa Línea de Las Vegas para recomendación:", min_value=20.0, max_value=70.0, value=45.5, step=0.5)

        if st.button("🚀 Ejecutar Modelo de Simulación", type="primary"):
            if eq_local == eq_vis:
                st.error("Selecciona equipos distintos.")
            else:
                std_local = np.nan_to_num(team_dict[eq_local]["Std_Dev"], nan=7.0)
                std_vis = np.nan_to_num(team_dict[eq_vis]["Std_Dev"], nan=7.0)

                # 1. Fuerza Relativa Base
                off_local = ((team_dict[eq_local]["Off_Pts"] + team_dict[eq_local]["Mom_Off"]) / 2) / league_avg
                def_vis = ((team_dict[eq_vis]["Def_Pts"] + team_dict[eq_vis]["Mom_Def"]) / 2) / league_avg
                off_vis = ((team_dict[eq_vis]["Off_Pts"] + team_dict[eq_vis]["Mom_Off"]) / 2) / league_avg
                def_local = ((team_dict[eq_local]["Def_Pts"] + team_dict[eq_local]["Mom_Def"]) / 2) / league_avg

                exp_local = (league_avg * off_local * def_vis) + hfa
                exp_vis = (league_avg * off_vis * def_local)

                # 2. Historial de Franquicias (H2H)
                matchups = historial_df[((historial_df['home_team'] == eq_local) & (historial_df['away_team'] == eq_vis)) | ((historial_df['home_team'] == eq_vis) & (historial_df['away_team'] == eq_local))]
                w_loc = sum((matchups['home_team'] == eq_local) & (matchups['home_score'] > matchups['away_score'])) + sum((matchups['away_team'] == eq_local) & (matchups['away_score'] > matchups['home_score']))
                w_vis = sum((matchups['home_team'] == eq_vis) & (matchups['home_score'] > matchups['away_score'])) + sum((matchups['away_team'] == eq_vis) & (matchups['away_score'] > matchups['home_score']))
                dif_h2h = w_loc - w_vis
                exp_local += (dif_h2h * 0.25) if dif_h2h > 0 else 0
                exp_vis += (abs(dif_h2h) * 0.25) if dif_h2h < 0 else 0

                # 3. Diferencial de Entregas
                to_margin = team_dict[eq_vis].get('Turnovers_Per_Game', 1.5) - team_dict[eq_local].get('Turnovers_Per_Game', 1.5)
                exp_local += (to_margin * 0.5) 

                # 4. Impacto Histórico del QB Titular vs Oponente
                qb_local = obtener_qb_titular(eq_local, weekly_data)
                qb_vis = obtener_qb_titular(eq_vis, weekly_data)
                
                texto_qb_local, texto_qb_vis = "", ""

                if not weekly_history.empty:
                    if qb_local and qb_local not in lesionados_local:
                        hist_qb_l = weekly_history[(weekly_history['player_display_name'] == qb_local) & (weekly_history['opponent_team'] == eq_vis)]
                        if not hist_qb_l.empty:
                            epa_medio = hist_qb_l['passing_epa'].mean()
                            bono_qb_l = max(-2.0, min(2.0, epa_medio * 0.2)) # Topado a +- 2 pts
                            exp_local += bono_qb_l
                            texto_qb_local = f"✅ **{qb_local} vs {eq_vis}**: Historial detectado. Ajuste al marcador: {bono_qb_l:+.1f} pts."
                    
                    if qb_vis and qb_vis not in lesionados_vis:
                        hist_qb_v = weekly_history[(weekly_history['player_display_name'] == qb_vis) & (weekly_history['opponent_team'] == eq_local)]
                        if not hist_qb_v.empty:
                            epa_medio = hist_qb_v['passing_epa'].mean()
                            bono_qb_v = max(-2.0, min(2.0, epa_medio * 0.2))
                            exp_vis += bono_qb_v
                            texto_qb_vis = f"✅ **{qb_vis} vs {eq_local}**: Historial detectado. Ajuste al marcador: {bono_qb_v:+.1f} pts."

                # 5. Cálculo Dinámico de Penalización por Lesiones
                penalidad_l, penalidad_v = 0, 0
                for jugador in lesionados_local:
                    pos = weekly_data[(weekly_data['recent_team'] == eq_local) & (weekly_data['player_display_name'] == jugador)]['position'].iloc[0]
                    if pos == 'QB': penalidad_l += (std_local * 0.7)
                    elif pos in ['WR', 'RB', 'TE']: penalidad_l += 1.5
                    else: penalidad_l += 0.5
                
                for jugador in lesionados_vis:
                    pos = weekly_data[(weekly_data['recent_team'] == eq_vis) & (weekly_data['player_display_name'] == jugador)]['position'].iloc[0]
                    if pos == 'QB': penalidad_v += (std_vis * 0.7)
                    elif pos in ['WR', 'RB', 'TE']: penalidad_v += 1.5
                    else: penalidad_v += 0.5

                exp_local = max(exp_local - penalidad_l, 0)
                exp_vis = max(exp_vis - penalidad_v, 0)

                # Simulación Monte Carlo
                sim_local = np.where(np.round(np.random.normal(exp_local, std_local, 10000)) < 0, 0, np.round(np.random.normal(exp_local, std_local, 10000)))
                sim_vis = np.where(np.round(np.random.normal(exp_vis, std_vis, 10000)) < 0, 0, np.round(np.random.normal(exp_vis, std_vis, 10000)))

                st.markdown("### 📊 Resultados de Simulación Avanzada")
                
                # Reportes Automáticos
                if penalidad_l > 0 or penalidad_v > 0:
                    st.warning(f"🩹 **Impacto por Lesiones:** -{penalidad_l:.1f} pts para {eq_local} | -{penalidad_v:.1f} pts para {eq_vis}")
                if texto_qb_local or texto_qb_vis:
                    st.info(f"{texto_qb_local}\n\n{texto_qb_vis}")

                # --- MEDIDOR DE CONFIANZA ---
                prob_loc_win = np.sum(sim_local > sim_vis) / 10000
                prob_vis_win = np.sum(sim_vis > sim_local) / 10000
                max_prob = max(prob_loc_win, prob_vis_win)

                if max_prob >= 0.75:
                    confianza, color_hex = "🔵 MUY SEGURO (Pick Fuerte)", "#1E88E5"
                elif max_prob >= 0.65:
                    confianza, color_hex = "🟢 CONFIABLE", "#43A047"
                elif max_prob >= 0.55:
                    confianza, color_hex = "🟡 RIESGO MEDIO", "#FDD835"
                else:
                    confianza, color_hex = "🔴 RIESGO ALTO (Moneda al aire)", "#E53935"

                st.markdown(f"**Nivel de Confianza del Modelo:** <span style='color:{color_hex}; font-weight:bold; font-size:1.2em;'>{confianza}</span>", unsafe_allow_html=True)
                
                # Fila 1: Probabilidades
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Prob. {eq_local}", f"{prob_loc_win * 100:.1f}%")
                c2.metric("Empate", f"{(np.sum(sim_local == sim_vis) / 10000) * 100:.1f}%")
                c3.metric(f"Prob. {eq_vis}", f"{prob_vis_win * 100:.1f}%")
                
                st.divider()
                
                # Fila 2: Marcadores y Spread
                c4, c5, c6 = st.columns(3)
                c4.metric("Marcador Esperado Exacto", f"{exp_local:.1f} - {exp_vis:.1f}")
                
                spread = exp_vis - exp_local
                c5.metric("Línea de Spread Proyectada", f"{eq_local} {spread:.1f}" if spread < 0 else f"{eq_vis} -{spread:.1f}")
                
                # Fila 3: Análisis Over/Under
                prob_over = np.sum((sim_local + sim_vis) > linea_vegas) / 10000
                prob_under = np.sum((sim_local + sim_vis) < linea_vegas) / 10000
                
                rec_texto = "🔥 OVER" if prob_over > prob_under else "🧊 UNDER"
                rec_prob = max(prob_over, prob_under) * 100

                c6.metric(f"Recomendación O/U (Línea {linea_vegas})", rec_texto, f"Probabilidad: {rec_prob:.1f}%")

# ==========================================
# PESTAÑA 2: PLAYER PROPS
# ==========================================
with tab2:
    if weekly_data.empty:
        st.warning("No hay datos de jugadores disponibles.")
    else:
        col_t, col_p = st.columns(2)
        with col_t:
            equipo_prop = st.selectbox("Selecciona Equipo:", lista_equipos)
            
        roster_props = weekly_data[weekly_data['recent_team'] == equipo_prop]
        jugadores = sorted(roster_props['player_display_name'].dropna().unique())
        
        with col_p:
            jugador = st.selectbox("Selecciona Jugador:", jugadores)
            
        player_stats = roster_props[roster_props['player_display_name'] == jugador]
        
        if not player_stats.empty:
            posicion = player_stats['position'].iloc[0]
            st.markdown(f"### 📈 Líneas de Probabilidad para {jugador} ({posicion})")
            
            metricas = []
            if posicion == 'QB': metricas = [('passing_yards', 'Yardas por Pase'), ('attempts', 'Intentos de Pase')]
            elif posicion in ['WR', 'TE']: metricas = [('receiving_yards', 'Yardas de Recepción'), ('receptions', 'Recepciones')]
            elif posicion == 'RB': metricas = [('rushing_yards', 'Yardas Terrestres'), ('carries', 'Acarreos (Intentos)')]
            else: metricas = [('receiving_yards', 'Yardas Totales'), ('receptions', 'Recepciones')]

            probabilidades = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
            etiquetas_prob = ["99% (Muy Seguro)", "90%", "80%", "70%", "60%", "50% (Promedio)", "40%", "30%", "20%", "10% (Arriesgado)"]

            df_props = pd.DataFrame({"Probabilidad (OVER)": etiquetas_prob})

            for col_stat, nombre_stat in metricas:
                data_stat = player_stats[col_stat].dropna()
                if len(data_stat) < 2: continue
                
                mu, sigma = data_stat.mean(), data_stat.std()
                if sigma == 0: sigma = 0.1
                
                lineas_calculadas = [max(0, np.round(norm.ppf(1 - p, loc=mu, scale=sigma), 1)) for p in probabilidades]
                df_props[f"Línea de {nombre_stat}"] = lineas_calculadas
                st.write(f"**{nombre_stat}:** Promedio {mu:.1f} | Desviación Estándar {sigma:.1f}")

            st.dataframe(df_props, use_container_width=True, hide_index=True)