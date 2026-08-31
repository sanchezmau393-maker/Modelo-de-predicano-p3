import streamlit as st
import numpy as np
import pandas as pd
import nfl_data_py as nfl

st.set_page_config(page_title="Predicciones NFL", page_icon="🏈", layout="wide")

# 1. CARGA DE DATOS Y ESTADÍSTICAS
@st.cache_data
def cargar_datos_nfl(anio):
    # Importamos el calendario del año seleccionado
    games = nfl.import_schedules([anio])
    
    # Separamos partidos jugados y por jugar
    played_games = games[games['home_score'].notna()].copy()
    unplayed_games = games[games['home_score'].isna()].copy()
    
    # Detectar la semana actual (la primera semana con juegos sin resultado)
    semana_actual = unplayed_games['week'].min() if not unplayed_games.empty else None
    partidos_semana = unplayed_games[unplayed_games['week'] == semana_actual] if semana_actual else pd.DataFrame()
    
    # Si la temporada no ha empezado, usamos estadísticas del año anterior para poder predecir
    if played_games.empty:
        games_prev = nfl.import_schedules([anio - 1])
        played_games = games_prev[games_prev['home_score'].notna()].copy()
        st.sidebar.info(f"🏈 Temporada nueva: Usando estadísticas de {anio-1} para evaluar equipos.")

    # Reestructuramos datos para calcular estadísticas
    home_df = played_games[['home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'pts_scored', 'away_score': 'pts_allowed'})
    away_df = played_games[['away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'pts_scored', 'home_score': 'pts_allowed'})
    
    all_games = pd.concat([home_df, away_df])
    league_avg = all_games['pts_scored'].mean()
    
    team_stats = all_games.groupby('team').agg(
        Off_Pts=('pts_scored', 'mean'),
        Def_Pts=('pts_allowed', 'mean'),
        Std_Dev=('pts_scored', 'std')
    ).reset_index()
    
    team_stats['Std_Dev'] = team_stats['Std_Dev'].fillna(7.0)
    nfl_teams = team_stats.set_index('team').to_dict('index')
    lista_equipos = sorted(team_stats['team'].unique())
    
    return nfl_teams, league_avg, lista_equipos, partidos_semana, semana_actual

# 2. CARGA DE HISTORIAL DIRECTO (H2H)
@st.cache_data
def obtener_historial():
    # Cargamos datos desde 2015 para el historial
    historial = nfl.import_schedules(list(range(2015, 2027)))
    return historial[historial['home_score'].notna()]

def calcular_h2h(historial_df, eq_local, eq_visitante):
    # Filtramos partidos donde jugaron estos dos equipos
    matchups = historial_df[
        ((historial_df['home_team'] == eq_local) & (historial_df['away_team'] == eq_visitante)) |
        ((historial_df['home_team'] == eq_visitante) & (historial_df['away_team'] == eq_local))
    ]
    
    wins_local, wins_visitante, empates = 0, 0, 0
    for _, row in matchups.iterrows():
        if row['home_team'] == eq_local:
            if row['home_score'] > row['away_score']: wins_local += 1
            elif row['home_score'] < row['away_score']: wins_visitante += 1
            else: empates += 1
        else:
            if row['away_score'] > row['home_score']: wins_local += 1
            elif row['away_score'] < row['home_score']: wins_visitante += 1
            else: empates += 1
            
    return wins_local, wins_visitante, empates, len(matchups)

# 3. MODELO PREDICTIVO MEJORADO
def predecir_partido_nfl(nfl_teams, league_avg, eq_local, eq_visitante, hfa, wins_local, wins_visit, simulaciones=10000):
    off_strength_local = nfl_teams[eq_local]["Off_Pts"] / league_avg
    def_strength_visit = nfl_teams[eq_visitante]["Def_Pts"] / league_avg
    off_strength_visit = nfl_teams[eq_visitante]["Off_Pts"] / league_avg
    def_strength_local = nfl_teams[eq_local]["Def_Pts"] / league_avg

    # Base: Fuerza relativa y Ventaja de localía
    exp_pts_local = (league_avg * off_strength_local * def_strength_visit) + hfa
    exp_pts_visitante = (league_avg * off_strength_visit * def_strength_local)
    
    # IMPACTO HISTÓRICO: +0.25 pts esperados por cada victoria neta sobre el rival en el historial reciente
    diferencia_historica = wins_local - wins_visit
    if diferencia_historica > 0:
        exp_pts_local += (diferencia_historica * 0.25)
    elif diferencia_historica < 0:
        exp_pts_visitante += (abs(diferencia_historica) * 0.25)

    std_local = nfl_teams[eq_local]["Std_Dev"]
    std_visitante = nfl_teams[eq_visitante]["Std_Dev"]

    # Monte Carlo
    sim_local = np.round(np.random.normal(exp_pts_local, std_local, simulaciones))
    sim_visitante = np.round(np.random.normal(exp_pts_visitante, std_visitante, simulaciones))

    sim_local = np.where(sim_local < 0, 0, sim_local)
    sim_visitante = np.where(sim_visitante < 0, 0, sim_visitante)

    prob_local = np.sum(sim_local > sim_visitante) / simulaciones
    prob_visitante = np.sum(sim_visitante > sim_local) / simulaciones
    prob_empate = np.sum(sim_local == sim_visitante) / simulaciones
    
    over_under = np.median(sim_local + sim_visitante)
    
    return prob_local, prob_visitante, prob_empate, exp_pts_local, exp_pts_visitante, over_under

# 4. INTERFAZ GRÁFICA
st.title("🏈 Modelo Predictivo NFL Avanzado")
st.write("Fuerza Relativa, Historial Directo (H2H) y Monte Carlo.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    anio_seleccionado = st.selectbox("Temporada:", [2026, 2025, 2024, 2023], index=0)
    hfa = st.slider("Ventaja de Localía (Puntos)", 0.0, 5.0, 2.0, 0.5)

# --- CARGA DE DATOS ---
with st.spinner(f'Descargando datos... esto toma unos segundos la primera vez.'):
    nfl_teams, league_avg, lista_equipos, partidos_semana, semana_actual = cargar_datos_nfl(anio_seleccionado)
    historial_df = obtener_historial()

# --- SELECTOR DE PARTIDO ---
modo_seleccion = st.radio("Método de Selección:", ["📅 Próximos Partidos (Semana Actual)", "🔍 Selección Manual"], horizontal=True)

equipo_local, equipo_visitante = None, None

if modo_seleccion == "📅 Próximos Partidos (Semana Actual)":
    if not partidos_semana.empty:
        st.subheader(f"Partidos de la Semana {semana_actual}")
        opciones = [f"{row['away_team']} @ {row['home_team']}" for _, row in partidos_semana.iterrows()]
        partido_elegido = st.selectbox("Selecciona un partido:", opciones)
        
        # Extraer equipos del string "AWAY @ HOME"
        partes = partido_elegido.split(" @ ")
        equipo_visitante, equipo_local = partes[0], partes[1]
    else:
        st.info("No hay partidos pendientes en esta temporada. Usa la selección manual.")

if modo_seleccion == "🔍 Selección Manual" or (modo_seleccion == "📅 Próximos Partidos (Semana Actual)" and partidos_semana.empty):
    col1, col2 = st.columns(2)
    with col1:
        equipo_local = st.selectbox("Equipo Local:", lista_equipos, index=lista_equipos.index("KC") if "KC" in lista_equipos else 0)
    with col2:
        equipo_visitante = st.selectbox("Equipo Visitante:", lista_equipos, index=lista_equipos.index("BAL") if "BAL" in lista_equipos else 1)

# --- EJECUCIÓN DEL MODELO ---
if equipo_local and equipo_visitante:
    # Obtener historial H2H
    w_loc, w_vis, emp, total_juegos = calcular_h2h(historial_df, equipo_local, equipo_visitante)
    
    st.markdown("---")
    # Mostrar Contexto Histórico
    st.markdown(f"### 📖 Historial Directo desde 2015: {equipo_local} vs {equipo_visitante}")
    if total_juegos > 0:
        st.write(f"Han jugado **{total_juegos}** veces. **{equipo_local}** ganó {w_loc}, **{equipo_visitante}** ganó {w_vis} y hubo {emp} empates.")
    else:
        st.write("No hay registros de partidos entre estos dos equipos desde 2015.")

    if st.button("🚀 Generar Predicción", type="primary"):
        if equipo_local == equipo_visitante:
            st.warning("Selecciona dos equipos diferentes.")
        else:
            with st.spinner('Ejecutando 10,000 simulaciones...'):
                prob_loc, prob_vis, prob_emp, pts_loc, pts_vis, ou = predecir_partido_nfl(
                    nfl_teams, league_avg, equipo_local, equipo_visitante, hfa, w_loc, w_vis
                )
                
                st.success("Simulación Completada")
                
                # --- RESULTADOS ---
                c1, c2, c3 = st.columns(3)
                c1.metric(label=f"Prob. {equipo_local} (Local)", value=f"{prob_loc * 100:.1f}%")
                c2.metric(label="Probabilidad Empate", value=f"{prob_emp * 100:.1f}%")
                c3.metric(label=f"Prob. {equipo_visitante} (Visitante)", value=f"{prob_vis * 100:.1f}%")
                
                c4, c5 = st.columns(2)
                c4.metric(label="Over/Under Proyectado", value=f"{ou:.1f} pts")
                
                spread = pts_vis - pts_loc
                spread_text = f"{equipo_local} {spread:.1f}" if spread < 0 else f"{equipo_visitante} -{spread:.1f}"
                c5.metric(label="Marcador Esperado Exacto", value=f"{pts_loc:.1f} - {pts_vis:.1f}")