"""FIFA World Cup 2026 Simulator and Predictor - Streamlit App."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.simulation import TournamentSimulator, MatchPredictor
from src.feature_engineering import get_last_n_matches, get_h2h_last_n

st.set_page_config(
    page_title="FIFA WC 2026 Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    :root {
        --primary: #1A365D;
        --secondary: #2B6CB0;
        --accent: #D69E2E;
        --bg: #F7FAFC;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: fadeInDown 0.8s ease-out;
    }
    .sub-header {
        color: #4A5568;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        border-top: 4px solid var(--accent);
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }
    .prob-bar-container {
        display: flex;
        height: 30px;
        border-radius: 15px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .prob-bar-segment {
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        transition: width 1s ease-in-out;
    }
    .home-segment { background: #38A169; }
    .draw-segment { background: #718096; }
    .away-segment { background: #E53E3E; }
    
    .team-badge {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        background: #EDF2F7;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    import traceback

    print("Creating MatchPredictor...")

    try:
        p = MatchPredictor()
        print("MatchPredictor created successfully")
        return p

    except Exception as e:
        print("ERROR TYPE:", type(e))
        print("ERROR:", repr(e))
        traceback.print_exc()
        raise


@st.cache_resource
def get_simulator():
    import traceback

    print("Creating TournamentSimulator...")

    try:
        sim = TournamentSimulator()
        print("TournamentSimulator created successfully")
        return sim

    except Exception as e:
        print("ERROR TYPE:", type(e))
        print("ERROR:", repr(e))
        traceback.print_exc()
        raise



def prob_bar_html(home_prob, draw_prob, away_prob, home_name, away_name):
    h_pct = int(home_prob * 100)
    d_pct = int(draw_prob * 100)
    a_pct = 100 - h_pct - d_pct
    
    return f"""
    <div class="prob-bar-container">
        <div class="prob-bar-segment home-segment" style="width: {h_pct}%" title="{home_name} Win">
            {h_pct}%
        </div>
        <div class="prob-bar-segment draw-segment" style="width: {d_pct}%" title="Draw">
            {d_pct}%
        </div>
        <div class="prob-bar-segment away-segment" style="width: {a_pct}%" title="{away_name} Win">
            {a_pct}%
        </div>
    </div>
    """

def create_radar_chart(team1_prof, team2_prof):
    categories = ['Elo Rating', 'FIFA Rank (inv)', 'Form 2y pts', 'Form 2y gd', 'Confed Strength']
    
    def normalize_prof(prof):
        # Normalize to 0-100 scale roughly
        elo = min(100, max(0, (prof['elo'] - 1300) / 7))
        rank = min(100, max(0, (100 - prof['fifa_rank'])))
        pts = min(100, max(0, prof['pts_2y'] * 33.3))
        gd = min(100, max(0, (prof['gd_2y'] + 1) * 25))
        conf = prof['conf_strength'] * 100
        return [elo, rank, pts, gd, conf]
        
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalize_prof(team1_prof),
        theta=categories,
        fill='toself',
        name=team1_prof['team'],
        line_color='#38A169'
    ))
    fig.add_trace(go.Scatterpolar(
        r=normalize_prof(team2_prof),
        theta=categories,
        fill='toself',
        name=team2_prof['team'],
        line_color='#E53E3E'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=40, r=40, t=20, b=20),
        height=400
    )
    return fig

def render_match_predictor():
    predictor = get_predictor()
    teams = predictor.tdata['all_teams']
    
    st.markdown('<div class="main-header">⚽ Match Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict any international match using advanced ML models.</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            home = st.selectbox("Home Team", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
        with col2:
            st.markdown("<h3 style='text-align: center; margin-top: 1.5rem;'>VS</h3>", unsafe_allow_html=True)
        with col3:
            away_opts = [t for t in teams if t != home]
            away = st.selectbox("Away Team", away_opts, index=away_opts.index("Brazil") if "Brazil" in away_opts else 0)
            
    neutral = st.checkbox("Neutral Venue (Tournament Style)", value=True)
    
    if st.button("Predict Match", type="primary", use_container_width=True):
        with st.spinner("Analyzing match..."):
            pred = predictor.predict(home, away, neutral=neutral)
            sw = predictor.strengths_weaknesses(home, away, pred)
            
            st.markdown("---")
            
            # Top Probability Bar
            st.markdown("### Match Outcome Probabilities")
            st.markdown(prob_bar_html(pred['home_win_prob'], pred['draw_prob'], pred['away_win_prob'], home, away), unsafe_allow_html=True)
            
            # Team Badges and Base Stats
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="team-badge" style="border-bottom: 4px solid #38A169;">{home}<br><span style="font-size: 0.9rem; font-weight: normal; color: #4A5568;">Win Prob: {pred["home_win_prob"]*100:.1f}%</span></div>', unsafe_allow_html=True)
                st.metric("Elo Rating", f"{pred['home_elo']:.0f}")
                st.metric("FIFA Rank", f"{int(pred['home_rank'])}")
            with c2:
                st.markdown(f'<div class="team-badge" style="border-bottom: 4px solid #E53E3E;">{away}<br><span style="font-size: 0.9rem; font-weight: normal; color: #4A5568;">Win Prob: {pred["away_win_prob"]*100:.1f}%</span></div>', unsafe_allow_html=True)
                st.metric("Elo Rating", f"{pred['away_elo']:.0f}")
                st.metric("FIFA Rank", f"{int(pred['away_rank'])}")
                
            st.markdown("---")
            
            # Tabs for deep dive
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Head-to-Head Profile", "💪 Strengths & Weaknesses", "📜 Recent Form", "🔄 H2H History"])
            
            with tab1:
                st.plotly_chart(create_radar_chart(sw[home]['profile'], sw[away]['profile']), use_container_width=True)
                
            with tab2:
                s1, s2 = st.columns(2)
                with s1:
                    st.markdown(f"#### {home} Analysis")
                    for s in sw[home]['strengths']: st.success(f"✅ {s}")
                    for w in sw[home]['weaknesses']: st.warning(f"⚠️ {w}")
                with s2:
                    st.markdown(f"#### {away} Analysis")
                    for s in sw[away]['strengths']: st.success(f"✅ {s}")
                    for w in sw[away]['weaknesses']: st.warning(f"⚠️ {w}")
                    
            with tab3:
                f1, f2 = st.columns(2)
                with f1:
                    st.markdown(f"#### {home} Last 5")
                    st.dataframe(get_last_n_matches(home, 5), use_container_width=True, hide_index=True)
                with f2:
                    st.markdown(f"#### {away} Last 5")
                    st.dataframe(get_last_n_matches(away, 5), use_container_width=True, hide_index=True)
                    
            with tab4:
                st.markdown("#### Last Meetings")
                h2h_df = get_h2h_last_n(home, away, 10)
                if len(h2h_df) > 0:
                    st.dataframe(h2h_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No recent head-to-head matches found.")

def render_tournament():
    simulator = get_simulator()
    
    st.markdown('<div class="main-header">🏆 World Cup 2026 Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate the entire tournament path from group stages to the final.</div>', unsafe_allow_html=True)
    
    if st.button("Simulate Full Tournament", type="primary", use_container_width=True):
        with st.spinner("Simulating Group Stages and Knockouts..."):
            res = simulator.simulate_tournament()
            
            st.success(f"🎉 Tournament Winner: **{res['summary']['winner']}**")
            
            tab_ko, tab_groups = st.tabs(["🥊 Knockout Stages", "📋 Group Stages"])
            
            with tab_ko:
                st.markdown("### Knockout Bracket Path")
                ko = res['knockout']
                
                stages = ['R32', 'R16', 'QF', 'SF', 'Final']
                for stage in reversed(stages):
                    st.markdown(f"#### {stage}")
                    stage_matches = ko[ko['round'] == stage]
                    for _, m in stage_matches.iterrows():
                        res_str = f"**{m['winner']}** won {m['home_goals']}-{m['away_goals']} against {m['loser']}"
                        if m.get('resolved_via') == 'penalties':
                            res_str += " (on pens)"
                        st.info(f"{m['home_team']} vs {m['away_team']} ➔ {res_str}")
                    st.markdown("---")
                        
            with tab_groups:
                groups = sorted(res['group_tables']['group'].unique())
                cols = st.columns(3)
                for i, g in enumerate(groups):
                    with cols[i % 3]:
                        st.markdown(f"#### Group {g}")
                        gt = res['group_tables'][res['group_tables']['group'] == g]
                        st.dataframe(gt[['pos', 'team', 'points', 'gd', 'gf']], use_container_width=True, hide_index=True)

def render_monte_carlo():
    simulator = get_simulator()
    
    st.markdown('<div class="main-header">🎲 Monte Carlo Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Run thousands of simulations to find true tournament probabilities.</div>', unsafe_allow_html=True)
    
    n_sims = st.slider("Simulations", 50, 1000, 100, 50)
    if st.button("Run Monte Carlo", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        
        def cb(curr, total):
            progress.progress(curr / total)
            status.text(f"Simulation {curr}/{total}")
            
        res = simulator.run_monte_carlo(n=n_sims, callback=cb)
        
        status.success(f"Completed {n_sims} simulations!")
        
        # Format probabilities
        for c in ['r32_prob', 'r16_prob', 'qf_prob', 'sf_prob', 'final_prob', 'winner_prob']:
            res[c] = (res[c] * 100).round(1)
            
        st.markdown("### Top Favorites")
        
        # Winner probabilities chart
        fig = px.bar(res.head(15), x='team', y='winner_prob', color='winner_prob',
                     color_continuous_scale='Viridis', title="Win Probability (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Full table
        st.markdown("### Stage Reach Probabilities (%)")
        st.dataframe(res[['team', 'confederation', 'r32_prob', 'r16_prob', 'qf_prob', 'sf_prob', 'final_prob', 'winner_prob']].head(30), 
                     use_container_width=True, hide_index=True)

def main():
    st.sidebar.markdown("## 🏆 FIFA WC 2026")
    page = st.sidebar.radio("Navigation", ["Match Predictor", "Tournament Simulation", "Monte Carlo"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Features:**")
    st.sidebar.markdown("- 5 ML Models Ensemble")
    st.sidebar.markdown("- Weighted 2y Form")
    st.sidebar.markdown("- Confederation Calibration")
    st.sidebar.markdown("- 48-Team Format")
    
    if page == "Match Predictor":
        render_match_predictor()
    elif page == "Tournament Simulation":
        render_tournament()
    elif page == "Monte Carlo":
        render_monte_carlo()

if __name__ == "__main__":
    main()
