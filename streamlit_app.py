"""
streamlit_app.py  —  DML ETF Picks Dashboard
"""

import streamlit as st
import pandas as pd
import requests
import json
import glob
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="P2-DML-ETF-PICKS",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .ticker-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
    }
    .confidence-high { color: #27ae60; font-weight: 600; }
    .confidence-medium { color: #f39c12; font-weight: 600; }
    .confidence-low { color: #e74c3c; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)


def load_data():
    """Load latest results from HuggingFace or local."""
    # Try HuggingFace
    try:
        repo_id = "P2SAMAPA/p2-dml-etf-payoffs-results"
        files_url = f"https://huggingface.co/api/datasets/{repo_id}/refs/main"
        response = requests.get(files_url, timeout=10)
        if response.status_code == 200:
            files = response.json()
            json_files = [f for f in files if f.endswith('.json') and f.startswith('dml_results_')]
            if json_files:
                latest = sorted(json_files)[-1]
                data_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{latest}"
                data_response = requests.get(data_url, timeout=10)
                if data_response.status_code == 200:
                    return data_response.json()
    except:
        pass
    
    # Try local
    try:
        json_files = glob.glob("dml_results_*.json")
        if json_files:
            latest = sorted(json_files)[-1]
            with open(latest, 'r') as f:
                return json.load(f)
    except:
        pass
    
    return None


def create_chart(picks):
    """Create bar chart of expected returns."""
    if not picks:
        return None
    
    df = pd.DataFrame(picks)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['expected_return'],
        y=df['ticker'],
        orientation='h',
        text=df['expected_return'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        marker_color=['#27ae60' if r > 0.5 else '#f39c12' if r > 0 else '#e74c3c' 
                      for r in df['expected_return']]
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
        xaxis_title="Expected Return (%)"
    )
    
    return fig


def main():
    st.markdown('<div class="main-header">📈 P2-DML-ETF-PICKS</div>', unsafe_allow_html=True)
    st.markdown("*Differential Machine Learning: Top ETF Picks for Next Trading Day*")
    
    data = load_data()
    
    if not data:
        st.error("⚠️ No data available. Please run the trainer first.")
        st.info("Run `python trainer.py` to generate results.")
        if st.button("🔄 Retry"):
            st.rerun()
        return
    
    run_date = data.get('run_date', 'Unknown')
    st.caption(f"📊 Results from: **{run_date}**")
    
    # Get top picks
    top_picks = data.get('top_picks', {})
    
    if not top_picks:
        st.warning("No picks generated. Please run trainer with proper configuration.")
        return
    
    # Display picks by universe
    for universe, picks in top_picks.items():
        st.markdown(f"## {universe}")
        
        if not picks:
            st.warning(f"No picks for {universe}")
            continue
        
        # Cards
        cols = st.columns(min(len(picks), 3))
        for idx, pick in enumerate(picks):
            col = cols[idx % len(cols)]
            with col:
                conf_class = f"confidence-{pick['confidence'].lower()}"
                st.markdown(f"""
                <div class="ticker-card">
                    <h3 style="margin:0;">{pick['ticker']}</h3>
                    <div style="font-size:2rem; font-weight:700; margin:0.5rem 0;">
                        {pick['expected_return']:.1f}%
                    </div>
                    <div class="{conf_class}">Confidence: {pick['confidence']}</div>
                    <div style="font-size:0.8rem; color:#666; margin-top:0.5rem;">
                        {pick.get('rationale', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Chart
        fig = create_chart(picks)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
    
    # Cross-universe summary
    st.markdown("## 🌟 Cross-Universe Top Picks")
    all_picks = []
    for universe, picks in top_picks.items():
        for pick in picks:
            pick['universe'] = universe
            all_picks.append(pick)
    
    if all_picks:
        df_all = pd.DataFrame(all_picks)
        df_all = df_all.sort_values('expected_return', ascending=False).head(10)
        st.dataframe(
            df_all[['ticker', 'universe', 'expected_return', 'confidence']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'ticker': 'Ticker',
                'universe': 'Universe',
                'expected_return': 'Expected Return %',
                'confidence': 'Confidence'
            }
        )
    
    st.caption(f"Data as of {run_date} | Powered by Differential Machine Learning")


if __name__ == "__main__":
    main()
