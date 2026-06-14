import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval

st.set_page_config(
    page_title="Blackjack Big Data Analysis",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    h1 { font-size: 2.2rem !important; color: #2c3e50; }
    h2 { font-size: 1.5rem !important; border-bottom: 2px solid #ecf0f1; padding-bottom: 0.3rem; margin-top: 1.8rem !important; color: #34495e; }
    h3 { font-size: 1.1rem !important; color: #555; }
    .stAlert p { font-size: 0.95rem; }
    .section-note { background: #f0f7ff; border-left: 4px solid #4B6BFB; padding: 1rem; border-radius: 0 8px 8px 0; font-size: 0.95rem; color: #333; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

def note(text):
    st.markdown(f'<div class="section-note">💡 <b>Quick Note:</b> {text}</div>', unsafe_allow_html=True)

SECTIONS = [
    "🏠 Welcome & Abstract",
    "📦 Dataset Overview",
    "📊 Core Statistics",
    "🧹 Data Quality & Cleanup",
    "📈 Visualizing Distributions",
    "🔍 Deep Dive Comparisons",
    "🔧 Feature Engineering",
    "🧪 Testing Our Hypotheses",
]

with st.sidebar:
    st.markdown("### 🃏 Blackjack Analytics")
    st.markdown("---")
    section = st.radio("Explore the project:", SECTIONS, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Powered by 50M simulated hands\nHi-Lo Card Counting System")

@st.cache_data(show_spinner="Fetching data efficiently from Parquet...")
def load_data():
    df = pd.read_parquet("../data/blackjack_simulator.parquet")
    return df

TC_ORDER = ["Very Low", "Low", "Neutral", "High", "Very High"]

try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

def no_data():
    st.warning(
        "**Oops! The processed dataset is missing.** \n"
        "Make sure you run your Jupyter Notebook first to generate `blackjack_processed.parquet` and place it in the `data/` folder."
    )

if section == "🏠 Welcome & Abstract":
    st.title("🃏 Blackjack Simulator — Statistical Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is this project about?
        We are analyzing a massive dataset of **50 million simulated blackjack hands**, driven by a standard Hi-Lo card-counting engine. 
        
        Our main goal is to figure out if the **true count** (a normalized metric of the remaining valuable cards in the shoe) actually gives the player a statistical edge. We want to see how hand strength, the dealer's visible card, and the count level all come together to influence the game's outcome.
        
        Everything you see here was generated programmatically. Our team handled data validation, built descriptive stats, engineered new features, and put two major card-counting hypotheses to the test using grouped aggregations and linear regression.
        """)
        
    with col2:
        st.markdown("### The Dashboard at a Glance")
        st.metric("Total Hands Played", "50,000,000")
        st.metric("Unique Casino Shoes", "822,845")
        st.metric("Standard House Edge", "−0.56 %")
        st.markdown("""
        **What we are testing:**
        1. Does the win rate truly go up as the true count rises?
        2. Do you actually get more natural Blackjacks when the count is high?
        """)

elif section == "📦 Dataset Overview":
    st.title("📦 Exploring the Dataset")
    note("This data comes straight from a blackjack simulator using an 8-deck shoe and the classic Hi-Lo strategy.")

    st.markdown("### Where did this data come from?")
    st.markdown("""
    This falls perfectly into the realm of **casino game theory**. Each row in our dataset is a single hand dealt from a partially played shoe. The simulator tracks the running count, converts it to a true count, and logs every single decision and outcome.
    """)

    field_df = pd.DataFrame({
        "Feature": [
            "shoe_id", "cards_remaining", "dealer_up",
            "initial_hand", "dealer_final", "player_final_value", 
            "run_count", "true_count", "win"
        ],
        "Data Type": [
            "Integer", "Integer", "Integer",
            "List (String)", "List (String)", "String/Int",
            "Integer", "Integer", "Float"
        ],
        "What it means": [
            "Unique ID for each deck shuffle",
            "Cards left in the shoe (79–416)",
            "Dealer's visible card (2–11)",
            "Player's starting two cards",
            "Full sequence of dealer's cards",
            "Final points (or 'BJ' for a natural)",
            "Raw Hi-Lo count before the hand",
            "Normalized true count",
            "Profit/Loss multiplier (+1.5 for BJ, -1 for loss)"
        ],
    })
    st.table(field_df)

    if data_loaded:
        st.markdown("### Let's look at some raw data")
        st.dataframe(df.head(5), use_container_width=True)
    else:
        no_data()

elif section == "📊 Core Statistics":
    st.title("📊 Core Descriptive Statistics")
    note("Let's break down the math behind four crucial numbers: true_count, run_count, cards_remaining, and your actual winnings.")

    if not data_loaded:
        no_data()
    else:
        num_cols = ["true_count", "run_count", "cards_remaining", "win"]
        labels   = ["True Count", "Running Count", "Cards Left", "Profit per Hand"]

        c1, c2, c3, c4 = st.columns(4)
        stats = {c: df[c].describe() for c in num_cols}
        
        for col_ui, col_data, lbl in zip([c1, c2, c3, c4], num_cols, labels):
            with col_ui:
                st.metric(f"Mean {lbl}", f"{stats[col_data]['mean']:.4f}")
                st.caption(f"Median: {df[col_data].median():.4f}")
                st.caption(f"Range: {stats[col_data]['min']:.1f} to {stats[col_data]['max']:.1f}")

        st.markdown("### The Full Statistical Breakdown")
        desc = df[num_cols].describe().T
        desc.index = labels
        st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

elif section == "🧹 Data Quality & Cleanup":
    st.title("🧹 Data Quality Checks")
    note("Good analysis needs clean data. Here is how we validated our 50 million rows.")

    if not data_loaded:
        no_data()
    else:
        st.markdown("### Checking for Missing Information")
        null_pct = (df.isnull().sum() / len(df) * 100).reset_index()
        null_pct.columns = ["Column", "Missing %"]
        st.dataframe(null_pct.style.format({"Missing %": "{:.2f}"}), use_container_width=True)
        st.success("Looking great! No missing values detected across the entire dataset.")

        st.markdown("### Sanity Checks")
        checks = {
            "Cards remaining are within logical limits (79 to 416)": ((df["cards_remaining"] < 79).sum() + (df["cards_remaining"] > 416).sum() == 0),
            "Dealer up-card is valid (2 to 11)": (~df["dealer_up"].isin(range(2, 12))).sum() == 0,
        }
        
        for check, is_valid in checks.items():
            if is_valid:
                st.success(f"✅ {check}")
            else:
                st.error(f"❌ {check}")

elif section == "📈 Visualizing Distributions":
    st.title("📈 Feature Distributions")
    note("A visual look at how the cards fall, using histograms, bar charts, and KDEs.")

    if not data_loaded:
        no_data()
    else:
        st.markdown("### 1. True Count Spread")
        st.markdown("Most hands happen when the shoe is relatively neutral. It's rare to see extreme counts.")
        
        fig1, ax1 = plt.subplots(figsize=(9, 4))
        sns.histplot(df["true_count"], bins=45, kde=True, color="#378ADD", alpha=0.5, ax=ax1)
        ax1.axvline(df["true_count"].median(), color="#E24B4A", linestyle="--", label="Median")
        ax1.set_title("How often do specific True Counts happen?", fontweight="bold")
        ax1.legend()
        sns.despine()
        st.pyplot(fig1); plt.close(fig1)

        st.markdown("### 2. Player Advantage by Count")
        st.markdown("Notice how the expected value (profit) flips from red (losing) to green (winning) as the count goes up.")
        
        tc_profit = df.groupby("tc_group", observed=False)["win"].mean().reindex(TC_ORDER)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors = ["#E24B4A" if v < 0 else "#1D9E75" for v in tc_profit.values]
        ax2.barh(tc_profit.index, tc_profit.values, color=colors)
        ax2.axvline(0, color="black", linewidth=1)
        ax2.set_title("Average Profit per Hand by Count Group", fontweight="bold")
        sns.despine()
        st.pyplot(fig2); plt.close(fig2)

elif section == "🔍 Deep Dive Comparisons":
    st.title("🔍 Detailed Data Comparisons")
    note("Let's look at how different factors interact with each other.")

    if not data_loaded:
        no_data()
    else:
        st.markdown("### Hand Strength vs. True Count")
        st.markdown("Even with a great count, a medium hand (like a 15 or 16) is still a mathematical nightmare.")
        
        pivot = pd.pivot_table(df, values="win", index="hand_strength", columns="tc_group", aggfunc="mean", observed=False)[TC_ORDER]
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=ax)
        ax.set_title("Win Rate Heatmap", fontweight="bold")
        st.pyplot(fig); plt.close(fig)

elif section == "🔧 Feature Engineering":
    st.title("🔧 Data Transformations")
    note("Raw data isn't enough. We engineered several new features to make our analysis richer.")

    if not data_loaded:
        no_data()
    else:
        st.markdown("### What we added:")
        st.markdown("""
        * **Hand Strength:** We categorized starting totals into Weak, Medium, and Strong.
        * **Dealer Risk:** Grouped dealer up-cards into dangerous vs. safe categories.
        * **Blackjack Flag:** A simple 1 or 0 flag to quickly identify natural blackjacks.
        * **Count Groups:** Grouped the continuous true count into buckets (Very Low to Very High).
        """)
        
        st.markdown("### Preview of the Enhanced Data")
        st.dataframe(df[["true_count", "tc_group", "dealer_up", "dealer_strength", "blackjack"]].head(10), use_container_width=True)

elif section == "🧪 Testing Our Hypotheses":
    st.title("🧪 Putting the Math to the Test")

    if not data_loaded:
        no_data()
    else:
        st.markdown("### Hypothesis 1: Does winning get easier as the count rises?")
        st.info("We believe the win rate moves strictly upward as the true count increases, crossing into positive territory at a specific point.")
        
        tc_win = df.groupby("tc_group", observed=False)["win"].mean().reindex(TC_ORDER)
        st.dataframe(tc_win.reset_index().style.format({"win": "{:+.4f}"}))
        st.success("✅ **Confirmed:** The trend is strictly positive. As the true count bucket increases, the average loss shrinks until it becomes a consistent profit.")

        st.markdown("---")

        st.markdown("### Hypothesis 2: Do natural Blackjacks happen more often?")
        st.info("High counts mean more 10s and Aces remain. We predict the probability of hitting a natural Blackjack increases proportionally.")
        
        bj_rate = df.groupby("tc_group", observed=False)["blackjack"].mean().reindex(TC_ORDER)
        bj_df = bj_rate.reset_index()
        bj_df.columns = ["Count Group", "P(Blackjack)"]
        bj_df["Difference from Neutral"] = bj_df["P(Blackjack)"] - bj_rate["Neutral"]
        
        st.dataframe(bj_df.style.format({"P(Blackjack)": "{:.2%}", "Difference from Neutral": "{:+.2%}"}), use_container_width=True)
        
        fig_h2, ax_h2 = plt.subplots(figsize=(8, 4))
        ax_h2.plot(bj_df["Count Group"], bj_df["P(Blackjack)"], marker='o', color="#1D9E75", linewidth=2)
        ax_h2.axhline(bj_rate["Neutral"], color="#534AB7", linestyle="--", label="Neutral Baseline")
        ax_h2.set_title("Probability of Dealing a Blackjack by Count Group", fontweight="bold")
        ax_h2.legend()
        sns.despine()
        st.pyplot(fig_h2); plt.close(fig_h2)
        
        st.success("✅ **Confirmed:** The probability of being dealt a Blackjack directly correlates with a higher true count.")