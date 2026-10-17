"""
Wealth and Health dashboard (1950 to 2022)

Charts are drawn with matplotlib; Streamlit adds the buttons,
sliders and drop-downs that make it interactive.

    streamlit run app.py

The app reads data/clean_joined.csv (made by joins_and_features.ipynb).
If that file is missing it builds the same table from
data/clean_life.csv, data/clean_gdp.csv and data/clean_mort.csv.
"""

from pathlib import Path  # builds file paths that work on Windows and Mac

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

# 1. Page setup

st.set_page_config(page_title="Does Money Buy a Longer Life?", page_icon="🌍", layout="wide")

HERE = Path(__file__).parent
DATA_DIRS = [HERE / "data", HERE.parent / "data"]

INCOME_ORDER = ["Low", "Lower middle", "Upper middle", "High"]
INCOME_COLORS = {
    "Low": "darkorange",
    "Lower middle": "gold",
    "Upper middle": "seagreen",
    "High": "steelblue",
}


# 2. Load the data (cached so it only runs once)

def find_file(name):
    for folder in DATA_DIRS:
        path = folder / name
        if path.exists():
            return path
    return None


@st.cache_data
def load_data():
    joined = find_file("clean_joined.csv")
    if joined is not None:
        df = pd.read_csv(joined)
    else:
        files = [find_file(n) for n in ("clean_life.csv", "clean_gdp.csv", "clean_mort.csv")]
        if None in files:
            return None
        life, gdp, mort = (pd.read_csv(f) for f in files)
        df = (
            life.merge(gdp, on=["Entity", "Code", "Year"])
            .merge(mort, on=["Entity", "Code", "Year"])
        )
        df = df[df["Year"].between(1950, 2022)]

    df = df[["Entity", "Code", "Year", "life_exp", "gdp_pc", "child_mort"]].dropna()

    # Income group within each year: split the countries of that year into
    # four equal-sized groups by GDP per capita (poorest 25% = "Low").
    df["income_group"] = df.groupby("Year")["gdp_pc"].transform(
        lambda s: pd.qcut(s, 4, labels=INCOME_ORDER)
    )
    df["income_group"] = pd.Categorical(df["income_group"], INCOME_ORDER, ordered=True)
    
    # Create a child mortality tier for Chi-Square analysis
    df["child_mortality_tier"] = pd.qcut(df["child_mort"], 3, labels=["Low", "Medium", "High"])
    
    return df


df = load_data()
if df is None:
    st.error(
        "Could not find the data. Put clean_joined.csv (or clean_life.csv, "
        "clean_gdp.csv and clean_mort.csv) in a folder called data/ next to app.py."
    )
    st.stop()

countries = sorted(df["Entity"].unique())
min_year, max_year = int(df["Year"].min()), int(df["Year"].max())



# 3. Sidebar filters

st.sidebar.header("Filters")

focus = st.sidebar.selectbox(
    "Focus country",
    countries,
    index=countries.index("Kenya") if "Kenya" in countries else 0,
)
year = st.sidebar.slider("Year", min_year, max_year, 2019)
groups = st.sidebar.multiselect("Income groups", INCOME_ORDER, default=INCOME_ORDER)
compare = st.sidebar.multiselect(
    "Countries to compare over time",
    countries,
    default=[c for c in [focus, "Uganda", "Tanzania", "Ghana"] if c in countries],
)

st.sidebar.caption(
    "Income groups are worked out separately for each year: the poorest "
    "quarter of countries that year is 'Low', the richest quarter is 'High'."
)
st.sidebar.caption("Data: Our World in Data (UN WPP, Maddison Project, UN IGME).")

year_df = df[(df["Year"] == year) & (df["income_group"].isin(groups))]
focus_df = df[df["Entity"] == focus].sort_values("Year")
focus_row = df[(df["Entity"] == focus) & (df["Year"] == year)]



# 4. Title and key numbers

st.title("🌍 Does Money Buy a Longer Life?")
st.subheader("Wealth and health in 163 countries, 1950 to 2022, with a closer look at Kenya")
st.markdown(
    "Do richer countries live longer and lose fewer children? "
    "Use the filters on the left to explore."
)

col1, col2, col3, col4 = st.columns(4)
if focus_row.empty:
    col1.metric(f"{focus} in {year}", "No data")
else:
    r = focus_row.iloc[0]
    first = focus_df.iloc[0]
    col1.metric(
        f"Life expectancy, {focus}",
        f"{r.life_exp:.1f} yrs",
        f"{r.life_exp - first.life_exp:+.1f} since {int(first.Year)}",
    )
    col2.metric(
        f"GDP per person, {focus}",
        f"${r.gdp_pc:,.0f}",
        f"{(r.gdp_pc / first.gdp_pc - 1) * 100:+.0f}% since {int(first.Year)}",
    )
    col3.metric(
        f"Child mortality, {focus}",
        f"{r.child_mort:.1f}%",
        f"{r.child_mort - first.child_mort:+.1f} pts since {int(first.Year)}",
        delta_color="inverse",  # going down is good
    )
if len(year_df) > 2:
    corr = np.corrcoef(np.log(year_df["gdp_pc"]), year_df["life_exp"])[0, 1]
    col4.metric(f"Wealth-health link in {year}", f"r = {corr:.2f}",
                help="Correlation between log GDP per person and life expectancy "
                     "for the countries shown. 1 = perfect link, 0 = no link.")



# 5. Tabs

tab1, tab2, tab3, tab4 = st.tabs(
    ["Wealth vs life expectancy", "Trends over time", f"{focus} vs expected", "Data"]
)

# Tab 1: scatter for one year 
with tab1:
    st.subheader(f"Richer countries live longer ({year})")
    if year_df.empty:
        st.info("No countries match these filters.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        for group in groups:
            part = year_df[year_df["income_group"] == group]
            ax.scatter(part["gdp_pc"], part["life_exp"], s=part["child_mort"] * 15,
                       color=INCOME_COLORS[group], alpha=0.6, label=group)

        x = np.log(year_df["gdp_pc"])
        slope, intercept = np.polyfit(x, year_df["life_exp"], 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(np.exp(xs), intercept + slope * xs, color="grey", linestyle="--", label="Trend")

        if not focus_row.empty:
            ax.scatter(focus_row["gdp_pc"], focus_row["life_exp"], marker="*", s=300,
                       color="black", label=focus)
            ax.annotate(focus, (focus_row["gdp_pc"].iloc[0], focus_row["life_exp"].iloc[0]),
                        xytext=(8, 8), textcoords="offset points", fontweight="bold")

        ax.set_xscale("log")
        ax.set_xlabel("GDP per person (international dollars, log scale)")
        ax.set_ylabel("Life expectancy (years)")
        ax.legend(title="Income group")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption("Bubble size = child mortality. Move the year slider to see how the picture changes.")

    st.subheader(f"Average life expectancy by income group ({year})")
    avg = year_df.groupby("income_group", observed=True)["life_exp"].mean()
    if not avg.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(avg.index.astype(str), avg.values,
                      color=[INCOME_COLORS[g] for g in avg.index])
        ax.bar_label(bars, fmt="%.1f")
        ax.set_xlabel("Income group")
        ax.set_ylabel("Average life expectancy (years)")
        st.pyplot(fig)
        st.caption("Income groups are worked out separately for each year.")

    # A/B Style Group Comparison (T-Test)
    st.subheader("A/B Style Group Comparison (T-Test)")
    col_a_sel, col_b_sel = st.columns(2)
    with col_a_sel:
        group_a = st.selectbox("Select Group A", INCOME_ORDER, index=0)
    with col_b_sel:
        group_b = st.selectbox("Select Group B", INCOME_ORDER, index=3)

    metric_labels = {
        "Life Expectancy": "life_exp",
        "GDP per Capita": "gdp_pc",
        "Child Mortality": "child_mort",
    }
    selected_label = st.selectbox("Metric to Compare", list(metric_labels.keys()))
    metric_choice = metric_labels[selected_label]

    data_a = df[df["income_group"] == group_a][metric_choice].dropna()
    data_b = df[df["income_group"] == group_b][metric_choice].dropna()

    t_stat, p_val = stats.ttest_ind(data_a, data_b, equal_var=False)

    col_res1, col_res2 = st.columns(2)
    col_res1.metric(label=f"Group A ({group_a}) Mean", value=f"{data_a.mean():.2f}")
    col_res2.metric(
        label=f"Group B ({group_b}) Mean",
        value=f"{data_b.mean():.2f}",
        delta=f"{data_b.mean() - data_a.mean():.2f}",
    )
    st.write(
        f"**T-Statistic:** {t_stat:.2f} | **P-Value:** {p_val:.4e} "
        f"({'Statistically Significant' if p_val < 0.05 else 'Not Significant'})"
    )


# Tab 2: trends over time 
with tab2:
    st.subheader("How countries changed over time")
    labels = {"life_exp": "Life expectancy (years)",
              "gdp_pc": "GDP per person (international dollars)",
              "child_mort": "Child mortality (%)"}
    measure = st.radio("Measure", list(labels), format_func=labels.get, horizontal=True)
    years = st.slider("Years to show", min_year, max_year, (min_year, max_year))
    in_range = df[df["Year"].between(*years)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for country in compare:
        c = in_range[in_range["Entity"] == country]
        ax.plot(c["Year"], c[measure], linewidth=2, label=country)
    world = in_range.groupby("Year")[measure].median()
    ax.plot(world.index, world.values, color="grey", linestyle=":", label="Median Country")
    ax.set_xlabel("Year")
    ax.set_ylabel(labels[measure])
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    if not compare:
        st.info("Pick countries in the sidebar to compare them.")


# Tab 3: is the focus country above or below what its income predicts?
with tab3:
    st.subheader(f"Does {focus} live as long as its income predicts?")
    st.markdown(
        "For each year we fit a line of life expectancy on log GDP across all "
        "countries, then read off what a country with this income *should* "
        "have. The gap is actual minus expected."
    )
    rows = []
    for y, g in df.groupby("Year"):
        s, i = np.polyfit(np.log(g["gdp_pc"]), g["life_exp"], 1)
        me = g[g["Entity"] == focus]
        if not me.empty:
            expected = i + s * np.log(me["gdp_pc"].iloc[0])
            rows.append({"Year": y, "Actual": me["life_exp"].iloc[0], "Expected": expected})
    gap = pd.DataFrame(rows)
    if gap.empty:
        st.info(f"No data for {focus}.")
    else:
        gap["Gap"] = gap["Actual"] - gap["Expected"]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(gap["Year"], gap["Actual"], color="steelblue", linewidth=2, label="Actual")
        ax.plot(gap["Year"], gap["Expected"], color="grey", linestyle="--", label="Expected from income")
        ax.set_xlabel("Year")
        ax.set_ylabel("Life expectancy (years)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        colors = ["seagreen" if v > 0 else "darkorange" for v in gap["Gap"]]
        ax.bar(gap["Year"], gap["Gap"], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Years above (+) / below (-)")
        st.pyplot(fig)

        latest = gap[gap["Year"] <= year].tail(1)
        if not latest.empty:
            v = latest["Gap"].iloc[0]
            st.markdown(
                f"In **{int(latest['Year'].iloc[0])}**, {focus} was "
                f"**{abs(v):.1f} years {'above' if v > 0 else 'below'}** "
                "what its income predicts."
            )

    # Statistical Tests Section
    st.markdown("---")
    st.subheader("Statistical Tests")

    # Chi-Square Test
    contingency_table = pd.crosstab(df["income_group"], df["child_mortality_tier"])
    chi2, p_chi, dof, ex = stats.chi2_contingency(contingency_table)
    st.write("**Chi-Square Test of Independence** (Income Tiers vs. Child Mortality Tiers):")
    st.write(f"- Chi2 Statistic: {chi2:.2f} | P-Value: {p_chi:.4e}")

    # ANOVA Test
    low_inc = df[df["income_group"] == "Low"]["life_exp"]
    lower_mid = df[df["income_group"] == "Lower middle"]["life_exp"]
    upper_mid = df[df["income_group"] == "Upper middle"]["life_exp"]
    high_inc = df[df["income_group"] == "High"]["life_exp"]

    f_stat, p_anova = stats.f_oneway(low_inc, lower_mid, upper_mid, high_inc)
    st.write("**One-Way ANOVA** (Life Expectancy across Income Groups):")
    st.write(f"- F-Statistic: {f_stat:.2f} | P-Value: {p_anova:.4e}")

#   Tab 4: the data itself 
with tab4:
    st.subheader(f"Data for {year}")
    show = year_df.sort_values("life_exp", ascending=False).rename(columns={
        "Entity": "Country", "life_exp": "Life expectancy", "gdp_pc": "GDP per person",
        "child_mort": "Child mortality (%)", "income_group": "Income group"})
    st.dataframe(show.drop(columns=["Code", "Year", "child_mortality_tier"], errors="ignore"), use_container_width=True, hide_index=True)
    st.download_button("Download this table (CSV)", show.to_csv(index=False),
                       file_name=f"wealth_health_{year}.csv", mime="text/csv")