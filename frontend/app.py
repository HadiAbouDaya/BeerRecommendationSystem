import streamlit as st
import requests
import pandas as pd
import json

# Configure page
st.set_page_config(
    page_title="Beer Recommendation System",
    page_icon="🍺",
    layout="wide"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #FF6B35;
    font-size: 3rem;
    margin-bottom: 2rem;
}
/* beer cards adapted for dark theme */
.beer-card {
    border: 2px solid #FF6B35;      /* accent border */
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #1E1E2E;      /* dark card bg */
    color: #E0E0E0;                 /* light text */
}
.beer-card h4 {
    color: #FFD369;                 /* gold accent for titles */
    margin-bottom: 0.5rem;
}
.beer-card p {
    margin: 0.2rem 0;
}
</style>
""", unsafe_allow_html=True)

def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_beer_styles():
    try:
        r = requests.get(f"{API_BASE_URL}/beer-styles")
        return r.json().get("beer_styles", []) if r.status_code == 200 else []
    except:
        return []

def get_breweries():
    try:
        r = requests.get(f"{API_BASE_URL}/breweries")
        return r.json().get("breweries", [])[:100] if r.status_code == 200 else []
    except:
        return []

def get_recommendations(preferences, top_n=5):
    try:
        r = requests.post(
            f"{API_BASE_URL}/recommend",
            json=preferences,
            params={"top_n": top_n}
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"Error: {r.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
    return []

def main():
    st.markdown('<h1 class="main-header">🍺 Beer Recommendation System</h1>', unsafe_allow_html=True)

    if not check_api_health():
        st.error("❌ Cannot connect to the recommendation API. Please make sure the backend is running.")
        return
    st.success("✅ Connected to recommendation API")

    with st.sidebar:
        st.header("🎯 Your Preferences")
        username = st.text_input("Username (optional)", value="Beer Enthusiast")
        location = st.text_input("Location (optional)", value="Montreal, QC")

        st.subheader("🍻 Preferred Beer Styles")
        styles = get_beer_styles()
        if styles:
            selected_styles = st.multiselect("Select beer styles you enjoy:", styles,
                                             default=["American IPA", "American Double / Imperial IPA"])
        else:
            st.error("Could not load beer styles")
            return

        st.subheader("🌡️ Alcohol Content")
        abv_range = st.slider("ABV Range (%)", 0.0, 20.0, (4.0, 8.0), 0.5)

        st.subheader("🏭 Preferred Breweries (optional)")
        breweries = get_breweries()
        selected_breweries = st.multiselect("Select specific breweries:", breweries)

        top_n = st.selectbox("Number of recommendations:", [3, 5, 10, 15], index=1)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📋 Your Selections")
        st.write(f"**Username:** {username}")
        st.write(f"**Location:** {location}")
        st.write(f"**Beer Styles:** {', '.join(selected_styles)}")
        st.write(f"**ABV Range:** {abv_range[0]}% - {abv_range[1]}%")
        if selected_breweries:
            st.write(f"**Breweries:** {', '.join(selected_breweries)}")

        if st.button("🔍 Get Recommendations"):
            if not selected_styles:
                st.error("Please select at least one beer style!")
            else:
                prefs = {
                    "username": username,
                    "beer_styles": selected_styles,
                    "abv_range": {"min": abv_range[0], "max": abv_range[1]},
                    "breweries": selected_breweries,
                    "location": location
                }
                with st.spinner("Finding the perfect beers for you..."):
                    st.session_state.recommendations = get_recommendations(prefs, top_n)
                    st.session_state.preferences = prefs

        with col2:
            st.subheader("🏆 Your Beer Recommendations")
            recs = st.session_state.get("recommendations", [])
            if recs:
                for beer in recs:
                    st.markdown(f"""
                        <div class="beer-card">
                            <h4>#{beer['rank']} 🍺 {beer['beer_name']}</h4>
                            <p><strong>Style:</strong> {beer['beer_style']}</p>
                            <p><strong>Brewery:</strong> {beer['brewery_name']}</p>
                            <p><strong>ABV:</strong> {beer['abv']}%</p>
                            <p><strong>Community Average:</strong> ⭐ {beer['avg_rating']}/5.0</p>
                            <p><em>{beer['reason']}</em></p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                # If user has already searched but got no recs, show something fun
                if "preferences" in st.session_state:
                    st.warning("😢 Oh no—your perfect brew is a mythical unicorn! 🦄\nTry broadening your style or ABV range.")
                    st.balloons()
                else:
                    st.info("👈 Use the sidebar to set your preferences and get personalized beer recommendations!")

    st.markdown("---")
    st.markdown("We automated Charlotte’s Beer Recommendation model ❤️ using embedding-based representations and neural ranking with TensorFlow Recommenders.")

if __name__ == "__main__":
    main()
