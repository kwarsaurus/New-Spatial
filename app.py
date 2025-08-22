import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Restaurant Location Analyzer",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .info-card {
        background: #d1ecf1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class RestaurantLocationPredictor:
    def __init__(self, model_dir="models"):
        """Initialize the production system"""
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.category_mapping = {}
        self.business_center = (-6.2088, 106.8217)  # Jakarta business center
        
        self.load_models()
        self.prepare_district_data()
    
    @st.cache_resource
    def load_models(_self):
        """Load all trained models and components"""
        try:
            # Load feature names
            _self.feature_names = joblib.load(f"{_self.model_dir}/feature_names.pkl")
            
            # Load category mapping
            _self.category_mapping = joblib.load(f"{_self.model_dir}/category_mapping.pkl")
            
            # Load models and scalers for each category
            for category, safe_name in _self.category_mapping.items():
                model_path = f"{_self.model_dir}/model_{safe_name}.pkl"
                scaler_path = f"{_self.model_dir}/scaler_{safe_name}.pkl"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    _self.models[category] = joblib.load(model_path)
                    _self.scalers[category] = joblib.load(scaler_path)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def prepare_district_data(self):
        """Prepare district reference data for faster lookup"""
        # Jakarta Selatan districts with approximate boundaries
        self.district_data = {
            'Kebayoran Baru': {
                'center': (-6.2297, 106.7975),
                'bounds': {
                    'lat_min': -6.2500, 'lat_max': -6.2100,
                    'lng_min': 106.7800, 'lng_max': 106.8150
                }
            },
            'Tebet': {
                'center': (-6.2267, 106.8578),
                'bounds': {
                    'lat_min': -6.2450, 'lat_max': -6.2100,
                    'lng_min': 106.8400, 'lng_max': 106.8750
                }
            },
            'Setiabudi': {
                'center': (-6.2088, 106.8217),
                'bounds': {
                    'lat_min': -6.2250, 'lat_max': -6.1950,
                    'lng_min': 106.8050, 'lng_max': 106.8380
                }
            },
            'Mampang Prapatan': {
                'center': (-6.2500, 106.8200),
                'bounds': {
                    'lat_min': -6.2700, 'lat_max': -6.2300,
                    'lng_min': 106.8050, 'lng_max': 106.8350
                }
            },
            'Pancoran': {
                'center': (-6.2400, 106.8450),
                'bounds': {
                    'lat_min': -6.2600, 'lat_max': -6.2200,
                    'lng_min': 106.8300, 'lng_max': 106.8600
                }
            },
            'Jagakarsa': {
                'center': (-6.3200, 106.8200),
                'bounds': {
                    'lat_min': -6.3500, 'lat_max': -6.2900,
                    'lng_min': 106.8000, 'lng_max': 106.8400
                }
            },
            'Pasar Minggu': {
                'center': (-6.2900, 106.8450),
                'bounds': {
                    'lat_min': -6.3200, 'lat_max': -6.2600,
                    'lng_min': 106.8250, 'lng_max': 106.8650
                }
            },
            'Cilandak': {
                'center': (-6.3100, 106.8000),
                'bounds': {
                    'lat_min': -6.3400, 'lat_max': -6.2800,
                    'lng_min': 106.7800, 'lng_max': 106.8200
                }
            },
            'Pesanggrahan': {
                'center': (-6.2600, 106.7650),
                'bounds': {
                    'lat_min': -6.2900, 'lat_max': -6.2300,
                    'lng_min': 106.7450, 'lng_max': 106.7850
                }
            },
            'Kebayoran Lama': {
                'center': (-6.2400, 106.7750),
                'bounds': {
                    'lat_min': -6.2700, 'lat_max': -6.2100,
                    'lng_min': 106.7550, 'lng_max': 106.7950
                }
            }
        }
    
    def get_available_categories(self):
        """Get list of available categories"""
        return list(self.models.keys())
    
    def get_available_districts(self):
        """Get list of available districts"""
        return list(self.district_data.keys())
    
    def generate_smart_grid(self, district_name, grid_density='medium'):
        """Generate optimized grid for specific district"""
        if district_name not in self.district_data:
            raise ValueError(f"District '{district_name}' not found")
        
        district = self.district_data[district_name]
        bounds = district['bounds']
        
        # Grid density settings
        density_settings = {
            'low': 0.008,     # ~50 points
            'medium': 0.005,  # ~100 points  
            'high': 0.003,    # ~200 points
        }
        
        grid_size = density_settings.get(grid_density, 0.005)
        
        # Generate grid points within district bounds
        lats = np.arange(bounds['lat_min'], bounds['lat_max'], grid_size)
        lngs = np.arange(bounds['lng_min'], bounds['lng_max'], grid_size)
        
        grid_points = []
        for lat in lats:
            for lng in lngs:
                # Filter points that are reasonably within district
                distance_to_center = geodesic((lat, lng), district['center']).kilometers
                if distance_to_center <= 3:  # Within 3km of district center
                    grid_points.append({
                        'latitude': lat,
                        'longitude': lng,
                        'districtName': district_name,
                        'distance_to_center': geodesic((lat, lng), self.business_center).kilometers
                    })
        
        return pd.DataFrame(grid_points)
    
    def calculate_market_features(self, grid_df, category):
        """Calculate market features for prediction"""
        # Default market features per district (from training data analysis)
        market_features = {
            'Kebayoran Baru': {'transaction_count': 25, 'avg_external_gtv': 42000, 'avg_external_aov': 35000, 'category_diversity': 15},
            'Tebet': {'transaction_count': 18, 'avg_external_gtv': 36000, 'avg_external_aov': 32000, 'category_diversity': 12},
            'Setiabudi': {'transaction_count': 32, 'avg_external_gtv': 48000, 'avg_external_aov': 40000, 'category_diversity': 16},
            'Mampang Prapatan': {'transaction_count': 15, 'avg_external_gtv': 33000, 'avg_external_aov': 28000, 'category_diversity': 10},
            'Pancoran': {'transaction_count': 14, 'avg_external_gtv': 31000, 'avg_external_aov': 27000, 'category_diversity': 11},
            'Jagakarsa': {'transaction_count': 10, 'avg_external_gtv': 26000, 'avg_external_aov': 24000, 'category_diversity': 8},
            'Pasar Minggu': {'transaction_count': 12, 'avg_external_gtv': 28000, 'avg_external_aov': 25000, 'category_diversity': 9},
            'Cilandak': {'transaction_count': 20, 'avg_external_gtv': 37000, 'avg_external_aov': 33000, 'category_diversity': 13},
            'Pesanggrahan': {'transaction_count': 13, 'avg_external_gtv': 30000, 'avg_external_aov': 26000, 'category_diversity': 9},
            'Kebayoran Lama': {'transaction_count': 16, 'avg_external_gtv': 34000, 'avg_external_aov': 29000, 'category_diversity': 11}
        }
        
        # Add market features to grid
        for idx, row in grid_df.iterrows():
            district = row['districtName']
            if district in market_features:
                features = market_features[district]
                for key, value in features.items():
                    grid_df.loc[idx, key] = value
        
        # Calculate distance features
        major_areas = {
            'sudirman': (-6.2088, 106.8217),
            'blok_m': (-6.2440, 106.7985),
            'kemang': (-6.2614, 106.8149),
            'pondok_indah': (-6.2669, 106.7831),
            'senayan': (-6.2297, 106.8019)
        }
        
        for area_name, (lat, lng) in major_areas.items():
            grid_df[f'distance_to_{area_name}'] = np.sqrt(
                (grid_df['latitude'] - lat)**2 + 
                (grid_df['longitude'] - lng)**2
            ) * 111
        
        # Add other required features
        grid_df['accessibility_score'] = (
            1 / (1 + grid_df['distance_to_center']) * 0.3 +
            1 / (1 + grid_df['distance_to_sudirman']) * 0.25 +
            1 / (1 + grid_df['distance_to_blok_m']) * 0.2 +
            1 / (1 + grid_df['distance_to_kemang']) * 0.15 +
            1 / (1 + grid_df['distance_to_pondok_indah']) * 0.1
        )
        
        # Add business features
        grid_df['gtv_2024'] = grid_df['avg_external_gtv']
        grid_df['aov_2024'] = grid_df['avg_external_aov']
        grid_df['external_aov'] = grid_df['avg_external_aov']
        grid_df['Qty'] = 10  # Estimated average
        grid_df['log_qty'] = np.log1p(grid_df['Qty'])
        grid_df['has_historical_gtv'] = 1
        
        # Add market features
        grid_df['location_diversity_lat'] = 5
        grid_df['location_diversity_lng'] = 5
        grid_df['avg_qty_volume'] = 8
        grid_df['total_lat_diversity'] = 10
        grid_df['total_lng_diversity'] = 10
        grid_df['market_density'] = grid_df['transaction_count'] / 25
        grid_df['market_competition'] = grid_df['transaction_count'] / 1000
        grid_df['district_maturity'] = grid_df['category_diversity'] / 18
        grid_df['gtv_data_availability'] = 0.8
        grid_df['external_performance_indicator'] = grid_df['avg_external_gtv'] / (grid_df['avg_external_aov'] + 1)
        
        return grid_df
    
    def predict_locations(self, category, district_name, top_n=10, grid_density='medium'):
        """Main prediction function"""
        if category not in self.models:
            raise ValueError(f"No model available for category: {category}")
        
        if district_name not in self.district_data:
            raise ValueError(f"District not found: {district_name}")
        
        # Generate smart grid for the district
        grid_df = self.generate_smart_grid(district_name, grid_density)
        
        # Calculate market features
        grid_df = self.calculate_market_features(grid_df, category)
        
        # Prepare features for prediction
        missing_features = [col for col in self.feature_names if col not in grid_df.columns]
        for feature in missing_features:
            grid_df[feature] = 0
        
        # Ensure all required features are present
        X = grid_df[self.feature_names].fillna(0)
        
        # Scale features and predict
        X_scaled = self.scalers[category].transform(X)
        predictions = self.models[category].predict(X_scaled)
        
        # Calculate recommendation scores
        grid_df['predicted_revenue'] = predictions
        
        # Competition penalty and distance bonus
        grid_df['competition_penalty'] = 1 / (1 + grid_df['transaction_count'] / 10)
        optimal_distance = 3
        grid_df['distance_bonus'] = 1 - abs(grid_df['distance_to_center'] - optimal_distance) / 10
        grid_df['distance_bonus'] = np.clip(grid_df['distance_bonus'], 0.1, 1)
        
        # Final recommendation score
        grid_df['recommendation_score'] = (
            (grid_df['predicted_revenue'] / grid_df['predicted_revenue'].max()) * 0.6 +
            grid_df['competition_penalty'] * 0.2 +
            grid_df['distance_bonus'] * 0.2
        ) * 100
        
        # Get top recommendations
        recommendations = grid_df.nlargest(top_n, 'recommendation_score')
        
        # Add reasoning
        recommendations['reasoning'] = recommendations.apply(self._generate_reasoning, axis=1)
        
        result_columns = [
            'latitude', 'longitude', 'districtName', 'predicted_revenue',
            'recommendation_score', 'distance_to_center', 'transaction_count', 'reasoning'
        ]
        
        return recommendations[result_columns].reset_index(drop=True)
    
    def _generate_reasoning(self, row):
        """Generate human-readable reasoning for each recommendation"""
        reasons = []
        
        # Revenue prediction
        if row['predicted_revenue'] > 100000:
            reasons.append("High revenue potential")
        elif row['predicted_revenue'] > 50000:
            reasons.append("Good revenue potential")
        else:
            reasons.append("Moderate revenue potential")
        
        # Competition analysis
        if row['transaction_count'] < 15:
            reasons.append("Low competition")
        elif row['transaction_count'] < 25:
            reasons.append("Moderate competition")
        else:
            reasons.append("High competition area")
        
        # Location analysis
        if row['distance_to_center'] < 2:
            reasons.append("Prime central location")
        elif row['distance_to_center'] < 5:
            reasons.append("Good accessibility")
        else:
            reasons.append("Residential area")
        
        return " | ".join(reasons)

@st.cache_resource
def load_predictor():
    """Load the predictor with caching"""
    predictor = RestaurantLocationPredictor()
    return predictor

def create_map(recommendations, district_center):
    """Create folium map with recommendations"""
    # Create map centered on district
    m = folium.Map(
        location=district_center,
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    
    # Add district center marker
    folium.Marker(
        district_center,
        popup="District Center",
        icon=folium.Icon(color='blue', icon='home')
    ).add_to(m)
    
    # Add recommendation markers
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    for idx, row in recommendations.iterrows():
        color = colors[min(idx, len(colors)-1)]
        
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=f"""
                <b>Rank #{idx+1}</b><br>
                Score: {row['recommendation_score']:.1f}/100<br>
                Revenue: Rp {row['predicted_revenue']:,.0f}<br>
                Distance: {row['distance_to_center']:.1f} km<br>
                {row['reasoning']}
            """,
            icon=folium.Icon(color=color, icon='star')
        ).add_to(m)
    
    return m

def main():
    # Header
    st.markdown('<h1 class="main-header">🏪 Restaurant Location Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">AI-powered location intelligence for Jakarta Selatan restaurants</div>', unsafe_allow_html=True)
    
    # Load predictor
    try:
        predictor = load_predictor()
        available_categories = predictor.get_available_categories()
        available_districts = predictor.get_available_districts()
        
        if not available_categories:
            st.error("❌ No trained models found! Please train models first.")
            return
            
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please ensure models are trained and saved in 'models/' directory")
        return
    
    # Sidebar controls
    st.sidebar.header("🎯 Analysis Configuration")
    
    # Category selection
    category = st.sidebar.selectbox(
        "🍽️ Restaurant Category",
        options=available_categories,
        help="Select the type of restaurant you want to analyze"
    )
    
    # District selection
    district = st.sidebar.selectbox(
        "📍 Target District",
        options=available_districts,
        help="Select the district in Jakarta Selatan to analyze"
    )
    
    # Analysis parameters
    st.sidebar.subheader("⚙️ Analysis Parameters")
    
    grid_density = st.sidebar.select_slider(
        "Analysis Density",
        options=['low', 'medium', 'high'],
        value='medium',
        help="Higher density = more detailed analysis but slower"
    )
    
    top_n = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        help="How many top locations to show"
    )
    
    # Analysis button
    if st.sidebar.button("🔍 Analyze Locations", type="primary"):
        if category and district:
            with st.spinner(f"Analyzing best locations for {category} in {district}..."):
                try:
                    # Get recommendations
                    recommendations = predictor.predict_locations(
                        category=category,
                        district_name=district,
                        top_n=top_n,
                        grid_density=grid_density
                    )
                    
                    # Store in session state
                    st.session_state.recommendations = recommendations
                    st.session_state.category = category
                    st.session_state.district = district
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    return
        else:
            st.sidebar.error("Please select both category and district")
    
    # Display results
    if 'recommendations' in st.session_state:
        recommendations = st.session_state.recommendations
        category = st.session_state.category
        district = st.session_state.district
        
        st.success(f"✅ Analysis complete for **{category}** in **{district}**")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = recommendations['recommendation_score'].mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_score:.1f}/100</h3><p>Avg Score</p></div>', unsafe_allow_html=True)
        
        with col2:
            avg_revenue = recommendations['predicted_revenue'].mean()
            st.markdown(f'<div class="metric-card"><h3>Rp {avg_revenue:,.0f}</h3><p>Avg Revenue</p></div>', unsafe_allow_html=True)
        
        with col3:
            best_score = recommendations['recommendation_score'].max()
            st.markdown(f'<div class="metric-card"><h3>{best_score:.1f}/100</h3><p>Best Score</p></div>', unsafe_allow_html=True)
        
        with col4:
            avg_distance = recommendations['distance_to_center'].mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_distance:.1f} km</h3><p>Avg Distance</p></div>', unsafe_allow_html=True)
        
        # Create two columns for map and recommendations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📍 Location Map")
            district_center = predictor.district_data[district]['center']
            map_obj = create_map(recommendations.head(10), district_center)
            st_folium(map_obj, width=700, height=500)
        
        with col2:
            st.subheader("🏆 Top Recommendations")
            for idx, row in recommendations.head(5).iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>#{idx+1} Score: {row['recommendation_score']:.1f}/100</h4>
                        <p><strong>Revenue:</strong> Rp {row['predicted_revenue']:,.0f}</p>
                        <p><strong>Location:</strong> {row['latitude']:.4f}, {row['longitude']:.4f}</p>
                        <p><strong>Distance:</strong> {row['distance_to_center']:.1f} km from center</p>
                        <p><strong>Analysis:</strong> {row['reasoning']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Detailed analysis
        st.subheader("📊 Detailed Analysis")
        
        # Revenue distribution chart
        fig_revenue = px.histogram(
            recommendations, 
            x='predicted_revenue',
            title="Revenue Distribution",
            labels={'predicted_revenue': 'Predicted Revenue (IDR)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Score vs Distance scatter
        fig_scatter = px.scatter(
            recommendations,
            x='distance_to_center',
            y='recommendation_score',
            size='predicted_revenue',
            title="Score vs Distance from Center",
            labels={
                'distance_to_center': 'Distance from Center (km)',
                'recommendation_score': 'Recommendation Score',
                'predicted_revenue': 'Predicted Revenue (IDR)'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Data table
        st.subheader("📋 Complete Results")
        st.dataframe(
            recommendations[['latitude', 'longitude', 'predicted_revenue', 'recommendation_score', 'distance_to_center', 'reasoning']],
            use_container_width=True
        )
        
        # Download button
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"restaurant_locations_{category}_{district}.csv",
            mime="text/csv"
        )
    
    else:
        # Show example/instructions
        st.info("👆 Select a category and district from the sidebar, then click 'Analyze Locations' to begin")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 How it works")
            st.markdown("""
            1. **Select** restaurant category (Coffee, Japanese Food, etc.)
            2. **Choose** target district in Jakarta Selatan
            3. **Configure** analysis parameters
            4. **Click** Analyze Locations button
            5. **Review** AI-generated recommendations with scores and reasoning
            """)
        
        with col2:
            st.subheader("🎯 What you get")
            st.markdown("""
            - **Location Scoring**: AI-powered ranking of best spots
            - **Revenue Predictions**: Expected revenue for each location
            - **Interactive Map**: Visual representation of recommendations
            - **Detailed Analysis**: Competition, accessibility, and market factors
            - **Export Options**: Download results for further analysis
            """)

if __name__ == "__main__":
    main()
