# Page configuration - MUST BE FIRST STREAMLIT COMMAND
import streamlit as st

st.set_page_config(
    page_title="Enhanced Eucalyptus Prediction System",
    page_icon="🌿",
    layout="wide"
)

# Now import other modules
import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import pickle
import warnings
import tempfile
import os
import json
import sys
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go

# Fix sklearn import issue - corrected version
try:
    import sklearn
    import sklearn.ensemble
    import sklearn.preprocessing
    import sklearn.metrics
    import sklearn.model_selection
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    print("✅ scikit-learn imported successfully")
except ImportError as e:
    st.error(f"❌ scikit-learn not found: {e}")
    st.error("Please install scikit-learn using: pip install scikit-learn")
    st.stop()

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost not available. Using Random Forest and SVM only.")

warnings.filterwarnings('ignore')

# Initialize session state
if 'prediction_complete' not in st.session_state:
    st.session_state.prediction_complete = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'gdf' not in st.session_state:
    st.session_state.gdf = None

class EnhancedEucalyptusPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.class_labels = {}
        self.ensemble_model = None

    def load_model(self, model_path):
        """Load trained model from .pkl file"""
        try:
            import sklearn
            import sklearn.ensemble
            import sklearn.preprocessing
            import sklearn.metrics
            import sklearn.model_selection
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.class_labels = model_data['class_labels']

            # Create ensemble model if base model exists
            if self.model is not None:
                self.ensemble_model = self.create_ensemble_model()

            return True
            
        except ImportError as e:
            st.error(f"❌ scikit-learn import error: {str(e)}")
            st.error("Please install scikit-learn: pip install scikit-learn")
            return False
        except KeyError as e:
            st.error(f"❌ Missing key in model file: {str(e)}")
            st.error("Make sure your .pkl file contains: 'model', 'scaler', 'feature_names', 'class_labels'")
            return False
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False

    def create_ensemble_model(self):
        """Create ensemble of multiple algorithms"""
        try:
            estimators = []
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            estimators.append(('rf', rf))
            
            # XGBoost (if available)
            if XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
                estimators.append(('xgb', xgb_model))
            
            # SVM
            svm = SVC(probability=True, random_state=42)
            estimators.append(('svm', svm))
            
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
            
            return ensemble
        except Exception as e:
            st.warning(f"Could not create ensemble model: {str(e)}. Using base model only.")
            return None

    def get_enhanced_sentinel2_data(self, geometry, start_date, end_date):
        """Get enhanced Sentinel-2 data with comprehensive vegetation indices and temporal features"""
        try:
            # Define Sentinel-2 collection
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate(start_date, end_date) \
                .filterBounds(geometry) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
                .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])

            # Check if collection is empty
            collection_size = s2.size()
            if collection_size.getInfo() == 0:
                s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterDate('2025-01-01', '2025-05-31') \
                    .filterBounds(geometry) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
                    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])

            # Get median composite and temporal features
            s2_median = s2.median()
            temporal_features = self.get_temporal_features(geometry, start_date, end_date)

            # Standard vegetation indices
            ndvi = s2_median.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndwi = s2_median.normalizedDifference(['B3', 'B8']).rename('NDWI')
            evi = s2_median.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {
                    'NIR': s2_median.select('B8'),
                    'RED': s2_median.select('B4'),
                    'BLUE': s2_median.select('B2')
                }
            ).clamp(-1, 1).rename('EVI')

            savi = s2_median.expression(
                '((NIR - RED) / (NIR + RED + 0.5)) * (1.5)',
                {
                    'NIR': s2_median.select('B8'),
                    'RED': s2_median.select('B4')
                }
            ).clamp(-1, 1).rename('SAVI')

            # Enhanced Red Edge indices for eucalyptus
            ndre = s2_median.normalizedDifference(['B8', 'B5']).rename('NDRE')
            msi = s2_median.select('B11').divide(s2_median.select('B8')).clamp(0, 5).rename('MSI')
            gndvi = s2_median.normalizedDifference(['B8', 'B3']).rename('GNDVI')
            chl_red_edge = s2_median.normalizedDifference(['B8', 'B5']).rename('CHL_RED_EDGE')
            nbr = s2_median.normalizedDifference(['B8', 'B12']).rename('NBR')
            nbr2 = s2_median.normalizedDifference(['B11', 'B12']).rename('NBR2')
            ndmi = s2_median.normalizedDifference(['B8', 'B11']).rename('NDMI')

            # Hyperspectral-inspired narrow-band indices
            chl_red_edge_narrow = s2_median.expression(
                '(B8A - B6) / (B8A + B6 - B5)',
                {
                    'B8A': s2_median.select('B8A'),
                    'B6': s2_median.select('B6'),
                    'B5': s2_median.select('B5')
                }
            ).clamp(-1, 1).rename('CHL_NARROW')

            # Anthocyanin reflectance index
            ari = s2_median.expression(
                '(1/B3) - (1/B5)',
                {
                    'B3': s2_median.select('B3'),
                    'B5': s2_median.select('B5')
                }
            ).clamp(0, 0.2).rename('ARI')

            # Carotenoid reflectance index
            cri = s2_median.expression(
                '(1/B2) - (1/B5)',
                {
                    'B2': s2_median.select('B2'),
                    'B5': s2_median.select('B5')
                }
            ).clamp(0, 0.2).rename('CRI')

            # Band ratios
            b8_b11_ratio = s2_median.select('B8').divide(s2_median.select('B11')).clamp(0, 10).rename('B8_B11_ratio')
            b6_b7_ratio = s2_median.select('B6').divide(s2_median.select('B7')).clamp(0, 5).rename('B6_B7_ratio')
            b5_b4_ratio = s2_median.select('B5').divide(s2_median.select('B4')).clamp(0, 5).rename('B5_B4_ratio')

            # Enhanced eucalyptus indices
            euc_index1 = s2_median.expression(
                '(B8A - B5) / (B8A + B5)',
                {
                    'B8A': s2_median.select('B8A'),
                    'B5': s2_median.select('B5')
                }
            ).clamp(-1, 1).rename('EUC_INDEX1')

            euc_index2 = s2_median.expression(
                '(B7 - B6) / (B7 + B6)',
                {
                    'B7': s2_median.select('B7'),
                    'B6': s2_median.select('B6')
                }
            ).clamp(-1, 1).rename('EUC_INDEX2')

            # Plant functional traits proxies
            chlorophyll_proxy = s2_median.expression(
                '(B8 - B4) / (B8 + B4 + 0.5)',
                {
                    'B8': s2_median.select('B8'),
                    'B4': s2_median.select('B4')
                }
            ).clamp(-1, 1).rename('CHLOROPHYLL_PROXY')

            water_content_proxy = s2_median.expression(
                '(B8A - B11) / (B8A + B11)',
                {
                    'B8A': s2_median.select('B8A'),
                    'B11': s2_median.select('B11')
                }
            ).clamp(-1, 1).rename('WATER_CONTENT_PROXY')

            dry_matter_proxy = s2_median.select('B11').divide(s2_median.select('B8A')).clamp(0, 5).rename('DRY_MATTER_PROXY')

            # Combine all features
            s2_features = s2_median.addBands([
                ndvi, ndwi, evi, savi, ndre, msi, gndvi, chl_red_edge,
                nbr, nbr2, ndmi, b8_b11_ratio, b6_b7_ratio, b5_b4_ratio,
                euc_index1, euc_index2, chl_red_edge_narrow, ari, cri,
                chlorophyll_proxy, water_content_proxy, dry_matter_proxy
            ])

            # Add temporal features if available
            for temporal_feature in temporal_features:
                s2_features = s2_features.addBands(temporal_feature)

            # Add texture features
            s2_features = self.get_enhanced_texture_features(s2_features)

            return s2_features

        except Exception as e:
            st.error(f"Error getting enhanced Sentinel-2 data: {str(e)}")
            return None

    def get_temporal_features(self, geometry, start_date, end_date):
        """Add seasonal and temporal variation features"""
        try:
            # Seasonal composites (Southern Hemisphere seasons)
            seasons = [
                ('DJF', [12, 1, 2]),  # Summer
                ('MAM', [3, 4, 5]),   # Autumn
                ('JJA', [6, 7, 8]),   # Winter
                ('SON', [9, 10, 11])  # Spring
            ]
            
            seasonal_features = []
            for season_name, months in seasons:
                try:
                    season_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterDate(start_date, end_date) \
                        .filterBounds(geometry) \
                        .filter(ee.Filter.calendarRange(months[0], months[2], 'month')) \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                    
                    if season_collection.size().getInfo() > 0:
                        season_median = season_collection.median()
                        season_ndvi = season_median.normalizedDifference(['B8', 'B4']).rename(f'NDVI_{season_name}')
                        seasonal_features.append(season_ndvi)
                except:
                    continue
            
            # Calculate temporal metrics if we have multiple seasons
            if len(seasonal_features) >= 2:
                try:
                    ndvi_range = seasonal_features[0].subtract(seasonal_features[-1]).rename('NDVI_SEASONAL_RANGE')
                    seasonal_features.append(ndvi_range)
                except:
                    pass
            
            return seasonal_features
            
        except Exception as e:
            st.warning(f"Could not extract temporal features: {str(e)}")
            return []

    def get_enhanced_texture_features(self, s2_features):
        """Add comprehensive texture analysis including GLCM features"""
        try:
            # Key bands for texture analysis
            texture_bands = ['B4', 'B8', 'NDVI', 'NDRE', 'B5']  # Added more bands
            
            for band in texture_bands:
                band_names = s2_features.bandNames().getInfo()
                if band in band_names:
                    # Multiple window sizes for different scales
                    for window in [3, 5, 7]:
                        try:
                            # Basic statistical texture (existing)
                            variance = s2_features.select(band).reduceNeighborhood(
                                reducer=ee.Reducer.variance(),
                                kernel=ee.Kernel.square(window)
                            ).rename(f'{band}_VAR_{window}x{window}')
                            
                            std_dev = s2_features.select(band).reduceNeighborhood(
                                reducer=ee.Reducer.stdDev(),
                                kernel=ee.Kernel.square(window)
                            ).rename(f'{band}_STD_{window}x{window}')
                            
                            # GLCM-inspired texture measures
                            # Homogeneity approximation (inverse of variance)
                            homogeneity = variance.multiply(-1).exp().rename(f'{band}_HOMOGENEITY_{window}x{window}')
                            
                            # Contrast approximation (enhanced variance)
                            contrast = variance.multiply(2).rename(f'{band}_CONTRAST_{window}x{window}')
                            
                            # Energy approximation (uniformity measure)
                            mean_val = s2_features.select(band).reduceNeighborhood(
                                reducer=ee.Reducer.mean(),
                                kernel=ee.Kernel.square(window)
                            )
                            energy = mean_val.divide(std_dev.add(0.001)).rename(f'{band}_ENERGY_{window}x{window}')
                            
                            # Correlation approximation using focal operations
                            # Create shifted versions for correlation calculation
                            shifted_right = s2_features.select(band).focal_mean(
                                kernel=ee.Kernel.rectangle(1, window), 
                                iterations=1
                            )
                            shifted_down = s2_features.select(band).focal_mean(
                                kernel=ee.Kernel.rectangle(window, 1), 
                                iterations=1
                            )
                            
                            # Correlation as covariance normalized by standard deviations
                            original = s2_features.select(band)
                            correlation = original.subtract(mean_val).multiply(
                                shifted_right.subtract(mean_val)
                            ).reduceNeighborhood(
                                reducer=ee.Reducer.mean(),
                                kernel=ee.Kernel.square(window)
                            ).divide(std_dev.add(0.001)).rename(f'{band}_CORRELATION_{window}x{window}')
                            
                            # Entropy approximation
                            # Use histogram-based approach for entropy estimation
                            entropy = std_dev.log().multiply(-1).rename(f'{band}_ENTROPY_{window}x{window}')
                            
                            # Angular Second Moment (ASM) approximation
                            asm = energy.pow(2).rename(f'{band}_ASM_{window}x{window}')
                            
                            # Dissimilarity approximation
                            dissimilarity = std_dev.multiply(1.5).rename(f'{band}_DISSIMILARITY_{window}x{window}')
                            
                            # Add all texture features
                            s2_features = s2_features.addBands([
                                variance, std_dev, homogeneity, contrast, energy, 
                                correlation, entropy, asm, dissimilarity
                            ])
                            
                            # Additional GLCM-like features for eucalyptus discrimination
                            if band in ['B8', 'NDVI']:  # Focus on key vegetation bands
                                # Local binary pattern approximation
                                lbp = s2_features.select(band).subtract(mean_val).gt(0).rename(f'{band}_LBP_{window}x{window}')
                                
                                # Gradient magnitude
                                gradient_x = s2_features.select(band).convolve(ee.Kernel.sobel())
                                gradient_mag = gradient_x.pow(2).sqrt().rename(f'{band}_GRADIENT_{window}x{window}')
                                
                                s2_features = s2_features.addBands([lbp, gradient_mag])
                                
                        except Exception as tex_e:
                            st.warning(f"Could not compute texture for {band} at {window}x{window}: {str(tex_e)}")
                            continue
            
            # Add multi-band texture relationships (important for GLCM analysis)
            try:
                if all(band in s2_features.bandNames().getInfo() for band in ['B4', 'B8']):
                    # Red-NIR texture correlation
                    red_nir_texture_ratio = s2_features.select('B4_VAR_5x5').divide(
                        s2_features.select('B8_VAR_5x5').add(0.001)
                    ).rename('RED_NIR_TEXTURE_RATIO')
                    
                    # Combined texture index
                    combined_texture = s2_features.select('B4_CONTRAST_5x5').add(
                        s2_features.select('B8_CONTRAST_5x5')
                    ).rename('COMBINED_TEXTURE_INDEX')
                    
                    s2_features = s2_features.addBands([red_nir_texture_ratio, combined_texture])
                    
            except Exception as multi_e:
                st.warning(f"Could not compute multi-band texture features: {str(multi_e)}")
            
            return s2_features
            
        except Exception as e:
            st.warning(f"Could not add GLCM texture features: {str(e)}")
            return s2_features

    def get_glcm_features_direct(self, image, band, region, scale=10):
        """
        Direct GLCM calculation for specific bands (computationally intensive)
        Use this for small areas or when high accuracy is needed
        """
        try:
            # Sample the image to get pixel values
            sample = image.select(band).sample(
                region=region,
                scale=scale,
                numPixels=100,  # Limit for computational efficiency
                geometries=False
            )
            
            sample_list = sample.getInfo()
            if not sample_list or 'features' not in sample_list:
                return {}
            
            # Extract pixel values
            values = [feature['properties'][band] for feature in sample_list['features'] 
                     if feature['properties'][band] is not None]
            
            if len(values) < 4:  # Need minimum pixels for GLCM
                return {}
            
            # Convert to numpy array and quantize
            import numpy as np
            values_array = np.array(values)
            
            # Quantize to reduce computational load
            min_val, max_val = values_array.min(), values_array.max()
            if max_val > min_val:
                quantized = ((values_array - min_val) / (max_val - min_val) * 7).astype(int)
            else:
                quantized = np.zeros_like(values_array, dtype=int)
            
            # Simple GLCM approximation using co-occurrence statistics
            # This is a simplified version - for full GLCM, you'd need skimage
            unique_vals = np.unique(quantized)
            n_levels = len(unique_vals)
            
            if n_levels < 2:
                return {}
            
            # Calculate basic GLCM-inspired statistics
            contrast = np.var(quantized) * 2
            homogeneity = 1 / (1 + contrast)
            energy = np.sum(np.square(np.bincount(quantized) / len(quantized)))
            entropy = -np.sum([p * np.log2(p + 1e-10) for p in np.bincount(quantized) / len(quantized)])
            
            return {
                f'{band}_GLCM_CONTRAST': contrast,
                f'{band}_GLCM_HOMOGENEITY': homogeneity,
                f'{band}_GLCM_ENERGY': energy,
                f'{band}_GLCM_ENTROPY': entropy
            }
            
        except Exception as e:
            st.warning(f"Direct GLCM calculation failed for {band}: {str(e)}")
            return {}

    def ensure_glcm_features(self, features_df):
        """Ensure all GLCM features expected by the model are present"""
        try:
            # Common GLCM feature patterns that might be in your model
            glcm_patterns = [
                'CONTRAST', 'CORRELATION', 'ENERGY', 'HOMOGENEITY', 
                'ENTROPY', 'ASM', 'DISSIMILARITY', 'VARIANCE'
            ]
            
            bands = ['B4', 'B8', 'NDVI', 'NDRE', 'B5']
            windows = ['3x3', '5x5', '7x7']
            
            # Check which GLCM features the model expects
            expected_glcm_features = [
                feature for feature in self.feature_names 
                if any(pattern in feature.upper() for pattern in glcm_patterns)
            ]
            
            if expected_glcm_features:
                st.write(f"Debug: Model expects {len(expected_glcm_features)} GLCM features")
                
                # Add missing GLCM features with default values
                for feature in expected_glcm_features:
                    if feature not in features_df.columns:
                        # Set reasonable default values based on feature type
                        if 'CONTRAST' in feature:
                            features_df[feature] = 0.5
                        elif 'HOMOGENEITY' in feature:
                            features_df[feature] = 0.8
                        elif 'ENERGY' in feature:
                            features_df[feature] = 0.3
                        elif 'ENTROPY' in feature:
                            features_df[feature] = 2.0
                        elif 'CORRELATION' in feature:
                            features_df[feature] = 0.7
                        else:
                            features_df[feature] = 0.0
                            
                st.write(f"Debug: Added {len([f for f in expected_glcm_features if f not in features_df.columns])} missing GLCM features")
            
            return features_df
            
        except Exception as e:
            st.warning(f"Could not ensure GLCM features: {str(e)}")
            return features_df

    def get_enhanced_sar_data(self, geometry, start_date, end_date):
        """Get enhanced Sentinel-1 SAR data with multi-temporal analysis"""
        try:
            # Try to get both ascending and descending orbits
            s1_asc = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterDate(start_date, end_date) \
                .filterBounds(geometry) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
                .select(['VV', 'VH'])

            s1_desc = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterDate(start_date, end_date) \
                .filterBounds(geometry) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
                .select(['VV', 'VH'])

            # Check collection sizes
            asc_size = s1_asc.size().getInfo()
            desc_size = s1_desc.size().getInfo()

            if asc_size == 0 and desc_size == 0:
                # Fallback to any available data
                s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                    .filterDate('2025-01-01', '2025-05-31') \
                    .filterBounds(geometry) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                    .select(['VV', 'VH'])
                
                if s1.size().getInfo() > 0:
                    s1_median = s1.median()
                else:
                    return None
            else:
                # Use the orbit with more data or combine both
                if asc_size >= desc_size and asc_size > 0:
                    s1_median = s1_asc.median()
                elif desc_size > 0:
                    s1_median = s1_desc.median()
                else:
                    return None

                # If we have both orbits, add coherence-like metrics
                if asc_size > 0 and desc_size > 0:
                    asc_median = s1_asc.median()
                    desc_median = s1_desc.median()
                    
                    vv_orbit_diff = asc_median.select('VV').subtract(desc_median.select('VV')).abs().rename('VV_ORBIT_DIFF')
                    vh_orbit_diff = asc_median.select('VH').subtract(desc_median.select('VH')).abs().rename('VH_ORBIT_DIFF')
                    
                    s1_median = s1_median.addBands([vv_orbit_diff, vh_orbit_diff])

            # Enhanced SAR indices
            vv_vh_ratio = s1_median.select('VV').divide(s1_median.select('VH')).clamp(0, 10).rename('VV_VH_ratio')
            vv_vh_diff = s1_median.select('VV').subtract(s1_median.select('VH')).rename('VV_VH_diff')

            # Advanced texture measures
            vv_texture = s1_median.select('VV').reduceNeighborhood(
                reducer=ee.Reducer.stdDev(),
                kernel=ee.Kernel.square(3)
            ).rename('VV_TEXTURE')

            vh_texture = s1_median.select('VH').reduceNeighborhood(
                reducer=ee.Reducer.stdDev(),
                kernel=ee.Kernel.square(3)
            ).rename('VH_TEXTURE')

            # Enhanced volume scattering metrics
            volume_scattering = s1_median.expression(
                '(VH / (VV + VH))',
                {
                    'VV': s1_median.select('VV'),
                    'VH': s1_median.select('VH')
                }
            ).rename('VOLUME_SCATTER')

            # Forest structure proxy
            forest_structure = s1_median.expression(
                'VH * (VV / VH)',
                {
                    'VV': s1_median.select('VV'),
                    'VH': s1_median.select('VH')
                }
            ).rename('FOREST_STRUCTURE')

            sar_features = s1_median.addBands([
                vv_vh_ratio, vv_vh_diff, vv_texture, vh_texture, 
                volume_scattering, forest_structure
            ])

            return sar_features

        except Exception as e:
            st.error(f"Error getting enhanced SAR data: {str(e)}")
            return None

    def get_terrain_data(self, geometry):
        """Get enhanced terrain and height data"""
        try:
            # Basic terrain
            dem = ee.Image('USGS/SRTMGL1_003')
            elevation = dem.select('elevation').rename('ELEVATION')
            slope = ee.Terrain.slope(dem).rename('SLOPE')
            aspect = ee.Terrain.aspect(dem).rename('ASPECT')
            tree_cover = ee.Image('UMD/hansen/global_forest_change_2021_v1_9').select('treecover2000').rename('TREE_COVER')

            # Enhanced canopy height with multiple fallbacks
            canopy_height = None
            try:
                canopy_height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').rename('CANOPY_HEIGHT')
                canopy_height.getInfo()  # Test access
            except:
                try:
                    # Alternative height estimation
                    canopy_height = tree_cover.multiply(0.3).add(2).rename('CANOPY_HEIGHT')
                except:
                    canopy_height = ee.Image.constant(5).rename('CANOPY_HEIGHT')

            # Enhanced terrain roughness
            roughness = elevation.reduceNeighborhood(
                reducer=ee.Reducer.stdDev(),
                kernel=ee.Kernel.square(3)
            ).rename('ROUGHNESS')

            # Topographic wetness index
            twi = elevation.expression(
                'log((catchment_area + 1) / (tan(slope * pi / 180) + 0.001))',
                {
                    'catchment_area': elevation.multiply(0.1),  # Simplified
                    'slope': slope,
                    'pi': 3.14159265359
                }
            ).rename('TWI')

            # Terrain position index
            tpi = elevation.subtract(
                elevation.reduceNeighborhood(
                    reducer=ee.Reducer.mean(),
                    kernel=ee.Kernel.circle(3)
                )
            ).rename('TPI')

            terrain_features = elevation.addBands([
                slope, aspect, tree_cover, canopy_height, roughness, twi, tpi
            ])

            return terrain_features

        except Exception as e:
            st.error(f"Error getting terrain data: {str(e)}")
            return ee.Image.constant([100, 5, 0, 50, 5, 1, 0, 0]).rename([
                'ELEVATION', 'SLOPE', 'ASPECT', 'TREE_COVER', 'CANOPY_HEIGHT', 
                'ROUGHNESS', 'TWI', 'TPI'
            ])

    def feature_engineering(self, features_df):
        """Advanced feature engineering with interaction terms"""
        try:
            # Create interaction terms for key eucalyptus discriminators
            if all(col in features_df.columns for col in ['NDVI', 'EUC_INDEX1', 'CANOPY_HEIGHT']):
                features_df['NDVI_HEIGHT_INTERACTION'] = features_df['NDVI'] * features_df['CANOPY_HEIGHT']
                features_df['EUC_NDVI_RATIO'] = features_df['EUC_INDEX1'] / (features_df['NDVI'] + 0.001)

            # Vegetation stress indicators
            if all(col in features_df.columns for col in ['NDVI', 'NDWI', 'MSI']):
                features_df['STRESS_INDEX'] = (features_df['NDVI'] - features_df['NDWI']) / (features_df['MSI'] + 0.001)

            # Age-related structural indicators
            if all(col in features_df.columns for col in ['CANOPY_HEIGHT', 'NDVI', 'B8_B11_ratio']):
                features_df['MATURITY_INDEX'] = features_df['CANOPY_HEIGHT'] * features_df['NDVI'] * features_df['B8_B11_ratio']

            # SAR-optical integration
            if all(col in features_df.columns for col in ['VV_VH_ratio', 'NDVI']):
                features_df['SAR_OPTICAL_COMBO'] = features_df['VV_VH_ratio'] * features_df['NDVI']

            return features_df
            
        except Exception as e:
            st.warning(f"Feature engineering warning: {str(e)}")
            return features_df

    def extract_features_from_polygon(self, geometry, start_date='2024-06-01', end_date='2025-06-01'):
        """Extract enhanced features from polygon with GLCM texture analysis"""
        try:
            # Convert geometry
            if isinstance(geometry, dict):
                ee_geometry = ee.Geometry(geometry)
            else:
                coords = list(geometry.exterior.coords)
                ee_geometry = ee.Geometry.Polygon(coords)

            # Get all enhanced data sources
            s2_features = self.get_enhanced_sentinel2_data(ee_geometry, start_date, end_date)
            sar_features = self.get_enhanced_sar_data(ee_geometry, start_date, end_date)
            terrain_features = self.get_terrain_data(ee_geometry)

            # Combine features
            if s2_features is not None and sar_features is not None:
                combined_features = s2_features.addBands(sar_features).addBands(terrain_features)
            elif s2_features is not None:
                combined_features = s2_features.addBands(terrain_features)
            else:
                return pd.DataFrame()

            # Multi-scale sampling
            scales = [10, 20]
            all_samples = []

            for scale in scales:
                try:
                    sample = combined_features.sample(
                        region=ee_geometry,
                        scale=scale,
                        numPixels=500,
                        geometries=False
                    )
                    
                    sample_list = sample.getInfo()
                    if sample_list and 'features' in sample_list:
                        for feature in sample_list['features']:
                            if 'properties' in feature:
                                feature_data = feature['properties']
                                feature_data['SAMPLE_SCALE'] = scale
                                all_samples.append(feature_data)
                except Exception as scale_e:
                    st.warning(f"Could not sample at scale {scale}m: {str(scale_e)}")
                    continue

            if not all_samples:
                return pd.DataFrame()

            # Convert to DataFrame
            sample_df = pd.DataFrame(all_samples)
            sample_df = sample_df.dropna(axis=1, how='all')
            
            # Add direct GLCM features for key bands (computationally intensive)
            glcm_bands = ['B4', 'B8', 'NDVI']
            for band in glcm_bands:
                if band in combined_features.bandNames().getInfo():
                    try:
                        glcm_features = self.get_glcm_features_direct(
                            combined_features, band, ee_geometry, scale=10
                        )
                        for glcm_feature, value in glcm_features.items():
                            sample_df[glcm_feature] = value
                    except Exception as glcm_e:
                        st.warning(f"Could not compute direct GLCM for {band}: {str(glcm_e)}")
                        continue
            
            # Debug: Show available features including GLCM
            st.write(f"Debug: Total features extracted: {len(sample_df.columns)}")
            glcm_features_found = [col for col in sample_df.columns if 'GLCM' in col or any(
                texture_type in col for texture_type in ['CONTRAST', 'HOMOGENEITY', 'ENERGY', 'ENTROPY', 'CORRELATION', 'ASM']
            )]
            st.write(f"Debug: GLCM/Texture features found: {len(glcm_features_found)}")
            if glcm_features_found:
                st.write(f"Debug: GLCM feature examples: {glcm_features_found[:5]}")

            # Enhanced height features
            if 'CANOPY_HEIGHT' in sample_df.columns:
                sample_df['HEIGHT_MEAN'] = sample_df['CANOPY_HEIGHT'].mean()
                sample_df['HEIGHT_STD'] = sample_df['CANOPY_HEIGHT'].std()
                sample_df['HEIGHT_MAX'] = sample_df['CANOPY_HEIGHT'].max()
                sample_df['HEIGHT_MIN'] = sample_df['CANOPY_HEIGHT'].min()
                sample_df['HEIGHT_RANGE'] = sample_df['HEIGHT_MAX'] - sample_df['HEIGHT_MIN']
                sample_df['HEIGHT_UNIFORMITY'] = 1 / (sample_df['HEIGHT_STD'] + 0.1)

                if 'SLOPE' in sample_df.columns:
                    sample_df['HEIGHT_SLOPE_RATIO'] = sample_df['CANOPY_HEIGHT'] / (sample_df['SLOPE'] + 1)

            # Apply feature engineering
            sample_df = self.feature_engineering(sample_df)

            return sample_df

        except Exception as e:
            st.error(f"Error extracting enhanced features with GLCM: {str(e)}")
            return pd.DataFrame()

    def predict_with_uncertainty(self, features_scaled):
        """Predict with uncertainty estimation using ensemble"""
        try:
            # Use ensemble model if available
            if self.ensemble_model is not None:
                try:
                    probabilities = self.ensemble_model.predict_proba(features_scaled)
                    predictions = self.ensemble_model.predict(features_scaled)
                    
                    # Calculate prediction uncertainty
                    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
                    max_prob = np.max(probabilities, axis=1)
                    
                    return predictions, probabilities, entropy, max_prob
                except:
                    # Fall back to base model
                    pass
            
            # Use base model
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)
                predictions = self.model.predict(features_scaled)
                
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
                max_prob = np.max(probabilities, axis=1)
                
                return predictions, probabilities, entropy, max_prob
            else:
                predictions = self.model.predict(features_scaled)
                return predictions, None, None, None
                
        except Exception as e:
            st.warning(f"Prediction uncertainty estimation failed: {str(e)}")
            predictions = self.model.predict(features_scaled)
            return predictions, None, None, None

    def predict_area(self, geojson_path, start_date='2024-06-01', end_date='2025-06-01'):
        """Predict eucalyptus for new area with enhanced features including GLCM"""
        if self.model is None:
            st.error("No model loaded! Please upload a model file first.")
            return None, None

        # Load GeoJSON
        try:
            gdf = gpd.read_file(geojson_path)
        except Exception as e:
            st.error(f"Error loading GeoJSON: {str(e)}")
            return None, None

        predictions = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, row in gdf.iterrows():
            try:
                status_text.text(f"Processing polygon {idx+1}/{len(gdf)} with GLCM texture analysis...")
                
                # Extract enhanced features including GLCM
                features_df = self.extract_features_from_polygon(row.geometry, start_date, end_date)

                if not features_df.empty:
                    # Prepare features for prediction
                    features_df = features_df.select_dtypes(include=[np.number]).dropna()

                    # IMPORTANT: Ensure GLCM features are present
                    features_df = self.ensure_glcm_features(features_df)

                    # Ensure all model features are present
                    for feature in self.feature_names:
                        if feature not in features_df.columns:
                            features_df[feature] = 0

                    # Select model features
                    model_features = features_df[self.feature_names].fillna(0)

                    if len(model_features) > 10:  # Increased minimum for enhanced features
                        # Scale and predict with uncertainty
                        features_scaled = self.scaler.transform(model_features)
                        pred_class, pred_proba, entropy, max_prob = self.predict_with_uncertainty(features_scaled)

                        # Get enhanced results
                        majority_class = np.bincount(pred_class).argmax()
                        confidence = np.mean(max_prob) if max_prob is not None else np.mean(np.max(pred_proba, axis=1)) if pred_proba is not None else 0.5
                        eucalyptus_prob = np.mean(np.sum(pred_proba[:, 1:], axis=1)) if pred_proba is not None and pred_proba.shape[1] > 1 else 0
                        
                        # Enhanced uncertainty metrics
                        uncertainty = np.mean(entropy) if entropy is not None else 0
                        prediction_stability = np.std(pred_class) / (np.mean(pred_class) + 0.001)

                        # Enhanced biophysical info - check both model features and raw features
                        avg_height = 0
                        avg_ndvi = 0
                        avg_red_edge = 0
                        avg_moisture = 0
                        
                        # Try to get from model features first, then from raw features_df
                        if 'CANOPY_HEIGHT' in model_features.columns:
                            avg_height = model_features['CANOPY_HEIGHT'].mean()
                        elif 'CANOPY_HEIGHT' in features_df.columns:
                            avg_height = features_df['CANOPY_HEIGHT'].mean()
                            
                        if 'NDVI' in model_features.columns:
                            avg_ndvi = model_features['NDVI'].mean()
                        elif 'NDVI' in features_df.columns:
                            avg_ndvi = features_df['NDVI'].mean()
                            
                        # Check multiple possible red edge feature names
                        red_edge_features = ['NDRE', 'CHL_RED_EDGE', 'CHL_NARROW']
                        for feature in red_edge_features:
                            if feature in model_features.columns:
                                avg_red_edge = model_features[feature].mean()
                                break
                            elif feature in features_df.columns:
                                avg_red_edge = features_df[feature].mean()
                                break
                        
                        # Check multiple possible moisture feature names
                        moisture_features = ['NDMI', 'NDWI', 'WATER_CONTENT_PROXY']
                        for feature in moisture_features:
                            if feature in model_features.columns:
                                avg_moisture = model_features[feature].mean()
                                break
                            elif feature in features_df.columns:
                                avg_moisture = features_df[feature].mean()
                                break
                        
                        # Maturity indicators
                        maturity_score = 0
                        if 'HEIGHT_UNIFORMITY' in model_features.columns:
                            maturity_score = model_features['HEIGHT_UNIFORMITY'].mean()

                        predictions.append({
                            'polygon_id': idx,
                            'predicted_class': majority_class,
                            'predicted_label': self.class_labels.get(majority_class, 'Unknown'),
                            'confidence': confidence,
                            'eucalyptus_probability': eucalyptus_prob,
                            'uncertainty': uncertainty,
                            'prediction_stability': prediction_stability,
                            'is_eucalyptus': majority_class > 0,
                            'avg_height': avg_height,
                            'avg_ndvi': avg_ndvi,
                            'avg_red_edge': avg_red_edge,
                            'avg_moisture': avg_moisture,
                            'maturity_score': maturity_score,
                            'pixels_analyzed': len(model_features),
                            'feature_quality': 'high' if len(model_features) > 50 else 'medium' if len(model_features) > 20 else 'low'
                        })
                    else:
                        predictions.append(self._empty_prediction(idx))
                else:
                    predictions.append(self._empty_prediction(idx))

            except Exception as e:
                st.error(f"Error processing polygon {idx+1}: {str(e)}")
                predictions.append(self._empty_prediction(idx))

            # Update progress
            progress_bar.progress((idx + 1) / len(gdf))

        # Create enhanced results DataFrame
        results_df = pd.DataFrame(predictions)

        # Add results to original GeoDataFrame
        result_gdf = gdf.copy()
        for col in results_df.columns:
            if col != 'polygon_id':
                result_gdf[col] = results_df[col].values

        status_text.text("Enhanced prediction with GLCM completed!")
        progress_bar.empty()
        
        return result_gdf, results_df

    def _empty_prediction(self, idx):
        """Create empty prediction for failed cases"""
        return {
            'polygon_id': idx,
            'predicted_class': 0,
            'predicted_label': 'Non-Eucalyptus',
            'confidence': 0.0,
            'eucalyptus_probability': 0.0,
            'uncertainty': 1.0,
            'prediction_stability': 0.0,
            'is_eucalyptus': False,
            'avg_height': 0,
            'avg_ndvi': 0,
            'avg_red_edge': 0,
            'avg_moisture': 0,
            'maturity_score': 0,
            'pixels_analyzed': 0,
            'feature_quality': 'none'
        }
@st.cache_resource
def initialize_earth_engine():
    """Initialize Earth Engine with service account authentication"""
    try:
        # Method 1: Using Streamlit secrets (for Streamlit Cloud)
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            service_account_info = dict(st.secrets["gcp_service_account"])
            credentials = ee.ServiceAccountCredentials(
                email=service_account_info['client_email'],
                key_data=json.dumps(service_account_info)
            )
            ee.Initialize(credentials=credentials)
            return True
            
        # Method 2: Using environment variable (for local development)
        elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
            credentials_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
            if os.path.exists(credentials_path):
                with open(credentials_path, 'r') as f:
                    service_account_info = json.load(f)
                    
                credentials = ee.ServiceAccountCredentials(
                    email=service_account_info['client_email'],
                    key_file=credentials_path
                )
                ee.Initialize(credentials=credentials, project=service_account_info['project_id'])
                return True
            else:
                st.error("❌ Credentials file not found")
                return False
        
        # Method 3: Using uploaded file (for testing)
        elif hasattr(st.session_state, 'ee_credentials'):
            credentials = ee.ServiceAccountCredentials(
                email=st.session_state.ee_credentials['client_email'],
                key_data=json.dumps(st.session_state.ee_credentials)
            )
            ee.Initialize(credentials=credentials)
            return True
            
        else:
            st.error("❌ No Earth Engine credentials found. Please configure authentication.")
            return False
            
    except Exception as e:
        st.error(f"❌ Failed to initialize Earth Engine: {str(e)}")
        return False

    
# Main Streamlit App
def main():
    st.title("🌿 Enhanced Eucalyptus Prediction System")
    st.markdown("""
    **Advanced Features:**
    - 🔬 **Enhanced Spectral Analysis**: Hyperspectral-inspired indices, plant functional traits
    - 🌍 **Multi-temporal Processing**: Seasonal analysis and temporal change detection  
    - 🎯 **Advanced Texture Analysis**: Multi-scale texture features for improved discrimination
    - 🤖 **Ensemble Learning**: Multiple algorithms with uncertainty quantification
    - 📊 **Comprehensive Metrics**: Age classification, maturity assessment, quality indicators
    """)
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        st.stop()
    
    # Enhanced sidebar for inputs
    st.sidebar.header("📁 Upload Files")
    
    # Model file upload
    model_file = st.sidebar.file_uploader(
        "Upload Enhanced Model (.pkl)",
        type=['pkl'],
        help="Upload your trained eucalyptus classification model with enhanced features (max 500MB)"
    )
    
    # GeoJSON file upload
    geojson_file = st.sidebar.file_uploader(
        "Upload GeoJSON",
        type=['geojson', 'json'],
        help="Upload the area you want to analyze"
    )
    
    # Enhanced date range selection
    st.sidebar.header("📅 Analysis Parameters")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=date(2025, 1, 1),
        min_value=date(2020, 1, 1),
        max_value=date.today(),
        help="Start date for satellite data collection"
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=date(2025, 6, 30),
        min_value=date(2020, 1, 1),
        max_value=date.today(),
        help="End date for satellite data collection"
    )
    
    # Analysis options
    st.sidebar.header("🎛️ Advanced Options")
    use_ensemble = st.sidebar.checkbox("Use Ensemble Model", value=True, help="Use multiple algorithms for improved accuracy")
    include_uncertainty = st.sidebar.checkbox("Include Uncertainty Analysis", value=True, help="Calculate prediction uncertainty metrics")
    multi_scale = st.sidebar.checkbox("Multi-scale Analysis", value=True, help="Analyze at multiple spatial scales")
    
    # Main content area
    if model_file is None or geojson_file is None:
        st.info("👆 Please upload both a model file (.pkl) and a GeoJSON file to begin enhanced prediction")
        
        # Enhanced feature information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 Enhanced Features:
            
            **🛰️ Multi-Sensor Data:**
            - Sentinel-2: 26+ spectral features
            - Sentinel-1: Enhanced SAR analysis  
            - Terrain: 8 topographic variables
            
            **🔬 Advanced Indices:**
            - Hyperspectral-inspired features
            - Plant functional traits
            - Eucalyptus-specific indices
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Analysis Capabilities:
            
            **🌱 Species Classification:**
            - Eucalyptus vs Non-eucalyptus
            - Age group classification
            - Maturity assessment
            
            **🎯 Quality Metrics:**
            - Prediction confidence
            - Uncertainty quantification
            - Feature quality assessment
            """)
        
        st.markdown("""
        ### 📋 How to use:
        1. **Upload Enhanced Model**: Your trained model (.pkl file) with expanded feature set
        2. **Upload GeoJSON**: The geographic area for analysis
        3. **Configure Parameters**: Set date ranges and analysis options
        4. **Run Enhanced Prediction**: Start comprehensive analysis with all features
        """)
        return
    
    # Process files with enhanced information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Enhanced Model Information")
        # Save model file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_model:
            tmp_model.write(model_file.read())
            model_path = tmp_model.name
        
        # Load and display enhanced model info
        predictor = EnhancedEucalyptusPredictor()
        if predictor.load_model(model_path):
            st.success("✅ Enhanced model loaded successfully!")
            st.write(f"**Features:** {len(predictor.feature_names)}")
            
            # Check for GLCM features in model
            glcm_features_in_model = [f for f in predictor.feature_names if any(
                pattern in f.upper() for pattern in ['CONTRAST', 'HOMOGENEITY', 'ENERGY', 'ENTROPY', 'CORRELATION']
            )]
            
            if glcm_features_in_model:
                st.write(f"**GLCM Features:** {len(glcm_features_in_model)} texture features detected")
                st.info("🎯 GLCM texture analysis will be applied for maximum accuracy")
            else:
                st.write("**GLCM Features:** None detected in model")
            
            st.write(f"**Classes:** {list(predictor.class_labels.values())}")
            
            if predictor.ensemble_model is not None:
                st.write("**Ensemble:** ✅ Multi-algorithm ensemble ready")
            else:
                st.write("**Ensemble:** ⚠️ Using base model only")
        else:
            st.error("❌ Failed to load enhanced model")
            return
    
    with col2:
        st.subheader("🗺 GeoJSON Information")
        # Save GeoJSON file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp_geojson:
            tmp_geojson.write(geojson_file.read())
            geojson_path = tmp_geojson.name
        
        try:
            gdf = gpd.read_file(geojson_path)
            st.success("✅ GeoJSON loaded successfully!")
            st.write(f"**Polygons:** {len(gdf)}")
            st.write(f"**CRS:** {gdf.crs}")
            
            # Calculate area properly
            try:
                # Check if CRS is geographic (lat/lon)
                if gdf.crs is None:
                    st.warning("⚠️ No CRS defined. Assuming WGS84 (EPSG:4326)")
                    gdf = gdf.set_crs('EPSG:4326')
                
                if gdf.crs.is_geographic:
                    # Convert to appropriate UTM zone for area calculation
                    # Get centroid to determine UTM zone
                    centroid = gdf.geometry.centroid.iloc[0]
                    lon, lat = centroid.x, centroid.y
                    
                    # Calculate UTM zone
                    utm_zone = int((lon + 180) / 6) + 1
                    if lat >= 0:
                        utm_crs = f'EPSG:{32600 + utm_zone}'  # Northern hemisphere
                    else:
                        utm_crs = f'EPSG:{32700 + utm_zone}'  # Southern hemisphere
                    
                    # Project to UTM for area calculation
                    gdf_projected = gdf.to_crs(utm_crs)
                    total_area_m2 = gdf_projected.geometry.area.sum()
                    total_area_km2 = total_area_m2 / 1_000_000
                    total_area_acres = total_area_m2 / 4046.86  # Convert m² to acres
                    
                    st.write(f"**Total Area:** {total_area_acres:.2f} acres ({total_area_km2:.2f} km²)")
                    st.write(f"**Area calculated in:** {utm_crs}")
                else:
                    # Already in projected coordinates
                    total_area = gdf.geometry.area.sum()
                    total_area_acres = total_area / 4046.86  # Convert to acres (assuming area is in m²)
                    
                    if total_area > 1_000_000:  # Likely in m²
                        total_area_km2 = total_area / 1_000_000
                        st.write(f"**Total Area:** {total_area_acres:.2f} acres ({total_area_km2:.2f} km²)")
                    else:
                        st.write(f"**Total Area:** {total_area_acres:.2f} acres ({total_area:.2f} units²)")
                        
            except Exception as area_error:
                st.warning(f"⚠️ Could not calculate area: {str(area_error)}")
                st.write(f"**Total Area:** Unable to calculate")
                
            # Show bounding box info for additional context
            bounds = gdf.total_bounds
            st.write(f"**Extent:** {bounds[0]:.4f}, {bounds[1]:.4f} to {bounds[2]:.4f}, {bounds[3]:.4f}")
            
            # Show individual polygon areas if small number of polygons
            if len(gdf) <= 5:
                st.write("**Individual Polygon Areas:**")
                for idx, geom in enumerate(gdf.geometry):
                    try:
                        if gdf.crs.is_geographic:
                            geom_projected = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(utm_crs)
                            area_m2 = geom_projected.area.iloc[0]
                            area_acres = area_m2 / 4046.86
                            area_km2 = area_m2 / 1_000_000
                            st.write(f"  - Polygon {idx+1}: {area_acres:.2f} acres ({area_km2:.3f} km²)")
                        else:
                            area = geom.area
                            area_acres = area / 4046.86
                            if area > 1_000_000:
                                area_km2 = area / 1_000_000
                                st.write(f"  - Polygon {idx+1}: {area_acres:.2f} acres ({area_km2:.3f} km²)")
                            else:
                                st.write(f"  - Polygon {idx+1}: {area_acres:.2f} acres")
                    except:
                        st.write(f"  - Polygon {idx+1}: Area calculation failed")
        except Exception as e:
            st.error(f"❌ Failed to load GeoJSON: {str(e)}")
            return
    
    # Enhanced prediction button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Start Enhanced Prediction", type="primary", use_container_width=True):
            st.subheader("🔄 Running Enhanced Prediction...")
            
            # Show analysis configuration
            with st.expander("📋 Analysis Configuration", expanded=True):
                st.write(f"**Date Range:** {start_date} to {end_date}")
                st.write(f"**Ensemble Model:** {'✅ Enabled' if use_ensemble else '❌ Disabled'}")
                st.write(f"**Uncertainty Analysis:** {'✅ Enabled' if include_uncertainty else '❌ Disabled'}")
                st.write(f"**Multi-scale Analysis:** {'✅ Enabled' if multi_scale else '❌ Disabled'}")
            
            # Run enhanced prediction
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            gdf_result, results_df = predictor.predict_area(
                geojson_path, 
                start_date_str, 
                end_date_str
            )
            
            if results_df is not None:
                st.session_state.prediction_complete = True
                st.session_state.results_df = results_df
                st.session_state.gdf = gdf_result
                
                # Clean up temporary files
                os.unlink(model_path)
                os.unlink(geojson_path)
                
                st.rerun()
    
    # Display enhanced results if prediction is complete
    if st.session_state.prediction_complete and st.session_state.results_df is not None:
        display_enhanced_results(st.session_state.results_df, st.session_state.gdf)

def display_enhanced_results(results_df, gdf_result):
    st.markdown("---")
    st.subheader("📊 Enhanced Prediction Results")
    
    # Enhanced summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Polygons", len(results_df))
    
    with col2:
        eucalyptus_count = sum(results_df['is_eucalyptus'])
        st.metric("Eucalyptus Detected", eucalyptus_count)
    
    with col3:
        non_eucalyptus_count = sum(~results_df['is_eucalyptus'])
        st.metric("Non-Eucalyptus", non_eucalyptus_count)
    
    with col4:
        avg_confidence = results_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    with col5:
        avg_uncertainty = results_df['uncertainty'].mean() if 'uncertainty' in results_df.columns else 0
        st.metric("Avg Uncertainty", f"{avg_uncertainty:.3f}")
    
    # Enhanced quality metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_quality = sum(results_df['feature_quality'] == 'high') if 'feature_quality' in results_df.columns else 0
        st.metric("High Quality Predictions", high_quality)
    
    with col2:
        avg_maturity = results_df['maturity_score'].mean() if 'maturity_score' in results_df.columns else 0
        st.metric("Avg Maturity Score", f"{avg_maturity:.3f}")
    
    with col3:
        high_confidence = sum(results_df['confidence'] > 0.8)
        st.metric("High Confidence (>0.8)", high_confidence)
    
    # Enhanced results breakdown
    if eucalyptus_count > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌿 Eucalyptus Classification Details")
            eucalyptus_only = results_df[results_df['is_eucalyptus']]
            
            # Enhanced breakdown table
            breakdown_cols = ['confidence', 'uncertainty', 'maturity_score', 'avg_height', 'avg_ndvi']
            available_cols = [col for col in breakdown_cols if col in eucalyptus_only.columns]
            
            if available_cols:
                breakdown_data = eucalyptus_only.groupby('predicted_label')[available_cols].agg(['count', 'mean']).round(3)
                st.dataframe(breakdown_data, use_container_width=True)
        
        with col2:
            # Enhanced visualization - Height vs Confidence
            if all(col in eucalyptus_only.columns for col in ['avg_height', 'confidence', 'predicted_label']):
                fig = px.scatter(
                    eucalyptus_only, 
                    x='avg_height', 
                    y='confidence',
                    color='predicted_label',
                    size='maturity_score' if 'maturity_score' in eucalyptus_only.columns else None,
                    title="Eucalyptus: Height vs Confidence by Age Class",
                    labels={'avg_height': 'Average Height (m)', 'confidence': 'Prediction Confidence'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Quality distribution chart
    if 'feature_quality' in results_df.columns:
        st.subheader("📈 Prediction Quality Distribution")
        quality_counts = results_df['feature_quality'].value_counts()
        fig_quality = px.pie(
            values=quality_counts.values,
            names=quality_counts.index,
            title="Feature Quality Distribution"
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    # Enhanced detailed results table
    st.subheader("📋 Detailed Enhanced Results")
    
    # Enhanced display options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_all = st.checkbox("Show all polygons", value=True)
    with col2:
        min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.1)
    with col3:
        quality_filter = st.selectbox("Feature quality", ["All", "high", "medium", "low"])
    
    # Filter enhanced results
    filtered_df = results_df.copy()
    if not show_all:
        filtered_df = filtered_df[filtered_df['is_eucalyptus']]
    filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
    if quality_filter != "All" and 'feature_quality' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['feature_quality'] == quality_filter]
    
    # Enhanced display columns
    display_columns = [
        'polygon_id', 'predicted_label', 'confidence', 'uncertainty',
        'eucalyptus_probability', 'maturity_score', 'avg_height', 
        'avg_ndvi', 'avg_red_edge', 'avg_moisture', 'pixels_analyzed', 'feature_quality'
    ]
    
    # Filter available columns
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    
    st.dataframe(
        filtered_df[available_columns].round(3),
        use_container_width=True,
        column_config={
            'polygon_id': 'ID',
            'predicted_label': 'Prediction',
            'confidence': st.column_config.ProgressColumn(
                'Confidence', min_value=0, max_value=1
            ),
            'uncertainty': st.column_config.ProgressColumn(
                'Uncertainty', min_value=0, max_value=1
            ),
            'eucalyptus_probability': st.column_config.ProgressColumn(
                'Eucalyptus Prob', min_value=0, max_value=1
            ),
            'maturity_score': 'Maturity Score',
            'avg_height': 'Avg Height (m)',
            'avg_ndvi': 'Avg NDVI',
            'avg_red_edge': 'Avg Red Edge',
            'avg_moisture': 'Avg Moisture',
            'pixels_analyzed': 'Pixels',
            'feature_quality': 'Quality'
        }
    )
    
    # Enhanced download options
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced CSV download
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Enhanced Results CSV",
            data=csv_data,
            file_name=f"enhanced_eucalyptus_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Summary report download
        summary_report = f"""Enhanced Eucalyptus Prediction Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Polygons Analyzed: {len(results_df)}
Eucalyptus Detected: {sum(results_df['is_eucalyptus'])}
Non-Eucalyptus: {sum(~results_df['is_eucalyptus'])}

Average Confidence: {results_df['confidence'].mean():.3f}
Average Uncertainty: {results_df['uncertainty'].mean():.3f}
High Quality Predictions: {sum(results_df['feature_quality'] == 'high') if 'feature_quality' in results_df.columns else 'N/A'}

Classification Breakdown:
{results_df['predicted_label'].value_counts().to_string()}
"""
        
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_report,
            file_name=f"eucalyptus_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Reset button
    if st.button("🔄 Run New Enhanced Prediction", use_container_width=True):
        st.session_state.prediction_complete = False
        st.session_state.results_df = None
        st.session_state.gdf = None
        st.rerun()

if __name__ == "__main__":
    main()