import ee
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd

ee.Initialize()

# Define a GEE FeatureCollection for the 'Large Scale International Boundary Polygons' dataset
countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")

# Filter to get only North America countries (e.g., US, Canada, Mexico, etc.)
north_america = countries.filter(ee.Filter.inList('wld_rgn', ['North America']))

# Function to extract geometries and country names
def extract_geometry(feature):
    geometry = feature.geometry()
    return {
        'country_name': feature.get('country_na').getInfo(),
        'geometry': shape(geometry.getInfo())
    }

# Apply the function to extract the geometries
north_america_features = north_america.toList(north_america.size()).map(lambda f: ee.Feature(f).geometry()).getInfo()

# Extract geometries and country names into a GeoDataFrame
data = [extract_geometry(ee.Feature(f)) for f in north_america_features]
gdf = gpd.GeoDataFrame(pd.DataFrame(data), crs="EPSG:4326")

gdf.to_file("north_america_shapefile.shp")

print("Shapefile for North America exported successfully!")
