import ee
import os
import pandas as pd
from datetime import datetime, timedelta
import argparse
import geemap
from tqdm import tqdm
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from multiprocessing import Pool
import warnings


COLLECTION_AND_BAND = {
    "s1": ["COPERNICUS/S1_GRD", ["VV", "VH"]],
    "s2": ["COPERNICUS/S2_SR_HARMONIZED", ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "MSK_CLDPRB", "QA60"]],
    "s2_toa": ["COPERNICUS/S2_HARMONIZED", ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]],
    "landsat8": ["LANDSAT/LC08/C02/T1_TOA", ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "QA_PIXEL"]],
    "landsat9": ["LANDSAT/LC09/C02/T1_TOA", ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "QA_PIXEL"]],
    "aster": ["ASTER/AST_L1T_003", ["B01", "B02", "B3N", "B04", "B05", "B06", "B07", "B08", "B09", "B10"]],
    "dw": [
        "GOOGLE/DYNAMICWORLD/V1",
        ["label"],
    ],
    "cld_shdw": ["PLACEHOLDER", ["placeholder"]],
}
METADATA_GROUP = ["s1", "s2", "s2_toa", "landsat8", "landsat9", "aster"]


def error_flagging(e, error_path):
    print(f"Error: {e} Error message saved to {error_path}.")
    with open(error_path, "a") as f:
        f.write(f"Error: {e}\n")


def convert_to_cog(input_filename, output_filename):
    """Convert GeoTIFF to Cloud Optimized GeoTIFF using GDAL with ZSTD compression."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cog_translate(source=input_filename, dst_path=output_filename, dst_kwargs=cog_profiles.get("zstd"), quiet=args.quiet)


def download_cog(img, image_path, scale, crs, region):
    """Download an image using geemap and convert to COG."""
    geemap.download_ee_image(img, filename=image_path, scale=scale, crs=crs, region=region.getInfo())
    convert_to_cog(image_path, image_path)


def download_metadata(image, crs, metadata_path, data_type):
    """Download metadata for the image."""
    if data_type == "s2":
        metadata = ee.Feature(
            None,
            {
                "capture_date": ee.Date(image.get("system:time_start")).format("Y-MM-dd HH:mm:ss"),
                # Capture date and time
                "resolution": args.res,  # Resolution
                "image_id": image.id(),  # Image ID
                "crs": crs,
                "dark_features_percentage": ee.Image(image).get("DARK_FEATURES_PERCENTAGE"),  # Cloud coverage
                "sun_elevation": image.get("MEAN_SOLAR_ZENITH_ANGLE"),  # Sun elevation angle
                "sun_azimuth": image.get("MEAN_SOLAR_AZIMUTH_ANGLE"),  # Sun azimuth angle
                "atmospheric_pressure": image.get("ATMOSPHERIC_PRESSURE"),  # Atmospheric pressure
                "water_percentage": image.get("WATER_PERCENTAGE"),  # Water vapor content
            },
        )
    elif data_type == "s2_toa":
        metadata = ee.Feature(
            None,
            {
                "capture_date": ee.Date(image.get("system:time_start")).format("Y-MM-dd HH:mm:ss"),
                "resolution": args.res,
                "image_id": image.id(),
                "crs": crs,
                "sun_elevation": image.get("MEAN_SOLAR_ZENITH_ANGLE"),
                "sun_azimuth": image.get("MEAN_SOLAR_AZIMUTH_ANGLE"),
                "orbit_direction": image.get("SENSING_ORBIT_DIRECTION"),
                "orbit_number": image.get("SENSING_ORBIT_NUMBER"),
            },
        )
    elif data_type in ["s1"]:
        metadata = ee.Feature(
            None,
            {
                "capture_date": ee.Date(image.get("system:time_start")).format("Y-MM-dd HH:mm:ss"),
                # Capture date and time
                "resolution": args.res,  # Resolution
                "image_id": image.id(),  # Image ID
                "crs": crs,
            },
        )
    elif data_type in ["landsat8", "landsat9"]:
        metadata = ee.Feature(
            None,
            {
                "capture_date": ee.Date(image.get("system:time_start")).format("Y-MM-dd HH:mm:ss"),
                "resolution": args.res,  # Resolution
                "image_id": image.id(),  # Image ID
                "crs": crs,
                "cloud_cover": image.get("CLOUD_COVER"),
                "collection_category": image.get("COLLECTION_CATEGORY"),
                "datum": image.get("DATUM"),
                "sun_azimuth": image.get("SUN_AZIMUTH"),
                "sun_elevation": image.get("SUN_ELEVATION"),
                "utm_zone": image.get("UTM_ZONE")
            }
        )
    else:
        raise NotImplementedError

    gdf = geemap.ee_to_gdf(ee.FeatureCollection(metadata))
    gdf.to_csv(metadata_path, index=False)


def add_cld_shdw_mask(img):
    SR_BAND_SCALE = 1e4
    CLD_PRJ_DIST = 1
    # Add cloud probability band
    cld_prb = ee.Image(img.get("s2cloudless")).select("probability").rename("s2_cld_prb_s2cloudless")
    cld_msk_30 = cld_prb.gt(30).focalMin(2).focalMax(4, iterations=2).reproject(**{'crs': img.select([0]).projection(), 'scale': 10}).rename("clouds_30")
    img = img.addBands(ee.Image([img, cld_prb, cld_msk_30]))
    # Add shadow band
    not_water = img.select("SCL").neq(6)
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    dark_pixels = img.select('B8').lt(0.30*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels').focalMin(2).focalMax(5)
    cld_proj = (img.select('clouds_30')
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 200})
        .select('distance')
        .mask()
        .rename('cloud_transform'))
    shadows_thres_30 = cld_proj.multiply(dark_pixels).rename('shadows_thres_30')

    dark_pixels = img.select('B8').lt(0.25*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels').focalMin(2).focalMax(5)
    cld_proj = (img.select('clouds_30')
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 200})
        .select('distance')
        .mask()
        .rename('cloud_transform'))
    shadows_thres_25 = cld_proj.multiply(dark_pixels).rename('shadows_thres_25')

    dark_pixels = img.select('B8').lt(0.20*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels').focalMin(2).focalMax(5)
    cld_proj = (img.select('clouds_30')
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 200})
        .select('distance')
        .mask()
        .rename('cloud_transform'))
    shadows_thres_20 = cld_proj.multiply(dark_pixels).rename('shadows_thres_20')

    return img.addBands(ee.Image([shadows_thres_20, shadows_thres_25, shadows_thres_30]))

def download_cloud_shadow(roi, start_date, end_date):
    s2_sr_collection = ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(roi).filterDate(start_date, end_date)
    s2_cloudless_collection = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(roi).filterDate(start_date, end_date)
    s2_sr_cloud_collection = ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            **{
                "primary": s2_sr_collection,
                "secondary": s2_cloudless_collection,
                "condition": ee.Filter.equals(**{"leftField": "system:index", "rightField": "system:index"}),
            }
        )
    )
    if s2_sr_cloud_collection.size().getInfo() == 0:
        return
    s2_sr_cloud_collection = s2_sr_cloud_collection.map(lambda img: img.clip(roi))
    s2_sr_cloud_collection = s2_sr_cloud_collection.map(add_cld_shdw_mask)
    img = s2_sr_cloud_collection.median().select(["s2_cld_prb_s2cloudless", "clouds_30", "shadows_thres_20", "shadows_thres_25", "shadows_thres_30"]).clip(roi)
    return img


def process_date(roi, roi_id, crs, date, collection_id, bands):
    """"""
    year = date.year
    month = date.month
    day = date.day

    month_stamp = f"{year}_{month}"
    os.makedirs(f"{args.root}/roi{roi_id}/{month_stamp}/{args.data_type}", exist_ok=True)

    img_name = f"roi{roi_id}_{args.data_type}_{year}_{month}_{day}"
    img_path = f"{args.root}/roi{roi_id}/{month_stamp}/{args.data_type}/{img_name}_median.tif"
    metadata_path = img_path.replace(".tif", "_metadata.csv")
    error_path = img_path.replace(".tif", ".txt")
    if args.resume and os.path.exists(img_path) and not os.path.exists(error_path):
        if args.data_type in METADATA_GROUP:
            if os.path.exists(metadata_path):
                return
        else:
            return

    start_date_str = date.strftime("%Y-%m-%d")
    end_date_str = (date + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        if args.data_type == "cld_shdw":
            img = download_cloud_shadow(roi, start_date_str, end_date_str)
            if img is None:
                return
        else:
            collection = ee.ImageCollection(collection_id).filterDate(start_date_str, end_date_str).filterBounds(roi)
            if collection.size().getInfo() == 0:
                return
            collection = collection.map(lambda img: img.clip(roi))
            if args.data_type in ["s2", "s2_toa", "dw", "s1", "landsat8", "landsat9"]:
                img = collection.median().select(bands).clip(roi)
            else:
                raise NotImplementedError
        download_cog(img, img_path, scale=args.res, crs=crs, region=roi)
        if args.data_type in METADATA_GROUP:
            probe = collection.first().select(bands[0])
            download_metadata(probe, crs, metadata_path, args.data_type)
        if os.path.exists(error_path):
            os.remove(error_path)
    except Exception as e:
        error_flagging(e, error_path)


def process_roi(roi_chunk):
    for roi_info in tqdm(roi_chunk):
        roi, roi_id, crs = roi_info
        collection_id, bands = COLLECTION_AND_BAND[args.data_type]
        for date in pd.date_range(start=datetime.strptime(args.start_date, "%Y-%m-%d"), end=datetime.strptime(args.end_date, "%Y-%m-%d"), freq="D"):
            process_date(roi, roi_id, crs, date, collection_id, bands)


def main_download_session(args):
    rois_df = pd.read_csv(args.rois).iloc[args.start_roi:args.end_roi]

    buffer_distance = args.radius * 1.2
    rois = [
        (
            ee.Geometry.Point([row['longitude'], row['latitude']]).buffer(buffer_distance).bounds(),
            row["index"],
            row['crs'],
        )
        for _, row in rois_df.iterrows()
    ]

    num_workers = args.workers
    chunk_size = len(rois) // num_workers + 1
    chunks = [rois[i : i + chunk_size] for i in range(0, len(rois), chunk_size)]

    print(f"Begin to download dataset...")
    with Pool(num_workers) as pool:
        results = [pool.apply_async(process_roi, (chunk,)) for chunk in chunks]
        list(tqdm((result.get() for result in results), total=len(results)))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download datasets from GEE")
    parser.add_argument("--ee-project-id", type=str, help="Google Earth Engine project id")
    parser.add_argument("--data-type", type=str, help="type of data to download", required=True, choices=COLLECTION_AND_BAND.keys())
    parser.add_argument("--rois", type=str, help="path to the csv file containing the ROIs", required=True)
    parser.add_argument("--radius", type=float, help="radius of the ROI in meter", default=1280)
    parser.add_argument(
        "--root", type=str, help="root directory to save the downloaded data", required=True
    )
    parser.add_argument("--start-date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--start-roi", type=int, default=0, help="Start ROI index of this session in the ROIs csv file")
    parser.add_argument("--end-roi", type=int, default=10, help="End ROI index of this session in the ROIs csv file")
    parser.add_argument("--crs", type=str, help="Coordinate reference system", default="EPSG:4326")
    parser.add_argument("--res", type=int, default=10, help="Resolution of the image to rescale to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers to use for downloading")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, help="Suppress processing step's status updates.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, help="Resume downloading from where it left off or start over")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    ee.Authenticate()
    ee.Initialize(project=args.ee_project_id)
    main_download_session(args)
