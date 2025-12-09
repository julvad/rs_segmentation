import arcpy
import os 
import numpy as np
import rasterio
import pyproj
from typing import Literal
from utils.utils_mgmt import check_missing_images, lowercase_tif
from datetime import datetime
import glob
from typing import Optional

##__________________________________________________
# FOR BACKGROUND TILE EXPORT ONLY
# Use a shp of one of your sample polygons, but make sure that it does not overlap with the background rectangles.
DUMMY_SHP_PATH = 'data/shp/dummy/dummy.shp' # change this with your path
assert os.path.isfile(DUMMY_SHP_PATH), f'A dummy shapefile path must be specified for background tile exports. See source code.'
##__________________________________________________

def export_training_samples(
        shp_folder_path:str,
        img_folder_path:str,
        out_folder_path:str, 
        tile_format:Literal['Classified_Tiles','RCNN_Masks'],
        tile_size:int=256,
        tile_overlap:int=0,
        rotation_angle:int=0,
        min_overlap_ratio:float=0,
        resample:bool=False,
        resample_pixel_size:int=10,
        is_background:bool=False
        ):
    """
    ArcPy code.
    Exports training samples in the ArcPy format.

    Inputs---
    shp_folder_path: A folder that contains folders of shapefiles. Ex.: <shp_folder_path>/<shp_folder_name1>/<shp_folder_name1>.shp, <shp_folder_name1>.shx, etc.
    img_folder_path: A folder that contains geotiff images. The names of the tiff must match the name of the shp. Ex.: <img_folder_path>/<shp_folder_name1>.tif, etc.
    out_folder_path: Path to an unexisting folder that will store the created samples.
    tile_format: ArcPy sample format argument. Ex.: Classified_Tiles, RCNN_Masks, etc. See the Export Training Samples tool docs for options.
    tile_size: The size in pixels for the square tiles that will be created.
    tile_overlap: Overlap between tiles when creating. 0=No overlap.
    rotation_angle: Creates rotated copies of samples. 180=2x the amount of samples. 90:4x the amount of samples.
    min_overlap_ratio: Removes tiles with a small portion of a polygon. Float between 0 and 1.
    resample: True/False. Resamples the pixel size of the geotiffs before export.
    resample_pixel_size: If resample is True. Output pixel size (in img CRS units).
    is_background: Use True if the shapefiles are empty background samples. NOTE: Must edit source code to specify a dummy shp path.
    """
    
    shp_list = glob.glob(os.path.join(shp_folder_path, '*')) # get shp folder paths
    c = 0 #counter
    len_shp = len(shp_list)
    
    if is_background:
        which_tiles = 'ALL_TILES'
    else:    
        which_tiles = 'ONLY_TILES_WITH_FEATURES'

    for shp_folder in shp_list:
        shp_name = os.path.basename(shp_folder)
        shp_filepath = os.path.join(shp_folder_path, f'{shp_name}.shp')
        assert os.path.isfile(shp_filepath), f'Error: shp not found at path {shp_filepath}'
        in_raster_path = None


        # define input raster path
        in_raster_path = os.path.join(img_folder_path, f'{shp_name}.tif')
        assert os.path.isfile(in_raster_path), f'Error: tif image not found at path {in_raster_path}'

        #define samples path
        out_path = os.path.join(out_folder_path, shp_name)
        
        if os.path.isdir(out_path):
            print(f'Skipping. {shp_name} samples already created.')
            continue

        assert os.path.isfile(in_raster_path), f'Error on image path {in_raster_path} for date {shp_name}'
        # create training samples for the tile size and the pixel size
        print(f'Creating training samples for image {shp_name} at tile_size={tile_size}')
        if resample:
            print(f'Samples will be resampled with pixel size= {resample_pixel_size}')

        #__________________________________________________________________________________________________________________
        #1. Fast clipping
        # get shapefile AOI 
        1 # try creating manual arcpy.Extent(minmaxcoords) if this does not work straightaway
        
        shp_extent = arcpy.Describe(shp_filepath).extent
        min_x = shp_extent.XMin
        min_y = shp_extent.YMin 
        max_x = shp_extent.XMax 
        max_y = shp_extent.YMax 

        # reproj aoi coordinates in local UTM CRS
        with rasterio.open(in_raster_path) as img:
            # get img crs
            raster_crs = img.crs
            # transform extent coordinates to img crs
            transformer = pyproj.Transformer.from_crs('EPSG:4326', raster_crs, always_xy=True)
            min_x, min_y = transformer.transform(min_x, min_y)
            max_x, max_y = transformer.transform(max_x, max_y)

        # If tile raster, give a 10 km buffer for clipping. ## Optional but speeds up samples creation
        if not is_background:
            # update shp_extent with buffer
            min_x = min_x - 10000 
            min_y = min_y - 10000
            max_x = max_x + 10000
            max_y = max_y + 10000
        
        # create the new arcpy Extent object
        shp_extent = arcpy.Extent(min_x, min_y, max_x, max_y)

        clipped = arcpy.ia.Clip(
            in_raster=in_raster_path,
            aoi=shp_extent 
        )

        #__________________________________________________________________________________________________________________
        #2. Fast resampling
        if resample: 
            resampled = arcpy.ia.Resample(
                raster=clipped,
                resampling_type='Bilinear',
                # input_cellsize=10, ## Might have to uncomment and fix this?
                output_cellsize=resample_pixel_size,
            )
        else:
            # no need to resample
            resampled = clipped

        if is_background: # If background tiles, only fix I found was to use a dummy shp so the tool knows the number of classes. The shp must not overlap with bg tiles.
            # update shp to dummy shp path
            shp_filepath = DUMMY_SHP_PATH
        #__________________________________________________________________________________________________________________
        #2. Create samples
        with arcpy.EnvManager(rasterStatistics='NONE', pyramid='NONE'):   
            arcpy.ia.ExportTrainingDataForDeepLearning(
                in_raster = resampled,
                out_folder = out_path,
                in_class_data = shp_filepath,
                image_chip_format = 'TIFF',
                tile_size_x = tile_size,
                tile_size_y = tile_size,
                stride_x = tile_overlap,
                stride_y = tile_overlap,
                output_nofeature_tiles = which_tiles,
                metadata_format = tile_format,
                start_index=0,
                class_value_field=None,
                buffer_radius=0,
                in_mask_polygons=None,
                rotation_angle=rotation_angle,
                reference_system = 'MAP_SPACE',
                processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
                blacken_around_feature="NO_BLACKEN",
                crop_mode="FIXED_SIZE",
                in_raster2=None,
                in_instance_data=None,
                instance_class_value_field=None,
                min_polygon_overlap_ratio=min_overlap_ratio
            )
        c+=1
        print(f'{datetime.now()}.    {c}/{len_shp}: Created samples at {out_path}')


def tile_raster2(
        in_raster, 
        out_folder_path,
        extent_4326:Optional[tuple]=None,
        tile_size:int=256, 
        overlap:int=128, 
        custom_file_name: Optional[str] = None,
        resample_pixel_size:Optional[int]=None
        ):
    """
    ArcPy code.
    Splits a raster into tiles. Can be used for predictions later.
    """
    
    out_folder_path = os.path.abspath(out_folder_path)

    filename = os.path.basename(in_raster)
    os.makedirs(out_folder_path, exist_ok=True)

    if custom_file_name is not None:
        out_filename = custom_file_name + '_'
    else:
        if filename.endswith('.tif'):
            filename = filename.strip('.tif')
        out_filename = filename + '_'


    if resample_pixel_size is not None:
        arcpy_raster = arcpy.ia.Resample(
            raster=in_raster,
            resampling_type='BilinearInterpolationPlus',
            input_cellsize=None,
            output_cellsize=resample_pixel_size
        )
    else:
        arcpy_raster = in_raster

    if extent_4326 is None:
        with arcpy.EnvManager(rasterStatistics="NONE", pyramid="NONE"):
            arcpy.management.SplitRaster(
                in_raster=arcpy_raster,
                out_folder=out_folder_path,
                out_base_name=out_filename,
                split_method="SIZE_OF_TILE",
                format='TIFF',
                resampling_type="BILINEAR",
                tile_size=f"{tile_size} {tile_size}",
                overlap=overlap,
                units="PIXELS",
                cell_size=None,
                origin=None,
                split_polygon_feature_class=None,
                clip_type="NONE",
                template_extent="DEFAULT",
            )
    else:
        # get img crs
        with rasterio.open(in_raster) as img:
            raster_crs = img.crs
        # transform extent coordinates to img crs
        transformer = pyproj.Transformer.from_crs('EPSG:4326', raster_crs, always_xy=True)
        min_x, min_y = transformer.transform(extent_4326[0], extent_4326[1])
        max_x, max_y = transformer.transform(extent_4326[2], extent_4326[3])

        with arcpy.EnvManager(rasterStatistics="NONE", pyramid="NONE"):
            arcpy.management.SplitRaster(
                in_raster=in_raster,
                out_folder=out_folder_path,
                out_base_name=out_filename,
                split_method="SIZE_OF_TILE",
                format="TIFF",
                resampling_type="BILINEAR",
                tile_size=f"{tile_size} {tile_size}",
                overlap=overlap,
                units="PIXELS",
                cell_size=None,
                origin=None,
                split_polygon_feature_class=None,
                clip_type="EXTENT",
                template_extent=f'{min_x} {min_y} {max_x} {max_y}'
            )
    print(f'{in_raster} has been tiled with size {tile_size} at {out_folder_path}')
    
    # remove extra metadata files
    tile_path_list = glob.glob(os.path.join(out_folder_path, f'{out_filename}*.TIF'))
    for tile_path in tile_path_list:
        ext_list = ['.TIF.aux.xml','.tif.ovr', '.tfw', '.TIF.vat.cpg', '.TIF.vat.dbf']
        remove_list = [tile_path.replace('.TIF', s) for s in ext_list]

        # check if its a border (empty) tile
        with rasterio.open(tile_path) as img:
            values = img.read(1)
            if np.all(values == 0):
                remove_list.append(tile_path)

        # delete all files not needed
        for remove_path in remove_list:
            try:
                os.remove(remove_path)
            except FileNotFoundError:
                pass
        
    lowercase_tif(out_folder_path)
