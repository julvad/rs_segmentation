import numpy as np
import random
import os
import glob
import geopandas as gpd
from PIL import Image
import rasterio
import shutil
from typing import Literal, Optional
from datetime import datetime

def resize_single_tile(
        in_raster_path, 
        out_raster_path, 
        method:Literal['nearest', 'bilinear'], 
        out_size:tuple=(256,256)
        ):
    
    img = Image.open(in_raster_path)
    assert out_raster_path.endswith('.tif')

    resample = {
        'nearest': Image.Resampling.NEAREST,
        'bilinear': Image.Resampling.BILINEAR
    }
    resized = img.resize(out_size, resample=resample[method])
    resized.save(out_raster_path)


def resize_tile_folder(
        tiles_folder_path:str, 
        out_folder_path:str, 
        method:Literal['nearest','bilinear'], 
        out_size:tuple=(256,256)
        ):
    
    tiles_folder_path = os.path.abspath(tiles_folder_path)
    out_folder_path = os.path.abspath(out_folder_path)
    assert os.path.isdir(tiles_folder_path), 'Error: invalid input folder path'
    os.makedirs(out_folder_path,exist_ok=True)
    resample = {
        'nearest': Image.Resampling.NEAREST,
        'bilinear': Image.Resampling.BILINEAR
    }
    
    img_path_list = glob.glob(os.path.join(tiles_folder_path, '*.tif'))
    assert len(img_path_list) != 0, f'Error: {tiles_folder_path} has no .tif images'
        
    to_resize = []
    for in_path in img_path_list:
        img = Image.open(in_path)
        if img.size != out_size:
            to_resize.append(in_path)

    if len(to_resize) == 0:
        print(f'All images are already of size {out_size} in {tiles_folder_path}')
    else:
        for in_path in to_resize:
            img = Image.open(in_path)
            resized = img.resize(out_size, resample=resample[method])
            out_path = os.path.join(out_folder_path, os.path.basename(in_path))
            resized.save(out_path)
        print('Resized images')

def parallel_unzip(in_folder_path:str):
    import zipfile
    from pathlib import Path
    import concurrent.futures
    from datetime import datetime
    import time

    counter = {
        'i': 0,
        'total': 0
    }

    def extract_here(zip_path):
        # safe_folder_path = zip_path.replace('.zip','.SAFE')

        zip_path = Path(zip_path)
        extract_to = zip_path.parent

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extracted : {zip_path.name}")
        except Exception as e:
            print(f"Failed to extract {zip_path.name}: {e}")

        counter['i'] += 1
        now = datetime.now()
        print(f"{now} — {counter['i']}/{counter['total']}")

    def unzip_files(folder_path, max_workers=20
                                , stagger_delay=0.5
                                ):
        folder = Path(folder_path)
        zip_files = [zip_file for zip_file in folder.glob("*.zip") if not (zip_file.with_suffix('.SAFE')).exists()]
        to_remove = [zip_file for zip_file in folder.glob("*.zip") if (zip_file.with_suffix('.SAFE')).exists()]
        counter['total'] = len(zip_files)

        for f in to_remove:
            os.remove(f)
            print(f'Deleted {f}')

        if not zip_files:
            print("No zip files found.")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for zip_file in zip_files:
                futures.append(executor.submit(extract_here, zip_file))
                time.sleep(stagger_delay)  # Stagger the thread starts to avoid burst I/O

    # 
    unzip_files(
        in_folder_path, 
        max_workers=20, 
        stagger_delay=0.4
        )
    

def prepare_training_folders(
        in_ts_folders:list, 
        dest_folder:str, 
        test_fraction:float, # 0 = no test fraction
        tile_size_tuple:tuple,
        bg_folder_paths:list=[],
        random_seed:int=24,
        check_dest_folder_valid:bool=True
        ):
    """
    Prepares a pytorch-ready samples folder from arcpy folders.

    ---Parameters---
    in_ts_folder: A ArcPy-resulting folder from the export samples script.
    dest_ts_folder: A pytorch ready folder, with /test and /train, and with /images and /labels.
    test_fraction: A random portion of the samples that will be split in a test folder. Applies evenly to background (empty) tiles.
    tile_size_tuple: A tuple, ex. (256,256), of the output tile size. Must be square.
    bg_folder_paths: A list of arcpy folder paths of background tiles. Blank labels will be created for pytorch.
    random_seed: If a test_fraction >0 is specified.
    check_dest_folder_valid: Simple validity check.
    """
    random.seed(random_seed)
    if os.path.isdir(dest_folder):
        raise FileExistsError(f'Error: {dest_folder} already exists.')
    
    # make images and labels folders
    if test_fraction ==0:  
        out_dirs = { # make no train/test_split
            'out_images_path': os.path.join(dest_folder,'images'),
            'out_labels_path': os.path.join(dest_folder,'labels')
        }
    else: # make train/test split
        out_dirs = {
            'out_train_images_path': os.path.join(dest_folder,'train','images'), 
            'out_train_labels_path': os.path.join(dest_folder,'train','labels'),
            'out_test_images_path': os.path.join(dest_folder,'test','images'),
            'out_test_labels_path': os.path.join(dest_folder,'test','labels')}
    for dir in out_dirs.items():
        os.makedirs(dir[1]) # create out dirs

    # empty samples lists
    list_imgs = []
    list_bgs = []

    for single_ts_folder in in_ts_folders:  
        image_folder_path = os.path.join(single_ts_folder, 'images')
        # copy images
        for file in os.listdir(image_folder_path):
            if file.endswith('.tif'):
                file_path = os.path.join(image_folder_path, file)
                list_imgs.append(file_path)
    
    # gather bg samples
    if len(bg_folder_paths) !=0:
        print(f'Including bg samples from {bg_folder_paths}')
        for single_bg_folder in bg_folder_paths:
            image_folder_path = os.path.join(single_bg_folder, 'images')
            # copy images
            for file in os.listdir(image_folder_path):
                if file.endswith('.tif'):
                    file_path = os.path.join(image_folder_path, file)
                    list_bgs.append(file_path)

    if test_fraction == 0:
        print('Not creating train/test split')
        for file_path in list_imgs + list_bgs:
            # define copy paths
            lbl_path = file_path.replace('images','labels')
            # get out name
            dirpath, filename = os.path.split(lbl_path)
            parent_dir = os.path.basename(os.path.dirname(dirpath))
            n = int(filename.strip('.tif'))
            out_name = f'{parent_dir}_tile_{n}.tif'

            dest_img_path = os.path.join(out_dirs['out_images_path'],out_name)
            dest_lbl_path =  dest_img_path.replace('images','labels')

            #check img size and copy
            with Image.open(file_path) as img:
                img_size = img.size
            if img_size == tile_size_tuple: # no need to resample
                shutil.copy(file_path, dest_img_path)
            else:
                resize_single_tile(
                    in_raster_path=file_path, 
                    out_raster_path=dest_img_path,
                    out_size=tile_size_tuple,
                    method='bilinear') #bilinear resampling for continuous value images
            print(f'{datetime.now()}: Copied {file_path} to {dest_img_path}')

            if not os.path.isfile(lbl_path):
                continue # bg images from arcpy have no labels. We will create the labels later
            #check lbl size and copy
            with Image.open(lbl_path) as img:
                img_size = img.size
            if img_size == tile_size_tuple: # no need to resample
                shutil.copy(lbl_path, dest_lbl_path)
            
            else:
                resize_single_tile(
                    in_raster_path=lbl_path, 
                    out_raster_path=dest_lbl_path,
                    out_size=tile_size_tuple,
                    method='nearest')# nearest neighbor resampling for labels
            print(f'{datetime.now()}: Copied {lbl_path} to {dest_lbl_path}')

    elif test_fraction >0:
        print('Creating train/test split')
        n_test_imgs = round(len(list_imgs)* test_fraction)
        n_test_bgs = round(len(list_bgs)* test_fraction)
        list_test_imgs = random.sample(list_imgs, n_test_imgs)
        list_test_bgs = random.sample(list_bgs, n_test_bgs)
        print(f'Keeping {n_test_imgs} test img samples out of {len(list_imgs)}')
        print(f'Keeping {n_test_bgs} test bg samples out of {len(list_bgs)}')
    
        set_test = set(list_test_imgs + list_test_bgs)
            
        for file_path in list_imgs + list_bgs:
            # define copy paths
            lbl_path = file_path.replace('images','labels')
            # get out path
            dirpath, filename = os.path.split(lbl_path)
            parent_dir = os.path.basename(os.path.dirname(dirpath))
            n = int(filename.strip('.tif'))
            out_name = f'{parent_dir}_tile_{n}.tif'

            if file_path in set_test:
                dest_img_path = os.path.join(out_dirs['out_test_images_path'],out_name)
                dest_lbl_path = dest_img_path.replace('images','labels')
            else:
                dest_img_path = os.path.join(out_dirs['out_train_images_path'],out_name)
                dest_lbl_path = dest_img_path.replace('images','labels')

            #check lbl size and copy
            with Image.open(file_path) as img:
                img_size = img.size

            if img_size == tile_size_tuple: # no need to resample
                shutil.copy(file_path, dest_img_path)
            else:
                resize_single_tile(
                    in_raster_path=file_path, 
                    out_raster_path=dest_img_path,
                    out_size=tile_size_tuple,
                    method='bilinear') #bilinear resampling for continuous value images
            print(f'{datetime.now()}: Copied {file_path} to {dest_img_path}')

            if not os.path.isfile(lbl_path):
                continue # bg images from arcpy have no labels. We will create the labels later

            #check lbl size and copy
            with Image.open(lbl_path) as img:
                img_size = img.size
            if img_size == tile_size_tuple: # no need to resample
                shutil.copy(lbl_path, dest_lbl_path)
            
            else:
                resize_single_tile(
                    in_raster_path=lbl_path, 
                    out_raster_path=dest_lbl_path,
                    out_size=tile_size_tuple,
                    method='nearest')# nearest neighbor resampling for labels
            print(f'{datetime.now()}: Copied {lbl_path} to {dest_lbl_path}')

    # create bg labels
    if len(bg_folder_paths) != 0:
        if test_fraction == 0:
            create_blank_labels(in_images_folder=out_dirs['out_images_path'],tile_size=tile_size_tuple[0])
        elif test_fraction >0:
            create_blank_labels(in_images_folder=out_dirs['out_train_images_path'],tile_size=tile_size_tuple[0])
            create_blank_labels(in_images_folder=out_dirs['out_test_images_path'],tile_size=tile_size_tuple[0])
        print(f'Create empty background labels')

    # validate out folders
    if check_dest_folder_valid:
        if test_fraction == 0:
            check_ts_folder(dest_folder,tile_size_tuple=tile_size_tuple)      
        elif test_fraction >0:
            check_ts_folder(os.path.join(dest_folder,'train'),tile_size_tuple=tile_size_tuple)
            check_ts_folder(os.path.join(dest_folder,'test'),tile_size_tuple=tile_size_tuple) 
    print(f'Prepared pytorch training folder at {dest_folder}')
        

def lowercase_tif(in_folder):
    img_folder = os.path.join(in_folder, 'images')
    lbl_folder = os.path.join(in_folder, 'labels')

    if os.path.isdir(img_folder):
    # Process image files
        listpaths = glob.glob(os.path.join(img_folder, '*.TIF'))
        for path in listpaths:
            if path.endswith('.TIF'):
                newpath = path.replace('.TIF', '.tif')
                os.rename(path, newpath)
                print(f'Renamed {path} to {newpath}')

    if os.path.isdir(lbl_folder):
    # Process label files
        listpaths = glob.glob(os.path.join(lbl_folder, '*.TIF'))
        for path in listpaths:
            if path.endswith('.TIF'):
                newpath = path.replace('.TIF', '.tif')
                os.rename(path, newpath)
                print(f'Renamed {path} to {newpath}')
    else:
        listpaths = glob.glob(os.path.join(in_folder, '*.TIF'))
        for path in listpaths:
            if path.endswith('.TIF'):
                newpath = path.replace('.TIF', '.tif')
                os.rename(path, newpath)
                print(f'Renamed {path} to {newpath}')


def create_shp_from_geojson(
        geojson_path, 
        out_folder_path,
        grouped_by_date:bool=False,
        split_by_zone:bool=False,
        specific_zones:list=[]
        ):
    
    # remove existing shp folder if existing
    if os.path.isdir(out_folder_path):
        print()

    # read geojson
    gdf = gpd.read_file(geojson_path)
    gdf.drop(columns={'OBJECTID', 'Shape_Length', 'Shape_Area', 'year', 'month', 'day', 'contains_ships_platforms'}, inplace=True, errors='ignore')
    gdf['date'] = gdf['date'].astype(str).str[:17]
    gdf.reset_index(inplace=True,drop=True)
    
    # NaN check
    for column in ['date', 'geometry']:
        assert not gdf[column].isna().any(), (f'error: NaN values in {column} column')
    last_date = None
    c=1 # counter for multiple shp with same date
    
    if grouped_by_date:
    #export as shp
        # Group gdf by image (date)
        grouped = gdf.groupby('date')

        # Iterate through image groups
        for date, group in grouped:
            # create a shp folder 
            date_folder = os.path.join(out_folder_path, date)
            if split_by_zone:
                zone = group['zone'].iloc[0] # gets the first 'zone' value of the group. Assumes groups have the same zone value.
                date_folder = os.path.join(out_folder_path, zone, date)
            os.makedirs(date_folder)
            
            # export group to shp path
            shp_path = os.path.join(date_folder, f'{date}.shp')
            group.to_file(shp_path)
            
            print(f'Shapefile {date} saved at {shp_path}')
    else: 
        # Export each row as a separate shapefile
        for n in range(len(gdf)):
            date = gdf.loc[n, 'date']
            if last_date == date:
                c+=1
            else:
                c=1
                
            date_folder = os.path.join(out_folder_path, date)
            if split_by_zone:
                zone = gdf.loc[n, 'zone']
                if specific_zones != []:
                    if zone not in specific_zones:
                        continue
                date_folder = os.path.join(out_folder_path, zone, f'{date}_bg{c}')
            os.makedirs(date_folder, exist_ok=True)
            shp_path = os.path.join(date_folder, f'{date}_bg{c}.shp')

            row_gdf = gdf.iloc[n:n+1] # gets a unique gdf row for export as shp (single polygon)
            row_gdf.to_file(shp_path)
            last_date = date
            print(f'Shapefile {date} saved at {shp_path}')


def check_missing_images(
        shp_folder_or_geojson:str,
        img_folder_path:str,
        image_type:Optional[Literal ['processed', 'safe_folders']] = 'processed'
        ):
    
    if image_type == 'processed':
        img_list = set(x.strip('.tif') for x in os.listdir(img_folder_path) if x.endswith('.tif'))
    elif image_type == 'safe_folders':
        img_list = set(os.listdir(img_folder_path))


    missing_list = []
    c = 0 # counter

    if os.path.isdir(shp_folder_or_geojson):
        for name in os.listdir(shp_folder_or_geojson):
            name = name[:17] # get only YYYY_MM_DD_HHHH
            if name not in img_list:        
                print(f'Image {name} not found')
                missing_list.append(name)
                c+=1 
    elif os.path.isfile(shp_folder_or_geojson) and shp_folder_or_geojson.endswith('.geojson'):
        gdf = gpd.read_file(shp_folder_or_geojson)
        for name in gdf['date']:
            name = name[:17]
            if name not in img_list:        
                print(f'Image {name} not found')
                missing_list.append(name)
                c+=1
    else:
        print(f'error with input shp_folder or geojson: {shp_folder_or_geojson}') 
    print(f'Checked missing images: {c} images missing') # Should print 0
    return missing_list

def rename_safe_to_date(main_folder_path):
    for dir in os.listdir(main_folder_path):
        if dir.startswith('S1') and dir.endswith('.SAFE'):
            dir_path = os.path.join(main_folder_path, dir)
            # find date
            y = dir[17:21]
            m = dir[21:23]
            d = dir[23:25] 
            t = dir[26:32]
            date = f'{y}_{m}_{d}_{t}'
            #rename folder
            new_path = os.path.join(main_folder_path,date)
            os.rename(dir_path, new_path)
            print(f'renamed {dir} to {date}')


def create_blank_labels(in_images_folder, tile_size: int = 256):

    # Get all image paths
    img_paths = glob.glob(os.path.join(in_images_folder, '*.tif'))
    if not img_paths:
        print(f'Warning: no .tif images found in {in_images_folder}')

    # Create labels folder
    labels_folder = in_images_folder.replace('images', 'labels')
    os.makedirs(labels_folder, exist_ok=True)

    # builds a set for fast check
    existing_labels = set(os.listdir(labels_folder))

    for img_path in img_paths:
        lbl_path = img_path.replace('images','labels')
        lbl_name = os.path.basename(lbl_path)

        if lbl_name in existing_labels:
            print(f'Skipping label, {lbl_path} already exists')
            continue

        # If no existing label, create blank label
        # copy georeferencing info from the input tile
        with rasterio.open(img_path) as src:
            profile = src.profile.copy()
            if (src.width != tile_size) or (src.height != tile_size):
                print(f'Warning: img {img_path} is not of square shape')

        # Create blank raster
        label_data = np.zeros((tile_size, tile_size), dtype=np.int16)

        # Add lzw compression
        profile.update(
            dtype='int16',
            count=1,
            compress='lzw',
            nodata=None, # no nodata value for labels
            width=tile_size,
            height=tile_size
        )

        # Save label geotiff
        with rasterio.open(lbl_path, 'w', **profile) as dst:
            dst.write(label_data, 1)
        print(f'Create blank label at {lbl_path}')

def check_ts_folder(in_ts_folder:str,tile_size_tuple:tuple=(256,256)):
    img_folder = os.path.join(in_ts_folder, 'images')
    lbl_folder = os.path.join(in_ts_folder, 'labels')
    assert os.path.isdir(img_folder), 'Error: images folder not found'
    assert os.path.isdir(lbl_folder), 'Error: labels folder not found'
    
    wrongsize_list = []
    nolabel_list = []
    wrongdtype_list = []

    img_list = glob.glob(os.path.join(img_folder, '*.tif'))
    for img_path in img_list:
        with Image.open(img_path) as img:
            if img.size != tile_size_tuple:
                wrongsize_list.append(img_path)
            if img.mode not in ['I','I;16','F']:
                wrongdtype_list.append(img_path)
        lbl_path = img_path.replace('images', 'labels')
        if not os.path.isfile(lbl_path):
            nolabel_list.append(img_path)
        else:
            with Image.open(lbl_path) as img:
                if img.size != tile_size_tuple:
                    wrongsize_list.append(img_path)
                if img.mode not in ['I','I;16']:
                    wrongdtype_list.append(lbl_path)
    
    if len(wrongdtype_list) + len(nolabel_list) + len(wrongsize_list) == 0:
        print(f'{in_ts_folder} checked. All good!')
    else:
        print(f'{len(wrongsize_list)} images not of shape {tile_size_tuple}:', wrongsize_list)
        print(f'{len(nolabel_list)} images without corresponding label:', nolabel_list)
        print(f'{len(wrongdtype_list)} images/labels in a wrong dtype format:', wrongdtype_list)


