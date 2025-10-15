import argparse
import math
import os
import random
import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

from astropy.coordinates import FK5, SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from functools import partial
from json import load,dump
from pathlib import Path
from tqdm import tqdm

np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--catalogue','-c',required=True,type=str, default=None)
parser.add_argument('--tile','-t',required=False,type=int, default=None)
parser.add_argument('--batchsize','-bs',type=int, default=128)
parser.add_argument('--processes','-p',type=int, default=8)
parser.add_argument('--batch_dir','-bd',type=str, default="batches")
parser.add_argument('--tile_dir','-td',type=str, default="tiles")

args = parser.parse_args()

print(args.batch_dir)
print(args.tile_dir)


os.makedirs(args.batch_dir, exist_ok=True)

cols_to_keep = ["object_id_euclid", 
                "tile_index_euclid", 
                "right_ascension_euclid", 
                "declination_euclid", 
                "flux_detection_total_euclid", 
                "fluxerr_detection_total_euclid",
                "segmentation_area"]

def to_mag(F):
    return -2.5*np.log10(F)+23.9

def get_tile_short_dict():
    paths = Path(args.tile_dir).rglob('*.fits')
    pathlist = [str(x) for x in paths]
    # print(pathlist)

    tile_short = {}
    path_len = len(args.tile_dir)
    for i in pathlist:
        
        short = i[path_len+30:path_len+39]
        tile_short[short] = i

    return tile_short

tile_short = get_tile_short_dict()

print(args)

with fits.open(args.catalogue) as tab:
    data = tab[1].data

data = Table(data)

print(f"\ndata loaded... {len(data)} rows\n")

def save_batches_from_tiles(cat_subset, split, chunk_id, total_chunks, batches_per_chunk):

    image_errors = {}
    missing_tile_ids = []

    global_batch_offset = chunk_id * batches_per_chunk

    batches_split = math.ceil(len(split)/args.batchsize)

    source_ids = {}
    if total_chunks > 1:
        b_idx = global_batch_offset
    else:
        b_idx = 0
    b = True
    batch = None
    source_ids[f"{b_idx}"] = []

    if args.tile is not None:
        with fits.open(tile_short[f"{args.tile}"]) as hdul:
            data_t = hdul[0].data
            header = hdul[0].header  

    for img_id in tqdm(split):

        r = cat_subset[cat_subset["object_id_euclid"] == img_id]

        if args.tile is None:

            if str(r["tile_index_euclid"][0]) not in list(tile_short.keys()):
                if not len(missing_tile_ids):
                    message = f'\nMissing tile {r["tile_index_euclid"][0]}, all missing tiles will be outputted in missing_tiles.txt after batching complete'
                    warnings.warn(message)
                
                missing_tile_ids.append(str(r["tile_index_euclid"][0]))
                continue


            with fits.open(tile_short[str(r["tile_index_euclid"][0])]) as hdul:
                data_t = hdul[0].data
                header = hdul[0].header

        ra, dec = (r["right_ascension_euclid"], r["declination_euclid"])

        position = SkyCoord(ra*u.deg,dec*u.deg, frame=FK5)
        size = 128*u.pix

        wcs = WCS(header)
        cutout = Cutout2D(data_t, position , size, wcs=wcs)

        cutout = cutout.data.astype(np.float32)

        if cutout.shape != (128,128):
            image_errors[str(r["object_id_euclid"].value[0])] = "cant create 128x128 cutout"
            continue

        if (cutout==0).sum() > 0:
            image_errors[str(r["object_id_euclid"].value[0])] = "0-value pixels in image"
            continue

        if b:
            b = False

            batch = np.zeros((128,128)).astype(np.float32)
            batch[:cutout.shape[0],:cutout.shape[1]] = cutout
            batch = batch[np.newaxis,:,:]
        else:
            t = np.zeros((128,128)).astype(np.float32)
            t[:cutout.shape[0],:cutout.shape[1]] = cutout                
            batch = np.vstack((batch, t[np.newaxis,:,:]))

        source_ids[f"{b_idx}"].append(str(r["object_id_euclid"].value[0]))

        if len(batch) == args.batchsize:
            with open(f'{args.batch_dir}/{b_idx}.npy', 'wb') as f:
                np.save(f, batch)

            if total_chunks > 1:
                with open(f'{args.batch_dir}/batch_sources_{chunk_id}.json', 'w') as f:
                    dump(source_ids, f, indent=2)
            else:
                with open(f'{args.batch_dir}/batch_sources_full.json', 'w') as f:
                    dump(source_ids, f, indent=2)

            b_idx += 1
            if b_idx == batches_split*(chunk_id+1):
                break
            b = True
            batch = None
            source_ids[f"{b_idx}"] = []

        else:
            continue

    with open(f'{args.batch_dir}/{b_idx}.npy', 'wb') as f:
        np.save(f, batch)
        # print(f"{chunk_id} saving {b_idx}")
    if total_chunks > 1:
        with open(f'{args.batch_dir}/batch_sources_{chunk_id}.json', 'w') as f:
            dump(source_ids, f, indent=2)
    else:
        with open(f'{args.batch_dir}/batch_sources_full.json', 'w') as f:
            dump(source_ids, f, indent=2)

    return sum([len(source_ids[k]) for k in source_ids]), image_errors, missing_tile_ids

def process_chunk(chunk_id, total_chunks, sources_id):

    total_rows = len(sources_id)
    rows_per_chunk = (total_rows // args.batchsize) // total_chunks * args.batchsize

    if total_chunks > 1:

        start_idx = chunk_id * rows_per_chunk

        if chunk_id == total_chunks - 1:
            end_idx = total_rows

        else:
            end_idx = start_idx + rows_per_chunk
            chunk_rows = end_idx - start_idx
            assert chunk_rows % args.batchsize == 0, f"Chunk {chunk_id} has {chunk_rows} rows, not divisible by batchsize {args.batchsize}"
        
        sources_id_ = sources_id[start_idx:end_idx].copy()

    else:
        sources_id_ = sources_id.copy()
    
    batches_per_chunk = rows_per_chunk // args.batchsize

    total_saved = save_batches_from_tiles(data, sources_id_, chunk_id, total_chunks,batches_per_chunk)

    return total_saved

if args.tile is not None:
    
    assert f"{args.tile}" in list(tile_short.keys()), f"{args.tile} not in {args.tile_dir}"

    sources_id = [x["object_id_euclid"] for x in data if x["tile_index_euclid"] == args.tile]

    assert len(sources_id), f"TILE {args.tile_dir} does not contain any sources from {args.catalogue}"

    print(f"TILE {args.tile} contains {len(sources_id)} sources from {args.catalogue}\n")

else:
    sources_id = [x for x in data["object_id_euclid"]]

random.shuffle(sources_id)

num_processes = args.processes

total_rows = len(sources_id)
rows_per_chunk = (total_rows // args.batchsize) // num_processes * args.batchsize


while rows_per_chunk < 1:
    num_processes -= 1
    rows_per_chunk = (total_rows // args.batchsize) // num_processes * args.batchsize


if num_processes != args.processes: print("num processes reduced to: ",num_processes)

with mp.Pool(processes=num_processes) as pool:

    process_func = partial(process_chunk,
                           total_chunks=num_processes,
                           sources_id=sources_id)
    
    chunk_ids = range(num_processes)

    try:
        all_results = pool.map(process_func, chunk_ids)
        total_processed = sum(results[0] for results in all_results)
        all_error_dicts = [results[1] for results in all_results]
        missing_tile_ids = [results[2] for results in all_results]
        missing_tile_ids = [y for x in missing_tile_ids for y in x]
        image_errors = {}
        for d in all_error_dicts:
            image_errors.update(d)

        print([results[0] for results in all_results])

        print("\nAll processes completed successfully\n")

        json_files_all = Path(args.batch_dir).rglob('*.json')
        json_files = [str(x) for x in json_files_all if "_full" not in str(x)]

        combined_dict = {}

        for i in json_files:
            temp_dict = {}
            with open(i,'r') as fp:
                temp_dict = load(fp)

            for j in list(temp_dict.keys()):

                assert j not in list(combined_dict.keys())

                new_list = []

                for k_idx, k in enumerate(temp_dict[j]):
                    new_list.append(str(k))

                combined_dict[j] = new_list


        with open(f'{args.batch_dir}/batch_sources_full.json','w') as fp:
            dump(combined_dict,fp, indent=2)

        full_json_amount = sum([len(combined_dict[k]) for k in combined_dict])

        assert full_json_amount == total_processed

        for f in json_files:
            os.remove(f)


    except KeyboardInterrupt:
        print("Interrupted by user")
        pool.terminate()
        pool.join()
    except Exception as e:
        print(f"Error occurred: {e}")
        pool.terminate()
        pool.join()

if len(missing_tile_ids):
    missing_tile_ids = list(set(missing_tile_ids))

    with open(f'missing_tiles.txt', 'w') as f:
        for line in missing_tile_ids:
            f.write(f"{line}\n")
tiles_message = f"All required tiles found!\n" if not len(missing_tile_ids) else f"Missing {len(missing_tile_ids)} tiles - See missing_tiles.txt for details.\n"
print(tiles_message)

missing = total_rows-total_processed
if missing>0:
    errors = []
    errors.append(f"ID, ERROR")

    for k,v in image_errors.items():
        errors.append(f"{k}, {v}")

    with open(f'image_errors.csv', 'w') as f:
        for line in errors:
            f.write(f"{line}\n")
image_message = f"All sources saved successfully!\n" if not len(list(image_errors.keys())) else f"{len(sources_id) - total_processed} unsaved sources. See image_errors.csv for details.\n"
print(image_message)
