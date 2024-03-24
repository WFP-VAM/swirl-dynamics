import os
import glob
import xarray as xr
import tensorflow as tf


# from generator method
def load_nc_dir_with_generator(dir_: str, split: str):
    def gen():
        for file in glob.glob(os.path.join(dir_, f"{split}/*.zarr")):
            ds = xr.open_zarr(file)
            yield {key: tf.convert_to_tensor(val) for key, val in ds.items()}

    sample = next(iter(gen()))

    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            key: tf.TensorSpec(val.shape, dtype=val.dtype)
            for key, val in sample.items()
        }
    )


def preproc_ecmwf(example: dict[str, tf.Tensor]):
    processed = {}
    mean_chirps, var_chirps = tf.nn.moments(example["chirps"], axes=[0, 1, 2])
    processed["x"] = tf.cast((example["chirps"] - mean_chirps) / (var_chirps + 1e-4), tf.float32) 
    
    # The "channel:" prefix indicate that the conditioning signal is to be
    # incorporated by resizing and concatenating along the channel dimension.
    # This is implemented at the backbone level.
    mean_ecmwf, var_ecmwf = tf.nn.moments(example["ecmwf"], axes=[0, 1, 2])
    processed["cond"] = {"channel:low_res": tf.cast((example["ecmwf"]- mean_ecmwf) / (var_ecmwf + 1e-4), tf.float32)}
    return processed


def get_cond_ecmwf_dataset(split: str, batch_size: int):
    ds = load_nc_dir_with_generator("data/", split)
    ds = ds.map(preproc_ecmwf)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()
    return ds


def get_mean_std_data(split: str):
    ds = xr.open_mfdataset(f"data/{split}/*.zarr", engine='zarr')
    mean_chirps, mean_ecmwf = float(ds.mean().chirps.load().values), float(ds.mean().ecmwf.load().values)
    var_chirps, var_ecmwf = float(ds.std().chirps.load().values), float(ds.std().ecmwf.load().values)
    return mean_chirps, var_chirps, mean_ecmwf, var_ecmwf