import os, tarfile, requests, numpy as np
from pathlib import Path

ZONES = ["alu","cam","cas","izu","ker","kur","phi","ryu","sam","sco","sol","sum","van"]

_SCIENCEBASE_URL = ("https://www.sciencebase.gov/catalog/file/get/"
    "5aa1b00ee4b0b1c392e86467"
    "?f=__disk__d5%2F91%2F39%2Fd591399bf4f249ab49ffec8a366e5070fe96e0ba")
_TAR_FILENAME = "Slab2Distribute_Mar2018.tar.gz"

def _tar_path(d): return Path(d) / _TAR_FILENAME

def download_slab2_archive(data_dir="data/slab2", force=False):
    d = Path(data_dir); d.mkdir(parents=True, exist_ok=True)
    dest = _tar_path(d)
    if dest.exists() and not force: return dest
    print("Downloading Slab2 archive (~134 MB) …", flush=True)
    r = requests.get(_SCIENCEBASE_URL, stream=True, timeout=300)
    r.raise_for_status()
    n = 0
    with open(dest,"wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk); n += len(chunk)
            print(f"\r  {n/1048576:.1f} MB", end="", flush=True)
    print(f"\n  Saved to {dest}"); return dest

def extract_depth_grids(data_dir="data/slab2", force=False):
    d = Path(data_dir); tar = _tar_path(d)
    if not tar.exists(): raise FileNotFoundError(f"Not found: {tar}")
    found = {}
    print("Extracting depth grids …")
    with tarfile.open(tar,"r:gz") as tf:
        for m in tf.getmembers():
            fname = os.path.basename(m.name)
            if "_slab2_dep_" in fname and fname.endswith(".grd"):
                zone = fname.split("_slab2_")[0]
                if zone not in ZONES: continue
                dest = d / fname
                if dest.exists() and not force: found[zone]=dest; continue
                fobj = tf.extractfile(m)
                if fobj: dest.write_bytes(fobj.read()); found[zone]=dest; print(f"  {zone}")
    print(f"Done. {len(found)}/{len(ZONES)} zones."); return found

def download_all_zones(data_dir="data/slab2", force=False):
    download_slab2_archive(data_dir, force=force)
    return extract_depth_grids(data_dir, force=force)

def load_zone_grid(path):
    import netCDF4, numpy as np
    with netCDF4.Dataset(str(path),"r") as ds:
        xk = next((k for k in ("x","lon","longitude") if k in ds.variables),None)
        yk = next((k for k in ("y","lat","latitude")  if k in ds.variables),None)
        zk = next((k for k in ("z","dep","depth")     if k in ds.variables),None)
        if not all([xk,yk,zk]):
            raise ValueError(f"Unknown vars in {path}: {list(ds.variables)}")
        lons  = np.array(ds.variables[xk][:], dtype=np.float64)
        lats  = np.array(ds.variables[yk][:], dtype=np.float64)
        dep2d = np.ma.filled(ds.variables[zk][:], np.nan).astype(np.float64)
    dep2d = np.where(np.abs(dep2d)>1e6, np.nan, -dep2d)
    if dep2d.ndim==3: dep2d=dep2d[0]
    lon2d,lat2d = np.meshgrid(lons,lats)
    spacing = float(lons[1]-lons[0]) if len(lons)>1 else 0.05
    return dict(lon2d=lon2d,lat2d=lat2d,dep2d=dep2d,lons=lons,lats=lats,spacing=spacing)

def load_all_zones(data_dir="data/slab2"):
    d = Path(data_dir); grids = {}
    for zone in ZONES:
        m = list(d.glob(f"{zone}_slab2_dep_*.grd"))
        if not m: print(f"  WARNING: {zone} not found"); continue
        grids[zone] = load_zone_grid(m[0])
    return grids
