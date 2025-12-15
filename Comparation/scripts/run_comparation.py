#%%
import sys
from pathlib import Path
import numpy as np

# -----------------------------------
# Permitir importar desde /src
# -----------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# -----------------------------------
# Imports del paquete CODIGO
# -----------------------------------
from comparation.types import SignalData
from comparation.runner.run_all import run_all_indicators, list_indicators, run_selected_indicators
from comparation.plotting.rms_cv_plots import plots_rms_cv
from comparation.plotting.plots_sst_svd import plots_sst_svd
from comparation.plotting.maxent_sprt_plots import plots_maxent

from typing import Sequence, Tuple, Dict, Optional, Callable, Mapping, List, Any, Union
import os
import sys
import h5py

import matplotlib.pyplot as plt

class HDF5Reader:
    # Cache for all discovered paths
    _all_paths_cache: Optional[List[str]]
    def __init__(self, filepath: str):
        """
        Initializes the reader and loads the entire structure into memory.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        self.filepath = filepath
        self.data = self._read_file()
        # Index of all paths for quick search
        self._all_paths_cache = None

    def _read_file(self) -> Dict[str, Any]:
        """
        Reads the entire HDF5 file and converts it into a complete dictionary.
        """
        with h5py.File(self.filepath, "r") as hdf_file:
            return self._read_group(hdf_file)

    def _read_group(self, group: h5py.Group) -> Union[Dict[str, Any], List[Any], Any]:
        """
        Reads a group or dataset and converts it into a Python data structure.
        """
        data = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data[key] = self._read_dataset(item)
            elif isinstance(item, h5py.Group):
                data[key] = self._read_group(item)
        return data

    def _read_dataset(self, dataset: h5py.Dataset) -> Union[List[Any], Any]:
        """
        Converts a dataset into a native Python type.
        """
        data = dataset[()]
    
        # Si es un array de numpy, devolver SIEMPRE un numpy.ndarray
        if isinstance(data, np.ndarray):
            # Caso: array de bytes (cadenas en formato bytes) -> decodificar a str
            if data.dtype.kind in {'S', 'a'}:  # Byte strings
                # Decodificación elemento a elemento, preservando la forma
                decode = np.vectorize(lambda x: x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else x,
                                      otypes=[object])
                decoded = decode(data)
                # Intentar convertir a dtype de cadenas nativas si es rectangular
                try:
                    return decoded.astype(str)
                except Exception:
                    return decoded  # Mantener dtype=object si es irregular

            # Caso: array de objetos (posibles listas por fila, bytes anidados, etc.)
            if data.dtype == object:
                def _decode_obj(x: Any) -> Any:
                    if isinstance(x, (bytes, np.bytes_)):
                        return x.decode('utf-8')
                    if isinstance(x, np.ndarray):
                        # Decodificar recursivamente arrays anidados
                        return np.array([_decode_obj(y) for y in x.tolist()], dtype=object)
                    if isinstance(x, list):
                        return [_decode_obj(y) for y in x]
                    return x

                decoded_list = [_decode_obj(x) for x in data.tolist()]
                # Devolver como array; si las filas son listas de distinta longitud, será dtype=object
                try:
                    return np.array(decoded_list)
                except Exception:
                    return np.array(decoded_list, dtype=object)

            # Para arrays numéricos u otros tipos estándar, devolver tal cual
            return data

        # Si es un byte string, decodificarlo
        if isinstance(data, (bytes, np.bytes_)):
            return data.decode('utf-8')
        
        return data

    def get_data(self) -> Dict[str, Any]:
        """
        Returns the complete dictionary.
        """
        return self.data

    def get_element(self, *keys: str) -> Any:
        """
        Access a specific element using hierarchical keys.

        Usage examples:
        - get_element('group', 'subgroup', 'dataset')
        - get_element('group/subgroup/dataset')
        - get_element('dataset', '0')  # index into list/np.ndarray
        - get_element('dataset', '0:10')  # slice
        - get_element('dataset', '1,2')  # multi-dim index for numpy arrays
        """

        def _parse_slice(token: str):
            # Supports 'start:end[:step]' with empty parts allowed (e.g., ':10', '5:')
            parts = token.split(':')
            if not 1 <= len(parts) <= 3:
                return None
            def _to_int(x):
                return int(x) if x != '' else None
            try:
                start = _to_int(parts[0]) if len(parts) >= 1 else None
                stop = _to_int(parts[1]) if len(parts) >= 2 else None
                step = _to_int(parts[2]) if len(parts) == 3 else None
            except ValueError:
                return None
            return slice(start, stop, step)

        def _parse_index(token: str):
            # Helper: split commas but ignore those inside brackets [...]
            def _split_top_level_commas(s: str) -> List[str]:
                parts = []
                buf = []
                depth = 0
                for ch in s:
                    if ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth = max(0, depth - 1)
                    if ch == ',' and depth == 0:
                        parts.append(''.join(buf).strip())
                        buf = []
                    else:
                        buf.append(ch)
                if buf:
                    parts.append(''.join(buf).strip())
                return parts

            # Helper: parse list token like '[0,2,4]'
            def _parse_list_token(t: str):
                if t.startswith('[') and t.endswith(']'):
                    inner = t[1:-1].strip()
                    if inner == '':
                        return []
                    try:
                        return [int(x.strip()) for x in inner.split(',')]
                    except ValueError:
                        return None
                return None

            # Multi-dim index like 'i,j' or with slices '1:5, :' or lists ':[0,2,4]'
            if ',' in token:
                idx_tokens = _split_top_level_commas(token)
                idx = []
                for t in idx_tokens:
                    lst = _parse_list_token(t)
                    if lst is not None:
                        idx.append(lst)
                        continue
                    s = _parse_slice(t) if ':' in t else None
                    if s is not None:
                        idx.append(s)
                    else:
                        try:
                            idx.append(int(t))
                        except ValueError:
                            return None
                return tuple(idx)

            # Single-dim: slice or int
            if ':' in token:
                s = _parse_slice(token)
                return s
            # Single-dim: list-of-indices '[0,2,4]'
            lst = _parse_list_token(token)
            if lst is not None:
                return lst
            try:
                return int(token)
            except ValueError:
                return None

        # If a single path was provided, split it by '/'
        auto_search = False
        if len(keys) == 1 and isinstance(keys[0], str):
            if '/' in keys[0]:
                path_parts = [p for p in keys[0].split('/') if p != '']
            else:
                # Single token; we may need to auto-search nested keys if not present at root
                path_parts = [keys[0]]
                auto_search = True
        else:
            path_parts = list(keys)

        current: Any = self.data
        for idx, key in enumerate(path_parts):
            # Navigate dictionaries (HDF5 groups)
            if isinstance(current, dict):
                if key in current:
                    current = current[key]
                    continue
                else:
                    # If we are at root, try nested resolution for two cases:
                    # 1) Single-token auto_search (handled as before)
                    # 2) Multi-part path starting with a nested key (e.g., 'tool_dyn/subkey')
                    if current is self.data:
                        # Case 1: single token search
                        if auto_search and len(path_parts) == 1:
                            found = self.find_first(key)
                            if found is None:
                                raise KeyError(f"Key not found in group (and no nested match): {key}")
                            return self.get_element(found)

                        # Case 2: multi-part path; try to resolve base segment against all matches
                        if len(path_parts) > 1:
                            remaining = path_parts[idx+1:]
                            # Candidates whose last segment equals the missing key
                            candidates = self.find_all(key)
                            for base in candidates:
                                composed = base + (('/' + '/'.join(remaining)) if remaining else '')
                                try:
                                    return self.get_element(composed)
                                except KeyError:
                                    continue
                    # If not resolved, raise
                    raise KeyError(f"Key not found in group: {key}")

            # Index lists or tuples
            if isinstance(current, (list, tuple)):
                idx = _parse_index(key)
                if idx is None:
                    raise KeyError(f"Invalid list/tuple index: {key}")
                try:
                    if isinstance(idx, list):
                        # Manual advanced indexing for Python lists
                        current = [current[i] for i in idx]
                    else:
                        current = current[idx]
                except Exception as e:
                    raise KeyError(f"Index error for '{key}': {e}")
                continue

            # Index numpy arrays
            if isinstance(current, np.ndarray):
                idx = _parse_index(key)
                if idx is None:
                    raise KeyError(f"Invalid numpy index: {key}")
                try:
                    current = current[idx]
                except Exception as e:
                    raise KeyError(f"Index error for '{key}': {e}")
                continue

            # Unsupported type for further navigation
            raise KeyError(f"Cannot navigate into type {type(current).__name__} with key '{key}'")

        return current

    def list_paths(self) -> List[str]:
        """
        List all dataset paths available in the loaded HDF5 structure, using '/' as separator.
        """
        if self._all_paths_cache is not None:
            return self._all_paths_cache

        paths: List[str] = []

        def _collect(node: Any, prefix: str = ""):
            if isinstance(node, dict):
                # Include group path itself so mid-path keys can be found
                if prefix:
                    paths.append(prefix)
                if not node:
                    return
                for k, v in node.items():
                    new_prefix = f"{prefix}/{k}" if prefix else k
                    _collect(v, new_prefix)
            else:
                if prefix:
                    paths.append(prefix)

        _collect(self.data)
        self._all_paths_cache = paths
        return paths

    def find_all(self, key: str) -> List[str]:
        """
        Find all full paths whose last segment equals the provided key.
        Example: find_all('tool_dyn') -> ['group1/tool_dyn', 'group2/sub/tool_dyn']
        """
        matches = []
        for p in self.list_paths():
            last = p.split('/')[-1]
            if last == key:
                matches.append(p)
        return matches

    def find_first(self, key: str) -> Optional[str]:
        """
        Return the first matching path for the given key, or None if not found.
        """
        matches = self.find_all(key)
        return matches[0] if matches else None
    
def _cut_signal( t,x , time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cuts the signal to the specified time range.
    """
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]

#%%
# def main() -> None:
dir_cono =  r'D:\Thesis\03-Code_Storage\02-Altintlas_Nessy2m_Storage\2DOF_Cono\1DOF_150Hz'
dir_path_use = dir_cono

data_dir = os.path.abspath(os.path.join(dir_path_use, 'out.hdf5' ))
data = HDF5Reader(data_dir)

tool_dyn = data.get_element('tool_dyn/data',)
t = tool_dyn[:,0]
tool_dyn = tool_dyn[:,1]
tool_dyn_vel = data.get_element('tool_dyn_o/data',)[:,1]
force_N = data.get_element('res_R_p/data',) #Newtons

t = t
v = tool_dyn_vel
fs = 1.0 / (t[1]-t[0])

t_cut, v_cut = _cut_signal( t, v , (0.05, 5.365770208787228) )



sig = SignalData(
    t=t_cut,
    x=v_cut,
    t_original=t,
    x_original=v,
    path=data_dir,
    fs=fs,
    meta={"AP": "5mm-15mm",
            "RPM": 12_000,}
)

print("Indicadores registrados:", list_indicators())

# ===========================
# Ejecutar todos los indicadores
# ===========================
# results = run_all_indicators(sig)
results = run_selected_indicators(sig, [ 'SST_SVD', 'RMS_CV', 'MaxEnt_SPRT'] )
# results = run_selected_indicators(sig, [ 'MaxEnt_SPRT'] )


#%%
# rms_cv_res = next(r for r in results if r.name == "RMS_CV")
zoom_x = (4,11)
# zoom_x = None
zoom_y = None
vlines = [5.365770208787228, 7.947208594272872]  # tiempos de eventos importantes
# hlines = [0.1, 0.2]  # niveles de CV relevantes

# plots_rms_cv(signal=sig, result=rms_cv_res, show_signal=True, 
#              zoom_x=zoom_x, zoom_y=zoom_y, vlines=vlines, hlines=None,)

#%%
if results.name_list().count("SST_SVD") > 0:
    sst_svd_res = next(r for r in results if r.name == "SST_SVD")
    zoom_x = None
    vlines = [5.3, 7.9]  # tiempos de eventos importantes
    plots_sst_svd(signal=sig, result=sst_svd_res, show_signal=True, 
                zoom_x=zoom_x, zoom_y=zoom_y, vlines=None, hlines=None,)

#%%
if results.name_list().count("MaxEnt_SPRT") > 0:
    maxent_sprt_res = next(r for r in results if r.name == "MaxEnt_SPRT")
    zoom_x = None
    vlines = [5.3]  # tiempos de eventos importantes
    plots_maxent(signal=sig, result=maxent_sprt_res, show_signal=True, 
                zoom_x=zoom_x, zoom_y=zoom_y, vlines=None, hlines=None,)
    
if results.name_list().count("RMS_CV") > 0:
    rms_cv_res = next(r for r in results if r.name == "RMS_CV")
    zoom_x = None
    vlines = [5.3]  # tiempos de eventos importantes
    plots_rms_cv(signal=sig, result=rms_cv_res, show_signal=True, 
                zoom_x=zoom_x, zoom_y=zoom_y, vlines=vlines, hlines=None,)



print(f"Ejecutados {len(results)} indicadores.\n")

for r in results:
    print("-------------------------------------")
    print("ID:", r.name)
    print("t size:", len(r.t))
    print("I_t size:", len(r.I_t))
    print("t_d:", r.t_d)
    # print("meta keys:", list(r.meta.keys()))
    print("-------------------------------------\n")

print("\n✅ Runner funcionando correctamente.")


# if __name__ == "__main__":
#     main()
