# Benchmark mesh cache

Benchmark meshes are published separately in the
[`dmishura/OpenFOAM-Benchmark-Meshes`](https://github.com/dmishura/OpenFOAM-Benchmark-Meshes)
repository, release `dataset-v1`. This directory is the local archive cache;
mesh archives are intentionally not tracked by Git.

Fetch a single archive with:

```bash
python3 meshes/fetch_mesh.py MTB_example
python3 meshes/fetch_mesh.py MTBHPC_small original
python3 meshes/fetch_mesh.py MTBHPC_small renumbered
```

Fetch the complete dataset explicitly with:

```bash
python3 meshes/fetch_mesh.py --all
```

The fetcher obtains the authoritative catalog from the external repository,
caches it locally, and verifies both archive size and SHA-256. Downloads use a
temporary file that is renamed only after verification. A valid cached archive
is never downloaded again. The cached catalog and archive make subsequent
runs work without network access.

The default Make and CMake MotorBike preparation paths call the fetcher before
extracting `MTB_example_polyMesh.tgz`. A benchmark that uses an explicitly
supplied `--mesh=/path/to/polyMesh` continues to use that prepared directory
directly; fetch and extract its corresponding archive before invoking the
benchmark.
