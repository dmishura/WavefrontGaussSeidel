# Benchmark meshes

The archives are local benchmark inputs and are intentionally ignored by Git.
Their exact sizes, SHA-256 digests, and topology statistics are recorded in
[`manifest.json`](manifest.json).

| Mesh | Variant | Cells | Internal faces | Archive size (bytes) |
|---|---|---:|---:|---:|
| AHBody | original | 2,845,652 | 8,592,613 | 111,531,462 |
| AHBody | renumbered | 2,845,652 | 8,592,613 | 112,158,065 |
| MTBHPC_small | original | 8,613,999 | 25,952,973 | 465,922,322 |
| MTBHPC_small | renumbered | 8,613,999 | 25,952,973 | 472,156,064 |
| MTB_example | original | 352,253 | 1,053,964 | 19,252,028 |
| WD_DamBreak | original | 9,376,387 | 27,975,312 | 330,305,556 |
| WD_DamBreak | renumbered | 9,376,387 | 27,975,312 | 336,870,809 |

## Metadata generation

Each archive was unpacked into a temporary directory and read with the
project's existing `PolyMeshReader`. The recorded topology fields are
`cells`, `internal_faces`, total `faces`, and `points`. The reader does not
currently expose the number of boundary patches, so that field is omitted
instead of introducing a separate metadata parser.

Archive sizes were obtained from the local files and digests were computed
with SHA-256. No archive content was modified.

## Original/renumbered invariants

For every available original/renumbered pair (`AHBody`, `MTBHPC_small`, and
`WD_DamBreak`), the following values are identical:

- cells;
- internal faces;
- total faces;
- points.

No topology-count discrepancies were found. Archive byte sizes and SHA-256
digests differ, as expected after renumbering.
