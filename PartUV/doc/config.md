# **Configuration Overview**

All parameters can be specified in a unified YAML configuration file, shared by both the C++ and Python APIs.
Users can provide the path to this configuration file when invoking the pipeline.

---

## **1. Core Pipeline Controls (Frequently Adjusted)**

| YAML path                  | Type   | Default | Description                                                                                                             |
| -------------------------- | ------ | ------- | ----------------------------------------------------------------------------------------------------------------------- |
| `pipeline.threshold`       | double | `1.25`  | Distortion threshold for recursive unwrapping. range: [1.0, +inf). Larger values yield more distortion but faster runtime and fewer charts. |
| `unwrap.pamo`              | bool   | `true`  | Enable PAMO (GPU-based simplification) for distortion approximation. Disable if running on CPU-only machines.               |
| `pipeline.num_omp_threads` | int    | `8`     | Number of OpenMP threads per nested level for parallel processing.                                                      |
| `unwrap.agg_parts`         | int    | `10`    | Number of clusters used for normal heuristic.                                                              |

---

## **2. Unwrapping Method & Optimization Settings**

| YAML path          | Type   | Default | Description                                  |
| ------------------ | ------ | ------- | -------------------------------------------- |
| `unwrap.method`    | string | `"abf"` | Unwrapping algorithm: `"abf"` or `"lscm"`.   |
| `unwrap.abf_iters` | int    | `5`     | Number of ABF iterations when `method: abf`. |

---

## **3. Parallelism and Resource Management**

| YAML path                     | Type | Default | Description                                                                                                  |
| ----------------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------ |
| `pipeline.parallelDepth`      | int  | `10`    | Beyond this tree depth, left/right children are no longer processed in parallel (as the benefit diminishes). |
| `pipeline.num_cuda_streams`   | int  | `10`    | Number of CUDA streams for PAMO parallelism.                                                                 |
| `pipeline.component_maxDepth` | int  | `10`    | When the input mesh contains multiple connected components, the heuristic step is skipped, and PartField is used to divide the mesh instead. This skipping behavior applies up to the depth level specified by component_maxDepth.                      |

---

## **4. Safety & Validation Checks**

| YAML path                        | Type | Default | Description                                                                                  |
| -------------------------------- | ---- | ------- | -------------------------------------------------------------------------------------------- |
| `pipeline.checkSelfIntersection` | bool | `false` | Run a 3D self-intersection check before unwrapping. The pipeline will exit if any are found. |
| `pipeline.checkNon2Manifold`     | bool | `true`  | Run non-2-manifold check before unwrapping. The pipeline will exit if violations are found.  |

---

## **5. Debugging, Logging, and Intermediate Outputs**

| YAML path    | Type | Default | Description                                            |
| ------------ | ---- | ------- | ------------------------------------------------------ |
| `verbose`    | bool | `false` | Print detailed logs if `true`.                         |
| `save_stuff` | bool | `false` | Save intermediate results (for debugging or analysis). |

