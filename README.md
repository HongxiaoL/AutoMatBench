| Entry | Description |
| --- | --- |
| jarvis_data\ | Raw data from JARVIS dataset |
| MP_data\ | Raw data from MP dataset |
| work\ |  |
| work\common\ | Dependencies |
| work\data\ | Split experiment data |
| work\figures\ | Plots |
| work\GT\ | Ground-truths |
| work\json\ | Optimization info and results |
| work\log\ |  |
| work\matched_result\ |  |
| work\result\ | Major experiment results |
| batch_process_ofm.py | Code for batch data processing |
| jarvis_download_data.py | Code for fetching data |
| MP_download_data.py | Code for fetching data |
| match_materials.py | Code for matching |
| log.txt |  |
| ailog.md |  |
| matbench_tasks.csv | Matched keys |
| matched_materials_{task}.csv | Matched data for {task} |

The standard execution order of Python files in this project:

Phase 1:
Install envs for `alignn`, `crabnet`, `rfscm`, `megnet`, and `schnet`

Phase 2: Preparation
1. `jarvis_download_data.py`
2. `MP_download_data.py`
3. `batch_process_ofm.py`
4. `process_data.py`

Phase 3: Experiment 1, 2, and 3
1. `optimization_data_gen.py`
2. `optimization_1st.py`
3. `optimization_1st_dual.py`
4. `optimization_2nd.py`

Phase 4: Case study
1. `match_materials.py`
2. `{model}_matched.py`

Phase 5: Plot
1. `plot_1st.py`
2. `plot_opt.py`
