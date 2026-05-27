task_name="band_gap"
R=0.0
N=48
S=6
print("#########################################################################################")
print(f"Generating data for({task_name},{R},{N},{S})...")
import process_data
process_data.process_data(task_name,R,N,S)
print(f"Finished generating data for({task_name},{R},{N},{S})!")
print("#########################################################################################")
