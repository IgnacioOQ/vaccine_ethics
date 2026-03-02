import json

def update_notebook():
    file_path = 'Precautionary_Principle_Analysis.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Standard \\n in python multiline strings becomes \n in the json text array
    new_source = """# Save results to CSV/Text
import datetime
import pandas as pd

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"{dump_path}precautionary_analysis_{timestamp}.txt"
with open(filename, "w") as f:
    f.write("Precautionary Principle Analysis Results\\n")
    f.write("========================================\\n\\n")
    f.write(f"Vaccinate All - Worst Case Death Proportion: {worst_death_prop_vax_all:.4f}\\n")
    f.write(f"Parameters: ({worst_params_vax_all[0]:.4f}, {worst_params_vax_all[1]:.4f}, {worst_params_vax_all[2]:.4f})\\n\\n")
    f.write(f"Vaccinate Vulnerable Only - Worst Case Death Proportion: {worst_death_prop_vax_vuln:.4f}\\n")
    f.write(f"Parameters: ({worst_params_vax_vuln[0]:.4f}, {worst_params_vax_vuln[1]:.4f}, {worst_params_vax_vuln[2]:.4f})\\n\\n")

    if worst_death_prop_vax_all < worst_death_prop_vax_vuln:
        f.write("CONCLUSION: 'Vaccinate All' is the safer precautionary strategy.\\n")
    else:
        f.write("CONCLUSION: 'Vaccinate Vulnerable Only' is the safer precautionary strategy.\\n")

print(f"Results saved to {filename}")

# Save the datasets for convergence plots
df_vax_all = pd.DataFrame(res_vax_all.x_iters, columns=['death_prob', 'vax_effect', 'viral_age_effect'])
df_vax_all['f_val'] = -res_vax_all.func_vals
csv_filename_all = f"{dump_path}precautionary_vax_all_iterations_{timestamp}.csv"
df_vax_all.to_csv(csv_filename_all, index=False)
print(f"Dataset for Vax All saved to {csv_filename_all}")

df_vax_vuln = pd.DataFrame(res_vax_vuln.x_iters, columns=['death_prob', 'vax_effect', 'viral_age_effect'])
df_vax_vuln['f_val'] = -res_vax_vuln.func_vals
csv_filename_vuln = f"{dump_path}precautionary_vax_vuln_iterations_{timestamp}.csv"
df_vax_vuln.to_csv(csv_filename_vuln, index=False)
print(f"Dataset for Vax Vuln Only saved to {csv_filename_vuln}")"""


    replaced_save = False
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if 'filename = f"{dump_path}precautionary_analysis_{timestamp}.txt"' in source:
                # the source lines for jupyter need to have literal `\n` characters at the end of every line strings.
                # However, the characters inside the cell line like f.write("...\\n") should be literal backslash followed by 'n'.
                lines = new_source.split('\\n')
                new_lines = [line + '\\n' for line in lines[:-1]] + [lines[-1]]
                cell['source'] = new_lines
                replaced_save = True
                print("Found and replaced the results saving cell.")

    if replaced_save:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook saved successfully.")
    else:
        print(f"Could not find cells to replace. Save: {replaced_save}")

if __name__ == '__main__':
    update_notebook()
