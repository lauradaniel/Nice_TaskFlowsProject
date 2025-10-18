import pandas as pd

# Load Excel file
excel_path = './data/L123Intent_AgentTaskLabel_Mapping.xlsx'  # Change this to your file path
df = pd.read_excel(excel_path)

# Drop duplicates to avoid redundancy
df = df[['Level1_Category_Mapped', 'Level2_Topic_Mapped', 'Level3_CallIntents_Mapped']].drop_duplicates()

# Sort for cleaner grouping
df = df.sort_values(['Level1_Category_Mapped', 'Level2_Topic_Mapped', 'Level3_CallIntents_Mapped'])

# Write to YAML-style .txt file
output_path = './data/YAML_mapping_fixed.txt'
with open(output_path, 'w') as f:
    current_L1 = current_L2 = ''
    for row in df.itertuples(index=False):
        L1, L2, L3 = row
        if L1 != current_L1:
            f.write(f'- {L1}\n')
            current_L1 = L1
            current_L2 = ''  # reset L2
        if L2 != current_L2:
            f.write(f'    - {L2}\n')
            current_L2 = L2
        f.write(f'        - {L3}\n')

print(f'✅ YAML-style category file saved to: {output_path}')
