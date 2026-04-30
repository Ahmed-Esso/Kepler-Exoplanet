with open('notebooks/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if line.startswith("# ── Global filters"):
        skip = True
        new_lines.append("# ── No Global filters ──────────────────────────────────────────────────────\n")
        new_lines.append("fdf=analysis.copy()\n")
        continue
    
    if skip and line.strip() == 'st.warning("No records match the current filter."); st.stop()':
        skip = False
        continue
        
    if skip:
        continue
        
    new_lines.append(line)

lines = new_lines
new_lines = []
skip = False

for i, line in enumerate(lines):
    if "total=len(fdf)" in line:
        skip = True
        new_lines.append("    pass\n")
        continue
        
    if skip and "st.plotly_chart(gauge(float(best[\"ROC-AUC\"])" in line:
        skip = False
        continue
        
    if skip:
        continue
        
    new_lines.append(line)

with open('notebooks/app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Done")
