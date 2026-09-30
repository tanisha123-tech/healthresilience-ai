import re

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start of TABS
start_idx = -1
for i, line in enumerate(lines):
    if "# TABS" in line:
        start_idx = i
        break

if start_idx != -1:
    # We want to indent everything from line start_idx + 13 to the end, except when we hit AI ALERTS or FEDERATED LEARNING
    # Wait, let's just use regular expressions or simple state machine.
    
    out_lines = lines[:start_idx+10]
    
    state = "DASH"
    
    for line in lines[start_idx+10:]:
        
        if "# AI ALERTS" in line:
            state = "ALERTS"
            out_lines.append("\n")
            out_lines.append("with tab_alerts:\n")
            out_lines.append("    " + line)
            continue
            
        if "# FEDERATED LEARNING" in line:
            state = "FED"
            out_lines.append("\n")
            out_lines.append("with tab_fed:\n")
            out_lines.append("    " + line)
            continue
            
        if "# FOOTER" in line:
            state = "FOOTER"
            out_lines.append(line)
            continue

        if state in ["DASH", "ALERTS", "FED"]:
            out_lines.append("    " + line)
        else:
            out_lines.append(line)
            
    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
    print("Formatted successfully")
else:
    print("Could not find TABS")
