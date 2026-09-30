import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

gemini_code = """
                    if gemini_key and GENAI_AVAILABLE:
                        if st.button(f"✨ Generate Crisis Action Plan (Gemini)", key=f"gemini_{row['phc_id']}_{row['medicine']}"):
                            with st.spinner("Gemini is analyzing the supply chain data..."):
                                try:
                                    system_instruction = (
                                        "You are an elite Healthcare Logistics AI. Your task is to output a 3-point crisis action plan. "
                                        "Edge case protocols: "
                                        "1. If distance > 20km, explicitly mandate cold-chain or secure transit verification. "
                                        "2. If risk is 'High', mandate immediate notification of the district medical officer. "
                                        "3. If surplus is tight, advise rationing remaining stock strictly for vulnerable patients."
                                    )
                                    model = genai.GenerativeModel(
                                        'gemini-1.5-flash',
                                        system_instruction=system_instruction
                                    )
                                    prompt = (
                                        f"Context: Clinic {row['phc_id']} is at {row['risk_level']} risk of a {row['medicine']} stockout. "
                                        f"Current stock: {row['current_stock']} | Predicted 7-day demand: {row['predicted_demand_7d']}. "
                                        f"Action: Transferring {transfer_quantity} units from {best['phc_id']} ({best['distance_km']:.1f} km away). "
                                        f"Draft the briefing."
                                    )
                                    response = model.generate_content(prompt)
                                    st.info(response.text)
                                except Exception as e:
                                    st.error(f"Gemini API Error: {e}")
"""

# Let's use regex to find the end of st.success(...)
pattern = r'(st\.success\(\s*f"""[^"]+"""\s*\))'
match = re.search(pattern, content)

if match:
    new_content = content[:match.end()] + "\n" + gemini_code + content[match.end():]
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully patched app.py")
else:
    print("Could not find the target string via regex.")
