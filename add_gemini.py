import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add GENAI imports
import_code = """
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
"""
content = content.replace("SHAP_AVAILABLE = False", "SHAP_AVAILABLE = False\n" + import_code)

# 2. Add Gemini key to sidebar
sidebar_code = """
st.sidebar.header("⚙️ Control Center")

gemini_key = st.sidebar.text_input("🔑 Gemini API Key (Optional)", type="password", help="Enter your Google AI Studio key to enable LLM features")
if gemini_key and GENAI_AVAILABLE:
    genai.configure(api_key=gemini_key)
"""
content = content.replace('st.sidebar.header("⚙️ Control Center")', sidebar_code)

# 3. Add Gemini button
gemini_button_code = """
                    st.success(
                        f\"\"\"
**🤖 AI Redistribution Recommendation**

Transfer **{transfer_quantity} units** of
**{row['medicine']}**

**From:** {best['phc_id']} ({best['distance_km']:.1f} km away, Lead time: {best['lead_time_days']} days)

**To:** {row['phc_id']}

**Reason:** {row['phc_id']} has a predicted shortage,
while {best['phc_id']} has sufficient projected surplus and is a suitable redistribution source based on distance and lead time.
\"\"\"
                    )
                    
                    if gemini_key and GENAI_AVAILABLE:
                        if st.button(f"✨ Generate Crisis Action Plan (Gemini)", key=f"gemini_{row['phc_id']}_{row['medicine']}"):
                            with st.spinner("Gemini is analyzing the supply chain data..."):
                                try:
                                    model = genai.GenerativeModel('gemini-1.5-flash')
                                    prompt = f"Act as a Healthcare Logistics Expert. A Primary Health Centre ({row['phc_id']}) is at {row['risk_level']} risk of running out of {row['medicine']}. Current stock: {row['current_stock']}, but predicted 7-day demand is {row['predicted_demand_7d']}. We are immediately transferring {transfer_quantity} units from a surplus clinic ({best['phc_id']}) which is {best['distance_km']:.1f} km away. Write a very concise, professional 3-point bulleted action plan for the medical staff receiving this shipment to manage the crisis."
                                    response = model.generate_content(prompt)
                                    st.info(response.text)
                                except Exception as e:
                                    st.error(f"Gemini API Error: {e}")
"""

content = re.sub(
    r'st\.success\(\s*f"""\s*\*\*🤖 AI Redistribution Recommendation.*?\n"""\s*\)', 
    gemini_button_code.strip(), 
    content, 
    flags=re.DOTALL
)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated app.py with Gemini integration!")
