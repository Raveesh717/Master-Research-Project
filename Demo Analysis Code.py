# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 13:34:45 2025

@author: RY163UL
"""

import pandas as pd, numpy as np, re
from collections import OrderedDict

PATH = r"C:\Users\RY163UL\Downloads\Demo Analysis.xlsx"  # updated file

# 1) Load, standardize headers, and rename
demo = pd.read_excel(PATH)
demo.columns = [c.strip() for c in demo.columns]
demo = demo.rename(columns={
    "Participant No": "participant",
    "Please state your age": "age",
    "Please state your gender": "gender",
    "What is your Highest Education Level?": "education",
    "What is your occupation?": "occupation",
    "What is your monthly household income (approx.)?": "income",
    "What is your nationality?": "nationality",
    "Which residential location are you from?": "res_location",
    "Which State are you from?": "state",
    "What house type do you own?": "house_type",
    "How would you rate your ability to use digital technology (smartphones, apps, computers)?": "digital_lit",
    "Who primarily manages electricity usage/bills in your home?": "bill_manager",
    "Do you use any “smart” devices or apps to monitor/manage home energy?": "smart_use",
    "If yes, please specify which device e.g., smart meter, solar panel system, energy app, etc.": "smart_device_text"
})

# 2) Remove Participant 86 and de-duplicate by participant ID
if "participant" in demo.columns:
    demo = demo[demo["participant"] != 86]
    demo = demo.drop_duplicates(subset=["participant"])

# 3) Cleaning helpers
def norm_str(x):
    if pd.isna(x): return np.nan
    return re.sub(r"\s+", " ", str(x)).strip()
def lower(x): return norm_str(x).lower() if pd.notna(x) else x

def clean_gender(x):
    s = lower(x)
    if s is None or pd.isna(s): return np.nan
    if s in ["male","m","man"]: return "Male"
    if s in ["female","f","woman"]: return "Female"
    if "prefer" in s and "say" in s: return "Prefer not to say"
    if "non" in s and "binary" in s: return "Other"
    return s.title()

def clean_education(x):
    s = lower(x)
    if s is None or pd.isna(s): return np.nan
    mapping = {
        "no formal": "No formal schooling",
        "primary": "Primary",
        "secondary": "Secondary",
        "diploma": "Diploma/Technical",
        "technical": "Diploma/Technical",
        "certificate": "Diploma/Technical",
        "bachelor": "Bachelor’s",
        "undergraduate": "Bachelor’s",
        "postgraduate": "Postgraduate",
        "master": "Postgraduate",
        "phd": "Postgraduate",
        "doctor": "Postgraduate",
        "mba": "Postgraduate"
    }
    for k,v in mapping.items():
        if k in s: return v
    return s.title()

def clean_income(x):
    s = norm_str(x)
    if pd.isna(s): return np.nan
    s_low = s.lower()
    replacements = {
        "below rm2000": "Below RM2000",
        "rm2000–4999": "RM2000–4999",
        "rm2000-4999": "RM2000–4999",
        "rm5000–9999": "RM5000–9999",
        "rm5000-9999": "RM5000–9999",
        "rm10,000 and above": "RM10,000 and above",
        "rm10000 and above": "RM10,000 and above",
        "prefer not to say": "Prefer not to say"
    }
    for k,v in replacements.items():
        if k in s_low: return v
    return s

def clean_nationality(x):
    s = lower(x)
    if s is None or pd.isna(s): return np.nan
    s = s.replace(".", "").strip()
    mapping = {
        "mauritian":"Mauritius","mauritius":"Mauritius",
        "malaysian":"Malaysia","malaysia":"Malaysia",
        "south africa":"South Africa","south african":"South Africa",
        "singapore":"Singapore","singaporean":"Singapore",
        "india":"India","indian":"India","pakistan":"Pakistan","pakistani":"Pakistan",
        "nigeria":"Nigeria","nigerian":"Nigeria","united kingdom":"United Kingdom","uk":"United Kingdom","british":"United Kingdom",
        "england":"United Kingdom","scotland":"United Kingdom","wales":"United Kingdom",
        "france":"France","french":"France","indonesia":"Indonesia","indonesian":"Indonesia",
        "china":"China","chinese":"China","sri lanka":"Sri Lanka","sri lankan":"Sri Lanka",
        "bangladesh":"Bangladesh","bangladeshi":"Bangladesh","philippines":"Philippines","filipino":"Philippines",
        "thailand":"Thailand","thai":"Thailand","vietnam":"Vietnam","vietnamese":"Vietnam",
        "nepal":"Nepal","nepali":"Nepal","nepalese":"Nepal","myanmar":"Myanmar","burmese":"Myanmar",
        "brunei":"Brunei","bruneian":"Brunei","australia":"Australia","australian":"Australia",
        "new zealand":"New Zealand","kiwi":"New Zealand","canada":"Canada","canadian":"Canada",
        "united states":"United States","usa":"United States","american":"United States",
        "iran":"Iran","iranian":"Iran","iraq":"Iraq","iraqi":"Iraq","saudi arabia":"Saudi Arabia","saudi":"Saudi Arabia",
        "uae":"United Arab Emirates","united arab emirates":"United Arab Emirates","emirati":"United Arab Emirates",
        "turkey":"Turkey","turkish":"Turkey","japan":"Japan","japanese":"Japan",
        "korea":"South Korea","south korea":"South Korea","korean":"South Korea","laos":"Laos","lao":"Laos",
        "cambodia":"Cambodia","khmer":"Cambodia","spain":"Spain","spanish":"Spain","italy":"Italy","italian":"Italy",
        "germany":"Germany","german":"Germany","netherlands":"Netherlands","dutch":"Netherlands",
        "sweden":"Sweden","swedish":"Sweden","norway":"Norway","norwegian":"Norway","denmark":"Denmark","danish":"Denmark",
        "ireland":"Ireland","irish":"Ireland","mexico":"Mexico","mexican":"Mexico","brazil":"Brazil","brazilian":"Brazil",
        "argentina":"Argentina","argentine":"Argentina","egypt":"Egypt","egyptian":"Egypt","ethiopia":"Ethiopia","ethiopian":"Ethiopia",
        "ghana":"Ghana","ghanaian":"Ghana","kenya":"Kenya","kenyan":"Kenya","tanzania":"Tanzania","tanzanian":"Tanzania",
        "zimbabwe":"Zimbabwe","zimbabwean":"Zimbabwe"
    }
    if s in mapping: return mapping[s]
    for k,v in mapping.items():
        if k in s: return v
    return s.title()

def clean_res_location(x):
    s = lower(x)
    if s is None or pd.isna(s): return np.nan
    if "urban" in s: return "Urban"
    if "rural" in s: return "Rural"
    return s.title()

def clean_state(x):
    s = norm_str(x); return s.title() if pd.notna(s) else np.nan

def clean_house_type(x):
    s = lower(x)
    if s is None or pd.isna(s): return np.nan
    if any(k in s for k in ["apartment","condo","flat"]): return "Apartment/Condo"
    if any(k in s for k in ["bungalow","terrace","landed","house"]): return "Landed house (bungalow/terrace)"
    return s.title()

def clean_digital_lit(x):
    s = lower(x)
    if s is None or pd.isna(s): return np.nan
    if "low" in s: return "Low"
    if "moderate" in s or "medium" in s: return "Moderate"
    if "high" in s: return "High"
    return s.title()

def clean_bill_manager(x):
    s = lower(x)
    if s is None or pd.isna(s): return np.nan
    if "self" in s: return "Self"
    if "spouse" in s or "family" in s: return "Spouse/Family"
    if "equal" in s or "shared" in s: return "Shared equally"
    return "Other"

def clean_smart_use(x):
    s = lower(x)
    if s is None or pd.isna(s): return np.nan
    if s in ["yes","y","true","1"]: return "Yes"
    if s in ["no","n","false","0"]: return "No"
    if "yes" in s: return "Yes"
    if "no" in s: return "No"
    return s.title()

# 4) Apply cleaning
for col, func in [
    ("gender", clean_gender),
    ("education", clean_education),
    ("income", clean_income),
    ("nationality", clean_nationality),
    ("res_location", clean_res_location),
    ("state", clean_state),
    ("house_type", clean_house_type),
    ("digital_lit", clean_digital_lit),
    ("bill_manager", clean_bill_manager),
    ("smart_use", clean_smart_use),
]:
    if col in demo.columns:
        demo[col] = demo[col].apply(func)

# 5) Age & bands
demo["age"] = pd.to_numeric(demo.get("age"), errors="coerce")
demo.loc[(demo["age"] < 15) | (demo["age"] > 100), "age"] = np.nan
bins = [17, 24, 34, 44, 54, 64, 100]
labels = ["18–24","25–34","35–44","45–54","55–64","65+"]
demo["age_band"] = pd.cut(demo["age"], bins=bins, labels=labels)

# 6) Build the sample characteristics table
N = len(demo)
def counts_with_pct(series):
    ct = series.dropna().value_counts(dropna=False)
    return ", ".join([f"{idx}: {int(n)} ({(n/N)*100:.1f}%)" for idx,n in ct.items()])

rows = []
rows.append(("Total Participants", f"{N} residential respondents across Malaysia"))
rows.append(("Gender", counts_with_pct(demo["gender"])))
nat_ct = demo["nationality"].fillna("Missing/Unspecified").value_counts(dropna=False)
rows.append(("Nationalities (all)", ", ".join([f"{idx}: {int(n)} ({(n/N)*100:.1f}%)" for idx,n in nat_ct.items()])))
rows.append(("Residential location", counts_with_pct(demo["res_location"])))
state_ct = demo["state"].fillna("Missing/Unspecified").value_counts(dropna=False)
rows.append(("States (all)", ", ".join([f"{idx}: {int(n)} ({(n/N)*100:.1f}%)" for idx,n in state_ct.items()])))
rows.append(("Education", counts_with_pct(demo["education"])))
rows.append(("Monthly household income", counts_with_pct(demo["income"])))
rows.append(("House type", counts_with_pct(demo["house_type"])))
rows.append(("Digital literacy", counts_with_pct(demo["digital_lit"])))
rows.append(("Primary manager of electricity bills", counts_with_pct(demo["bill_manager"])))
rows.append(("Current smart energy use", counts_with_pct(demo["smart_use"])))
age_mean = demo["age"].mean(); age_sd = demo["age"].std(ddof=1)
bands = demo["age_band"].value_counts(sort=False)
rows.append(("Age (years)", f"M = {age_mean:.1f}, SD = {age_sd:.1f}; Bands → " + ", ".join([f"{idx}: {int(n)} ({(n/N)*100:.1f}%)" for idx,n in bands.items() if pd.notna(idx)])))

sample_table = pd.DataFrame(rows, columns=["Category","Details"])
sample_table.to_excel("sample_characteristics_updated.xlsx", index=False)
demo.to_excel("demographics_cleaned_updated.xlsx", index=False)
print("Saved sample_characteristics_updated.xlsx and demographics_cleaned_updated.xlsx; N =", N)
