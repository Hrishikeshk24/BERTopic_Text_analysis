"""
generate_data.py
----------------
Generates a synthetic credit-risk warning-signal dataset.

Output: data/warning_signals.csv
Schema:
    PD_ID                 unique borrower key (e.g., PD100001)
    comment_date          date the warning-signal comment was logged
    industry              borrower industry (context field)
    warning_signal_comment  unstructured underwriter comment
    pd_grade_before       PD grade before the warning signal (1 best .. 14 worst)
    pd_grade_after        PD grade after the warning signal (1 best .. 14 worst)
    grade_change          pd_grade_after - pd_grade_before (positive = downgrade)
    migration             Downgrade / Upgrade / No Change
    true_theme            hidden ground-truth theme label (for validating BERTopic)

Convention: on the 1-14 masterscale, a HIGHER number means HIGHER risk,
so an increase in grade number is a DOWNGRADE.
"""
import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

random.seed(42)
np.random.seed(42)

N = 2000

INDUSTRIES = ["Manufacturing", "Retail", "Healthcare", "Commercial Real Estate",
              "Oil & Gas", "Technology", "Transportation", "Construction",
              "Hospitality", "Agriculture", "Wholesale Trade", "Business Services"]

# ---------------------------------------------------------------------------
# Theme definitions. Each theme: (name, sentiment_bias, downgrade_severity, templates)
# downgrade_severity: expected notches of downgrade (negative = upgrade)
# Templates use {} placeholders filled with realistic random values.
# ---------------------------------------------------------------------------

def pct(lo, hi):
    return f"{random.randint(lo, hi)}%"

def money(lo, hi):
    return f"${random.randint(lo, hi)}M"

def ratio(lo, hi):
    return f"{random.uniform(lo, hi):.1f}x"

def quarter():
    return random.choice(["Q1", "Q2", "Q3", "Q4"]) + " " + random.choice(["2024", "2025"])

THEMES = [
    # ---- negative themes ----
    ("liquidity_stress", 2.2, [
        lambda: f"Borrower has fully drawn the revolver with utilization at {pct(85,100)}. Cash position declined to {money(2,15)} and liquidity runway is under {random.randint(3,9)} months without additional support.",
        lambda: f"Operating cash flow turned negative in {quarter()}. Working capital squeeze evident; payables stretched beyond {random.randint(60,120)} days and the borrower requested a temporary overadvance.",
        lambda: f"Significant cash burn observed. Unrestricted cash fell {pct(30,70)} quarter over quarter and the company is deferring capex to preserve liquidity. Monitoring weekly 13-week cash flow forecasts.",
        lambda: f"Liquidity concerns escalating. The borrower missed a scheduled sweep and drew the remaining {money(3,20)} of revolver availability to fund payroll. Treasury reports minimal headroom.",
        lambda: f"Cash flow from operations insufficient to cover fixed charges this cycle. FCCR dropped to {ratio(0.6,0.95)}. Sponsor has been approached for an equity injection to shore up liquidity.",
    ]),
    ("covenant_breach", 2.0, [
        lambda: f"Borrower breached the maximum leverage covenant in {quarter()}; reported {ratio(4.5,7.5)} against a {ratio(3.5,4.5)} threshold. Waiver request under negotiation with the bank group.",
        lambda: f"Fixed charge coverage covenant violation reported. FCCR of {ratio(0.7,1.05)} versus required 1.10x. This is the second consecutive quarterly breach; amendment discussions ongoing.",
        lambda: f"Financial covenants tripped at the {quarter()} test date. The borrower requested a covenant holiday through year end. Pricing step-up and additional reporting agreed as conditions.",
        lambda: f"Minimum EBITDA covenant missed by {pct(8,30)}. Compliance certificate delivered late with a qualified calculation. Documentation team reviewing reservation of rights letter.",
        lambda: f"Springing covenant triggered as revolver utilization exceeded {pct(30,40)}. Borrower is now subject to full quarterly covenant testing and failed the first leverage test.",
    ]),
    ("revenue_decline", 1.6, [
        lambda: f"Revenue contracted {pct(10,35)} year over year driven by softening demand. Gross margin compressed {random.randint(150,600)} bps on input cost inflation the borrower could not pass through.",
        lambda: f"Topline deterioration continues for a third consecutive quarter. Sales down {pct(8,25)} and order backlog thinning. Management lowered full-year guidance materially.",
        lambda: f"EBITDA declined {pct(15,45)} versus plan on volume weakness and pricing pressure. Cost reduction program announced but savings not yet visible in results.",
        lambda: f"Same-store sales fell {pct(5,20)} and foot traffic remains weak. Inventory build of {money(5,40)} raises markdown risk into next season.",
        lambda: f"Margin erosion accelerating; operating margin at {pct(1,6)} versus {pct(8,14)} prior year. Competitive discounting in the sector shows no sign of abating.",
    ]),
    ("management_turnover", 1.2, [
        lambda: f"CFO resigned abruptly in {quarter()} with no permanent successor named; interim finance lead has limited sector experience. Second C-suite departure in twelve months.",
        lambda: f"CEO transition announced amid board disagreement over strategy. Key-person risk elevated given founder's central role in top customer relationships.",
        lambda: f"Significant management turnover noted: controller, treasurer, and head of operations all departed within two quarters. Institutional knowledge loss is a concern for reporting quality.",
        lambda: f"Governance concerns raised after the audit committee chair stepped down. Board composition now lacks independent financial expertise.",
        lambda: f"Ownership dispute among family shareholders is distracting management. Succession plan remains undocumented despite repeated requests at annual review.",
    ]),
    ("industry_headwinds", 1.3, [
        lambda: f"Sector outlook deteriorating; industry volumes down {pct(8,20)} and two regional competitors filed for Chapter 11 this year. Borrower exposed to the same demand drivers.",
        lambda: f"New tariffs raise landed costs approximately {pct(10,25)} on the borrower's primary import lines. Ability to pass through to customers is unproven.",
        lambda: f"Commodity price downturn pressuring the entire sub-sector. Rig counts and shipment volumes at multi-year lows; borrower hedges roll off in {random.randint(2,4)} quarters.",
        lambda: f"Regulatory change effective next year is expected to reduce reimbursement rates by {pct(5,15)}, directly compressing borrower revenue in its core segment.",
        lambda: f"Prolonged softness in the freight market; spot rates below breakeven for {random.randint(3,6)} consecutive quarters. Fleet utilization at {pct(60,80)}.",
    ]),
    ("reporting_delinquency", 1.4, [
        lambda: f"Audited financial statements are {random.randint(60,180)} days past due. Auditor cited unresolved revenue recognition matters; a going concern qualification has not been ruled out.",
        lambda: f"Borrower delinquent on quarterly reporting for the second time this year. Field exam identified borrowing base discrepancies of approximately {money(2,12)}.",
        lambda: f"Auditor issued a qualified opinion related to inventory valuation. Management restated prior period results, reducing retained earnings by {money(3,25)}.",
        lambda: f"Repeated delays in delivering compliance certificates and monthly borrowing base certificates. Finance function appears under-resourced following staff departures.",
        lambda: f"Material weakness in internal controls disclosed. Segregation of duties issues in the treasury function; remediation plan requested by {quarter()}.",
    ]),
    ("refinancing_risk", 1.8, [
        lambda: f"Term loan of {money(50,400)} matures in {random.randint(6,15)} months with no committed refinancing. Current leverage of {ratio(5.0,7.5)} makes a market refinance challenging.",
        lambda: f"Debt maturity wall approaching; {pct(40,80)} of funded debt comes due within 18 months. Sponsor exploring amend-and-extend but lender appetite is uncertain.",
        lambda: f"Refinancing risk elevated. The borrower's bonds trade at {random.randint(55,80)} cents on the dollar, signaling constrained capital markets access.",
        lambda: f"Interest burden up sharply post rate resets; cash interest coverage at {ratio(1.0,1.6)}. Hedges expired and were not replaced, leaving full floating-rate exposure.",
        lambda: f"Leverage remains elevated at {ratio(5.5,8.0)} with no clear deleveraging path. PIK toggle exercised on the mezzanine tranche this period, growing the debt stack.",
    ]),
    ("customer_concentration", 1.5, [
        lambda: f"Top customer representing {pct(25,55)} of revenue did not renew its contract at term. Replacement pipeline covers less than half of the lost volume.",
        lambda: f"Loss of a major account announced; {money(10,80)} of annual revenue affected beginning next quarter. Concentration risk previously flagged at underwriting.",
        lambda: f"Key customer initiated in-sourcing of the borrower's product line. Customer concentration remains high with top three accounts at {pct(50,75)} of sales.",
        lambda: f"Largest customer filed for bankruptcy protection; receivable exposure of {money(3,20)} likely impaired. Trade credit insurance covers only a portion.",
        lambda: f"Contract repricing with the anchor customer cut unit margins {pct(10,30)}. Borrower has limited negotiating leverage given dependence on this relationship.",
    ]),
    ("legal_regulatory", 1.4, [
        lambda: f"Class action lawsuit filed alleging product defects; potential exposure estimated at {money(10,150)}. Insurance coverage position remains unclear.",
        lambda: f"Regulatory investigation opened into billing practices. Legal reserves increased by {money(5,40)} this quarter; reputational impact on new bookings possible.",
        lambda: f"EPA notice of violation received relating to the primary facility. Remediation cost estimates range up to {money(8,60)}; operations could face interruption.",
        lambda: f"Adverse arbitration ruling requires payment of {money(5,45)} within 90 days, straining liquidity. Appeal under consideration but counsel views prospects as limited.",
        lambda: f"OFAC compliance review identified potential sanctions exposure in an overseas subsidiary. Enhanced monitoring in place pending outside counsel findings.",
    ]),
    # ---- positive / improving themes ----
    ("performance_improvement", -1.5, [
        lambda: f"Turnaround gaining traction: EBITDA up {pct(15,50)} year over year and margins recovered to {pct(9,16)}. Cost actions delivered ahead of plan.",
        lambda: f"Strong {quarter()} results with revenue growth of {pct(8,25)} and record backlog of {money(40,300)}. Guidance raised for the full year.",
        lambda: f"Borrower returned to profitability after two loss-making years. Cash conversion improved and the revolver is now undrawn.",
        lambda: f"New contract wins add {money(15,120)} of committed multi-year revenue, diversifying the customer base and improving visibility.",
        lambda: f"Operational restructuring complete; plant consolidation yields {money(5,30)} of annualized savings. Order intake trending {pct(10,30)} above prior year.",
    ]),
    ("deleveraging", -1.8, [
        lambda: f"Successful refinancing completed; maturities extended to {random.choice(['2029','2030','2031'])} at improved pricing. Near-term refinancing risk eliminated.",
        lambda: f"Asset sale proceeds of {money(25,200)} applied to term debt. Leverage reduced to {ratio(2.0,3.5)} from {ratio(4.0,5.5)} at last review.",
        lambda: f"Sponsor injected {money(15,100)} of fresh equity, curing the covenant breach and funding the growth capex program. Liquidity restored to {money(30,120)}.",
        lambda: f"Voluntary prepayment of {money(10,80)} made this quarter. FCCR improved to {ratio(1.5,2.5)} and all covenants passed with comfortable headroom.",
        lambda: f"Debt paydown ahead of schedule; net leverage now {ratio(1.8,3.0)}. Rating outlook revised to stable given strengthened balance sheet.",
    ]),
]

# Sampling weights: negative themes more common in a warning-signal dataset
WEIGHTS = [0.14, 0.13, 0.12, 0.07, 0.09, 0.08, 0.11, 0.08, 0.06, 0.07, 0.05]
assert len(WEIGHTS) == len(THEMES) and abs(sum(WEIGHTS) - 1) < 1e-9

rows = []
start_date = date(2024, 1, 1)
for i in range(N):
    theme_idx = np.random.choice(len(THEMES), p=WEIGHTS)
    name, severity, templates = THEMES[theme_idx]
    comment = random.choice(templates)()

    # PD grade before: mid-scale mass (warning-signal population skews mid/weak)
    before = int(np.clip(round(np.random.normal(7, 2.2)), 1, 14))

    # Grade shift: theme severity + noise; positive shift = downgrade
    shift = int(round(np.random.normal(severity, 0.9)))
    # ~12% of cases: no rating action despite the signal (committee held the grade)
    if random.random() < 0.12:
        shift = 0
    after = int(np.clip(before + shift, 1, 14))

    change = after - before
    migration = "Downgrade" if change > 0 else ("Upgrade" if change < 0 else "No Change")

    rows.append({
        "PD_ID": f"PD{100001 + i}",
        "comment_date": (start_date + timedelta(days=random.randint(0, 540))).isoformat(),
        "industry": random.choice(INDUSTRIES),
        "warning_signal_comment": comment,
        "pd_grade_before": before,
        "pd_grade_after": after,
        "grade_change": change,
        "migration": migration,
        "true_theme": name,
    })

df = pd.DataFrame(rows)
import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/warning_signals.csv", index=False)
print(df["migration"].value_counts().to_string())
print("\nTheme distribution:")
print(df["true_theme"].value_counts().to_string())
print(f"\nSaved {len(df)} rows to data/warning_signals.csv")
