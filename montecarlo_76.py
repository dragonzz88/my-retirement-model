import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility
n_simulations = 3000  # Number of Monte Carlo simulations

# ---- Trust Parameters ----
initial_trust_cash = 100000
trust_etf_return_mean = 0.07
trust_etf_return_volatility = 0.14
trust_etf_dividend_yield = 0.04
child_initial_age = 13
annual_trust_expense_real = 1500 # Annual real trust administration expenses
retirement_age = 68

# ---- Retirement Parameters ----
accumulation_years = 15
retirement_years = 30
total_years = accumulation_years + retirement_years

# ---- Trust Simulation ----

def calc_beneficiary_tax(income, age):
    tax = 0
    taxable_income = max(0, income) # Ensure income is not negative for tax calculation

    # Medicare levy (2%) for resident individuals
    medicare_levy = 0
    # Simplified threshold for medicare levy - typically applies above certain income levels
    if taxable_income > 26000:
        medicare_levy = taxable_income * 0.02

    # Simplified Australian Resident Individual Tax Rates (approx. 2024-2025 financial year)
    # NOTE: This is a simplification. Actual tax rules for minors' unearned income from trusts
    # can be complex and may involve penalty tax rates if not 'excepted income'.
    # This simulation assumes income is taxed at general adult resident rates,
    # consistent with the 'gifted back' scenario often structured for tax efficiency.
    if taxable_income <= 18200:
        tax_before_medicare = 0
    elif taxable_income <= 45000:
        tax_before_medicare = (taxable_income - 18200) * 0.19 # Corrected bracket calculation
    elif taxable_income <= 135000:
        tax_before_medicare = 5092 + (taxable_income - 45000) * 0.30
    elif taxable_income <= 190000:
        tax_before_medicare = 32092 + (taxable_income - 135000) * 0.37
    else:
        tax_before_medicare = 52642 + (taxable_income - 190000) * 0.45

    return tax_before_medicare + medicare_levy

def simulate_trust_module(sim_years, child_current_age):
    trust_records = []
    for sim in range(n_simulations):
        trust_balance = initial_trust_cash
        inflation_rate = 0.025 # Consistent with retirement model
        child_age = child_current_age
        sim_data = []

        for year in range(sim_years):
            total_return = np.random.normal(trust_etf_return_mean, trust_etf_return_volatility)

            dividend_income = trust_balance * trust_etf_dividend_yield
            capital_appreciation = trust_balance * (total_return - trust_etf_dividend_yield)

            # --- Tax Calculation for Trust Income Attributed to Beneficiary ---
            # Assume all dividend income is distributed and taxed
            # Assume positive capital appreciation is 'realized' annually for tax purposes,
            # and receives the 50% CGT discount before being taxed.

            income_for_beneficiary_tax = dividend_income + max(0, capital_appreciation) * 0.5

            tax_paid_by_beneficiary = calc_beneficiary_tax(income_for_beneficiary_tax, child_age)

            net_amount_gifted_back = income_for_beneficiary_tax - tax_paid_by_beneficiary

            trust_expense_nom = annual_trust_expense_real * (1 + inflation_rate)**year
            net_amount_gifted_back -= trust_expense_nom # Deduct expenses from the amount gifted back

            # Update Trust Balance
            # 1. Trust balance grows by the full capital appreciation (market value change)
            trust_balance += capital_appreciation

            # 2. Add the after-tax distributed amount (dividends + realized gains)
            #    that is then gifted back into the Trust for reinvestment.
            trust_balance += net_amount_gifted_back

            sim_data.append([year + 1, trust_balance, dividend_income, child_age]) # dividend_income is gross for info
            child_age += 1

        trust_records.append(pd.DataFrame(sim_data, columns=["Yr", "TrustBalance", "Dividends", "ChildAge"]))
    return trust_records

def run_retirement_model_bucket(initial_etf_amount=0,
                                annual_self_contribution_you=18000,
                                asfa_modest_real=48000, # ASFA Modest in real terms (initial year)
                                initial_living_expenses_real=40000, # Living expenses in accumulation phase, real terms
                                spending_floor_percentage_of_asfa=0.6,
                                years_in_cash_bucket=2, # Number of years of expenses to hold in cash bucket
                                years_in_medium_bucket=5, # Number of years of expenses to hold in medium bucket
                                initial_current_health_cost_real=3000, # Added parameter for regular health cost
                                initial_sudden_health_cost_real=15000): # Added parameter for potential sudden health cost

    # --- Model Parameters ---
    years_accumulation = 15 # Years until retirement (e.g., from age 53 to 68)
    years_retirement = 30 # Years in retirement (e.g., from age 68 to 98)
    total_years = years_accumulation + years_retirement

    # Inflation and Salary Growth
    inflation_rate = 0.025 # Annual inflation rate
    initial_salary_you = 77000 # Your initial gross annual salary
    initial_super_you = 36000 # Your initial super balance
    initial_super_spouse = 0 # Spouse's initial super balance
    annual_self_contribution_spouse = 3000 # Spouse's annual self-contribution to super
    employer_contribution_rate = 0.12 # Superannuation Guarantee (SG) rate

    # Superannuation Tax Rates
    concessional_contribution_tax_rate = 0.15 # Tax on concessional contributions
    super_earnings_tax_acc_phase = 0.15 # Earnings tax in accumulation phase
    super_earnings_tax_ret_phase = 0.0 # Earnings tax in retirement phase (tax-free)

    # Investment Returns (Expected Annual Returns and Volatility)
    expected_return_acc = 0.063 # Super return in accumulation phase
    volatility_acc = 0.12
    expected_return_ret_super = 0.058 # Super return in retirement phase (Bucket 2/3 asset)
    volatility_ret_super = 0.10
    expected_return_etf_growth = 0.07 # ETF return (Bucket 3 asset)
    volatility_etf_growth = 0.14
    etf_dividend_yield = 0.04 # ETF dividend yield (component of total return)

    # We need a low-risk return for the cash bucket (Bucket 1)
    expected_return_cash = inflation_rate # Assuming cash keeps pace with inflation
    volatility_cash = 0.01 # Very low volatility for cash

    # Correlation between Super and ETF returns
    corr_matrix = np.array([[1.0, 0.4],[0.4, 1.0]])
    chol = np.linalg.cholesky(corr_matrix) # Cholesky decomposition for correlated random numbers

    # Age Pension Parameters (current 2025 figures, will be inflated)
    initial_full_pension_couple = 42700
    initial_full_pension_single = 28500
    initial_lower_threshold_couple = 481500
    initial_upper_threshold_couple = 1059000
    initial_lower_threshold_single = 301750
    initial_upper_threshold_single = 888250
    asset_taper_rate = 78 # Pension reduces by $7800/year for every $10000 over lower threshold

    # Deeming Rates and Thresholds
    deeming_rate_lower = 0.0025 # 0.25%
    deeming_rate_upper = 0.0225 # 2.25%
    initial_deeming_threshold_single = 60400 # Per individual, not per couple for thresholds
    initial_deeming_threshold_couple = 100200 # Combined threshold for couples (different rule for combined assets)

    # Income Free Areas (per fortnight, converted to annual)
    income_free_area_couple = 360 * 26
    income_free_area_single = 204 * 26

    # Taxation Parameters
    initial_medicare_levy_threshold = 41089

    # Superannuation Transfer Balance Cap (TBC)
    transfer_balance_cap = 1_900_000

    # --- Helper Functions ---
    def calc_personal_tax(income, medicare_levy_threshold):
        tax = 0
        if income <= 18200: tax = 0
        elif income <= 45000: tax = (income - 18200) * 0.19
        elif income <= 135000: tax = (45000 - 18200) * 0.19 + (income - 45000) * 0.30
        elif income <= 190000: tax = (45000 - 18200) * 0.19 + (135000 - 45000) * 0.30 + (income - 135000) * 0.37
        else: tax = (45000 - 18200) * 0.19 + (135000 - 45000) * 0.30 + (190000 - 135000) * 0.37 + (income - 190000) * 0.45
        if income > medicare_levy_threshold:
            tax += income * 0.02
        return tax

    drawdown_ages = [65, 75, 80, 85, 90, 95]
    drawdown_rates = [0.05, 0.06, 0.07, 0.09, 0.11, 0.14]
    def get_min_drawdown_rate(age):
        for i, a in enumerate(drawdown_ages):
            if age < a:
                return drawdown_rates[i - 1] if i > 0 else drawdown_rates[0]
        return drawdown_rates[-1]

    # --- Simulation Loop ---
    all_sims = []
    for sim in range(n_simulations):
        super_you_acc = initial_super_you
        super_spouse_acc = initial_super_spouse

        # Initialize bucket balances
        etf_cash_bucket = 0
        etf_growth_bucket = initial_etf_amount # All initial ETF goes to growth
        etf_growth_cost_base = initial_etf_amount

        super_pension_bucket_you = 0
        super_pension_bucket_spouse = 0

        salary = initial_salary_you
        life_you = int(np.random.normal(90, 5))
        life_spouse = int(np.random.normal(90, 5))
        survivor_mode = False
        retirement_start_assets = None  # Will be set once at the start of retirement

        sim_data = []

        for year in range(total_years):
            # Initialize alive_you and alive_spouse for safety
            alive_you = True
            alive_spouse = True

            inflation_factor = (1 + inflation_rate)**year
            age_you = 53 + year
            age_spouse = 52 + year
            asfa_nom = asfa_modest_real * inflation_factor

            # Initialize for all years, will be overwritten in retirement phase
            lower_asset_thr = 0
            upper_asset_thr = 0

            # Inflate parameters
            pension_nom_couple = initial_full_pension_couple * inflation_factor
            pension_nom_single = initial_full_pension_single * inflation_factor
            lower_thr_couple = initial_lower_threshold_couple * inflation_factor
            upper_thr_couple = initial_upper_threshold_couple * inflation_factor
            lower_thr_single = initial_lower_threshold_single * inflation_factor
            upper_thr_single = initial_upper_threshold_single * inflation_factor

            medicare_levy_threshold = initial_medicare_levy_threshold * inflation_factor
            tbc = transfer_balance_cap * inflation_factor # Simplified TBC inflation

            # Generate Correlated Market Returns
            z = np.random.normal(size=2)
            correlated_z = chol @ z
            if year < years_accumulation:
                r_super = expected_return_acc + correlated_z[0] * volatility_acc
                r_etf_growth = expected_return_etf_growth + correlated_z[1] * volatility_etf_growth
                r_cash = expected_return_cash + np.random.normal(0, volatility_cash) # Cash separate volatility
            else:
                r_super = expected_return_ret_super + correlated_z[0] * volatility_ret_super # Super pension bucket return
                r_etf_growth = expected_return_etf_growth + correlated_z[1] * volatility_etf_growth
                r_cash = expected_return_cash + np.random.normal(0, volatility_cash)

            # --- Expense Calculations ---
            # Random shocks
            # 1. Major unexpected expense (original one, prob increased to 15%)
            unexpected_expense_generic = np.random.randint(10000, 50000) if np.random.rand() < 0.15 else 0

            # 2. Smaller, more frequent unexpected expense (new)
            unexpected_expense_minor = np.random.randint(2000, 5000) if np.random.rand() < 0.35 else 0

            # Combine generic unexpected expenses
            unexpected_expense = unexpected_expense_generic + unexpected_expense_minor

            # Current Health Cost (new)
            current_health_cost = initial_current_health_cost_real * inflation_factor

            # Sudden Health Cost (new, a specific type of larger unexpected health shock)
            sudden_health_cost = 0
            if year >= years_accumulation: # Only in retirement phase
                if np.random.rand() < 0.05: # 5% chance of a sudden health cost
                    sudden_health_cost = initial_sudden_health_cost_real * inflation_factor


            # Age Care Cost (CareCost)
            age_care_cost = 0
            if year >= years_accumulation: # Only in retirement phase
                care_prob = 0
                if age_you >= 75 and age_you < 80:
                    care_prob = 0.10
                elif age_you >= 80 and age_you < 85:
                    care_prob = 0.20
                elif age_you >= 85:
                    care_prob = 0.50

                if np.random.rand() < care_prob:
                    age_care_cost = np.random.randint(20000, 60000) # Increased magnitude

            # Survivorship
            alive_you = age_you < life_you
            alive_spouse = age_spouse < life_spouse

            if not survivor_mode:
                if not alive_you and alive_spouse:
                    if super_you_acc > 0 or super_pension_bucket_you > 0:
                        super_spouse_acc += super_you_acc + super_pension_bucket_you # Transfer super to spouse
                    super_you_acc = super_pension_bucket_you = 0
                    survivor_mode = True
                elif not alive_spouse and alive_you:
                    if super_spouse_acc > 0 or super_pension_bucket_spouse > 0:
                        super_you_acc += super_spouse_acc + super_pension_bucket_spouse # Transfer super to you
                    super_spouse_acc = super_pension_bucket_spouse = 0
                    survivor_mode = True

            # Retirement Transition
            if year == years_accumulation:
                # Move super accumulation balances to retirement phase (Super Pension Bucket)
                move_you = min(super_you_acc, tbc)
                super_pension_bucket_you += move_you
                super_you_acc -= move_you
                move_spouse = min(super_spouse_acc, tbc)
                super_pension_bucket_spouse += move_spouse
                super_spouse_acc -= move_spouse

                # Initial bucket allocation for ETF and Super Pension
                initial_planned_spending_base = asfa_modest_real * inflation_factor # Use ASFA as initial guide

                # Calculate initial cash bucket target
                cash_target = initial_planned_spending_base * years_in_cash_bucket

                # Prioritize filling cash bucket from ETF Growth first, then Super Accumulation if still remaining
                if etf_growth_bucket >= cash_target:
                    etf_cash_bucket += cash_target
                    etf_growth_bucket -= cash_target
                    # Adjust cost base proportionally to the amount moved out
                    if initial_etf_amount > 0: # Avoid division by zero if initial_etf_amount was 0
                        etf_growth_cost_base -= etf_growth_cost_base * (cash_target / initial_etf_amount)
                    else: # If no initial ETF, no cost base to adjust
                        etf_growth_cost_base = 0
                else: # Need to draw from Super if ETF growth is insufficient to meet cash target
                    etf_cash_bucket += etf_growth_bucket
                    remaining_cash_needed = cash_target - etf_growth_bucket
                    etf_growth_bucket = 0
                    etf_growth_cost_base = 0

                    total_super_to_move_to_cash = remaining_cash_needed

                    # Distribute the initial cash transfer from super proportionally
                    total_available_super_for_initial_cash = super_pension_bucket_you + super_pension_bucket_spouse

                    if total_available_super_for_initial_cash > 0:
                        amount_from_super = min(total_super_to_move_to_cash, total_available_super_for_initial_cash)

                        # Proportionally deduct from each super pension bucket
                        ratio_you_initial_transfer = super_pension_bucket_you / total_available_super_for_initial_cash
                        super_pension_bucket_you -= amount_from_super * ratio_you_initial_transfer
                        super_pension_bucket_spouse -= amount_from_super * (1 - ratio_you_initial_transfer)
                        etf_cash_bucket += amount_from_super
                    else:
                        pass # etf_cash_bucket already has what it could get from ETF growth

                # Record initial assets for guardrail benchmark
                retirement_start_assets = (super_you_acc + super_spouse_acc + \
                                           super_pension_bucket_you + super_pension_bucket_spouse) + \
                                           etf_cash_bucket + etf_growth_bucket

            # --- Accumulation Phase Logic ---
            if year < years_accumulation:
                employer_contrib_net = salary * employer_contribution_rate * (1 - concessional_contribution_tax_rate)
                self_contrib_you_net = annual_self_contribution_you * (1 - concessional_contribution_tax_rate)
                self_contrib_spouse_net = annual_self_contribution_spouse * (1 - concessional_contribution_tax_rate)

                super_you_acc += super_you_acc * r_super * (1 - super_earnings_tax_acc_phase) + employer_contrib_net + self_contrib_you_net
                super_spouse_acc += super_spouse_acc * r_super * (1 - super_earnings_tax_acc_phase) + self_contrib_spouse_net

                etf_dividend = etf_growth_bucket * etf_dividend_yield
                etf_growth_value = etf_growth_bucket * (r_etf_growth - etf_dividend_yield) # Capital appreciation
                gross_salary = salary * (1 + inflation_rate)

                planned_spending = initial_living_expenses_real * inflation_factor # Use living expenses for accumulation
                current_health_cost_acc = initial_current_health_cost_real * inflation_factor # Inflated current health cost
                sudden_health_cost_acc = 0 # No sudden health cost during accumulation unless specified

                tax_on_salary = calc_personal_tax(gross_salary, medicare_levy_threshold)
                # Tax on dividends from ETF growth bucket
                tax_on_dividend = calc_personal_tax(etf_dividend, medicare_levy_threshold)
                net_salary = gross_salary - tax_on_salary
                net_dividend = etf_dividend - tax_on_dividend

                etf_growth_bucket += etf_growth_value + net_dividend # Dividends reinvested into growth bucket
                etf_growth_cost_base += net_dividend # Cost base increases with reinvested dividends

                disposable_income = net_salary - planned_spending - unexpected_expense - current_health_cost_acc - sudden_health_cost_acc # Deduct all expenses
                if disposable_income > 0:
                    etf_growth_bucket += disposable_income
                    etf_growth_cost_base += disposable_income

                salary = gross_salary
                total_assets = (super_you_acc + super_spouse_acc) + etf_growth_bucket + etf_cash_bucket
                pension_assets = pension_income = pension_recv = draw_super = draw_etf = tax = 0
                total_income = net_salary + net_dividend + max(disposable_income, 0) # This needs to reflect total cash inflow
                percent_pension = percent_super = percent_etf = 0
                cash_from_etf_for_spending = 0 # Not used in accumulation but in output

                # Set plotting values for accumulation phase explicitly
                actual_planned_spending = planned_spending
                actual_current_health_cost = current_health_cost_acc
                actual_sudden_health_cost = sudden_health_cost_acc
                actual_age_care_cost = 0 # No age care in accumulation
                actual_unexpected_expense = unexpected_expense
                actual_asfa = asfa_nom # Use ASFA as a reference even in accumulation

            # --- Retirement Phase Logic ---
            else:
                draw_super = 0
                draw_etf = 0
                additional_cash_draw_from_cash_bucket_for_shortfall = 0
                additional_super_draw_from_rem = 0
                additional_etf_growth_draw_from_rem = 0

                # Define current_pension_age for drawdown rate calculation
                current_pension_age = age_you # Using 'age_you' as the primary age for pension calculations

                # Base for guardrail calculation
                current_planned_spending_base = asfa_modest_real * inflation_factor

                # Asset growth for buckets
                super_pension_bucket_you *= (1 + r_super)
                super_pension_bucket_spouse *= (1 + r_super)
                etf_cash_bucket *= (1 + r_cash) # Cash bucket earns low return

                etf_dividend = etf_growth_bucket * etf_dividend_yield
                etf_growth_value = etf_growth_bucket * (r_etf_growth - etf_dividend_yield)
                tax_on_dividend = calc_personal_tax(etf_dividend, medicare_levy_threshold)
                net_dividend = etf_dividend - tax_on_dividend
                etf_growth_bucket += etf_growth_value + net_dividend
                etf_growth_cost_base += net_dividend # Cost base increases with reinvested net dividends

                # Make sure assets don't go negative before calculations
                super_pension_bucket_you = max(0, super_pension_bucket_you)
                super_pension_bucket_spouse = max(0, super_pension_bucket_spouse)
                etf_cash_bucket = max(0, etf_cash_bucket)
                etf_growth_bucket = max(0, etf_growth_bucket)

                total_super_pension_bucket = super_pension_bucket_you + super_pension_bucket_spouse
                total_super_pension_bucket = max(0, total_super_pension_bucket)

                total_assets = (super_you_acc + super_spouse_acc + super_pension_bucket_you + super_pension_bucket_spouse) + etf_cash_bucket + etf_growth_bucket

                if np.isnan(total_assets):
                    total_assets = 0

                # --- Guardrail Spending Strategy ---
                if retirement_start_assets is None:
                    retirement_start_assets = total_assets

                adj = (total_assets / (retirement_start_assets * inflation_factor)) - 1
                adj = max(-0.2, min(adj, 0.5))
                planned_spending = current_planned_spending_base * (1 + adj) # Adjusted planned spending
                minimum_spending_floor_nom = asfa_modest_real * spending_floor_percentage_of_asfa * inflation_factor
                planned_spending = max(planned_spending, minimum_spending_floor_nom)

                retirement_benchmark_assets = retirement_start_assets * ((1 + inflation_rate)**(year - years_accumulation))
                if total_assets < (0.85 * retirement_benchmark_assets):
                    planned_spending = minimum_spending_floor_nom

                # --- Age Pension Calculation ---
                if survivor_mode:
                    pension_max = pension_nom_single
                    lower_asset_thr = lower_thr_single
                    upper_asset_thr = upper_thr_single
                    income_free = income_free_area_single
                    deeming_threshold_apply = initial_deeming_threshold_single * inflation_factor
                else:
                    pension_max = pension_nom_couple
                    lower_asset_thr = lower_thr_couple
                    upper_asset_thr = upper_thr_couple
                    income_free = income_free_area_couple
                    deeming_threshold_apply = initial_deeming_threshold_couple * inflation_factor

                if total_assets <= lower_asset_thr:
                    pension_assets = pension_max
                elif total_assets >= upper_asset_thr:
                    pension_assets = 0
                else:
                    pension_assets = max(pension_max - ((total_assets - lower_asset_thr) / 1000) * asset_taper_rate, 0)

                deemed_income = 0
                if total_assets > 0:
                    deemed_income_lower_tier = min(total_assets, deeming_threshold_apply) * deeming_rate_lower
                    deemed_income_upper_tier = max(0, total_assets - deeming_threshold_apply) * deeming_rate_upper
                    deemed_income = deemed_income_lower_tier + deemed_income_upper_tier

                assessable_income = deemed_income
                income_over_free_area = max(0, assessable_income - income_free)
                pension_income = max(pension_max - (income_over_free_area * 0.50), 0)
                pension_recv = min(pension_assets, pension_income)

                # --- Bucket Strategy Drawdown and Replenishment Logic ---

                min_rate = get_min_drawdown_rate(current_pension_age)
                min_super_drawdown_required = min_rate * total_super_pension_bucket

                # Total cash outflow target for the year
                provisional_tax_on_dividends = calc_personal_tax(etf_dividend, medicare_levy_threshold)
                total_cash_outflow_target = planned_spending + unexpected_expense + age_care_cost + current_health_cost + sudden_health_cost + provisional_tax_on_dividends

                other_available_income = pension_recv

                # Replenish Cash Bucket first if needed
                cash_bucket_target = planned_spending * years_in_cash_bucket

                if etf_cash_bucket < cash_bucket_target and total_super_pension_bucket > 0:
                    amount_to_replenish_cash = min(cash_bucket_target - etf_cash_bucket, total_super_pension_bucket)
                    etf_cash_bucket += amount_to_replenish_cash

                    if total_super_pension_bucket > 0:
                        ratio_you_replenish = super_pension_bucket_you / total_super_pension_bucket
                        super_pension_bucket_you -= amount_to_replenish_cash * ratio_you_replenish
                        super_pension_bucket_spouse -= amount_to_replenish_cash * (1 - ratio_you_replenish)
                        super_pension_bucket_you = max(0, super_pension_bucket_you)
                        super_pension_bucket_spouse = max(0, super_pension_bucket_spouse)

                if etf_cash_bucket < cash_bucket_target and etf_growth_bucket > 0:
                    amount_to_replenish_cash = min(cash_bucket_target - etf_cash_bucket, etf_growth_bucket)
                    etf_cash_bucket += amount_to_replenish_cash

                    etf_growth_bucket_before_replenish = etf_growth_bucket
                    if etf_growth_bucket_before_replenish > 0:
                        etf_sell_ratio_replenish = amount_to_replenish_cash / etf_growth_bucket_before_replenish
                        etf_growth_cost_base -= etf_growth_cost_base * etf_sell_ratio_replenish
                    etf_growth_bucket -= amount_to_replenish_cash

                # Draw from Buckets for Spending
                cash_from_buckets_needed = max(0, total_cash_outflow_target - other_available_income)

                draw_from_cash_bucket = min(cash_from_buckets_needed, etf_cash_bucket)
                etf_cash_bucket -= draw_from_cash_bucket
                cash_from_buckets_needed -= draw_from_cash_bucket

                if min_super_drawdown_required > 0 and total_super_pension_bucket > 0:
                    additional_super_draw_for_min = min(min_super_drawdown_required, total_super_pension_bucket)
                    draw_super += additional_super_draw_for_min
                    cash_from_buckets_needed = max(0, cash_from_buckets_needed - additional_super_draw_for_min)

                current_total_super_pension_bucket = super_pension_bucket_you + super_pension_bucket_spouse
                if cash_from_buckets_needed > 0 and current_total_super_pension_bucket > 0:
                    draw_from_super_additional = min(cash_from_buckets_needed, current_total_super_pension_bucket)
                    draw_super += draw_from_super_additional
                    cash_from_buckets_needed -= draw_from_super_additional

                draw_from_etf_growth = 0 # Initialize here for current year's direct draw
                if cash_from_buckets_needed > 0 and etf_growth_bucket > 0:
                    draw_from_etf_growth = min(cash_from_buckets_needed, etf_growth_bucket)
                    draw_etf += draw_from_etf_growth # Accumulate draw_etf for reporting
                    cash_from_buckets_needed -= draw_from_etf_growth


                capital_gain = 0

                final_total_outflow_required = planned_spending + unexpected_expense + age_care_cost + current_health_cost + sudden_health_cost + provisional_tax_on_dividends
                current_available_cash = pension_recv + draw_super + draw_from_cash_bucket + additional_etf_growth_draw_from_rem + etf_dividend

                shortfall = max(0, final_total_outflow_required - current_available_cash)

                if shortfall > 0:
                    additional_cash_draw_from_cash_bucket_for_shortfall = min(shortfall, etf_cash_bucket)
                    etf_cash_bucket -= additional_cash_draw_from_cash_bucket_for_shortfall
                    shortfall -= additional_cash_draw_from_cash_bucket_for_shortfall

                    if shortfall > 0:
                        remaining_super_after_draw = total_super_pension_bucket - draw_super # Remaining after initial draw_super
                        additional_super_draw_from_rem = min(shortfall, remaining_super_after_draw)
                        draw_super += additional_super_draw_from_rem
                        shortfall -= additional_super_draw_from_rem

                        if shortfall > 0:
                            remaining_etf_after_draw = etf_growth_bucket - draw_from_etf_growth # Remaining after initial etf draw
                            additional_etf_growth_draw_from_rem = min(shortfall, remaining_etf_after_draw)
                            draw_etf += additional_etf_growth_draw_from_rem
                            shortfall -= additional_etf_growth_draw_from_rem

                actual_draw_from_etf_growth_for_spending = draw_from_etf_growth + additional_etf_growth_draw_from_rem

                capital_gain = 0

                if actual_draw_from_etf_growth_for_spending > 0:
                    etf_growth_bucket_value_pre_final_draw = etf_growth_bucket + actual_draw_from_etf_growth_for_spending
                    etf_growth_cost_base_pre_final_draw = etf_growth_cost_base

                    if etf_growth_bucket_value_pre_final_draw > 0:
                        avg_cost_per_dollar = etf_growth_cost_base_pre_final_draw / etf_growth_bucket_value_pre_final_draw
                        cost_sold = actual_draw_from_etf_growth_for_spending * avg_cost_per_dollar
                        capital_gain = max(actual_draw_from_etf_growth_for_spending - cost_sold, 0)

                discounted_gain = capital_gain * 0.5
                tax = calc_personal_tax(discounted_gain + etf_dividend, medicare_levy_threshold)

                etf_growth_bucket -= actual_draw_from_etf_growth_for_spending
                etf_growth_cost_base -= etf_growth_cost_base * (actual_draw_from_etf_growth_for_spending / (etf_growth_bucket + actual_draw_from_etf_growth_for_spending)) if (etf_growth_bucket + actual_draw_from_etf_growth_for_spending) > 0 else 0

                total_draw_from_super_actual = draw_super

                current_total_super_pension_bucket_before_draw = super_pension_bucket_you + super_pension_bucket_spouse
                if current_total_super_pension_bucket_before_draw > 0:
                    ratio_you = super_pension_bucket_you / current_total_super_pension_bucket_before_draw
                    super_pension_bucket_you -= total_draw_from_super_actual * ratio_you
                    super_pension_bucket_spouse -= total_draw_from_super_actual * (1 - ratio_you)

                super_pension_bucket_you = max(0, super_pension_bucket_you)
                super_pension_bucket_spouse = max(0, super_pension_bucket_spouse)
                etf_cash_bucket = max(0, etf_cash_bucket)
                etf_growth_bucket = max(0, etf_growth_bucket)
                etf_growth_cost_base = max(0, etf_growth_cost_base)

                gross_total_inflow = pension_recv + draw_super + draw_from_cash_bucket + actual_draw_from_etf_growth_for_spending + etf_dividend

                surplus_cash = max(0, gross_total_inflow - final_total_outflow_required)
                etf_cash_bucket += surplus_cash

                total_income = min(gross_total_inflow, final_total_outflow_required)

                cash_from_etf_for_spending = draw_from_cash_bucket + actual_draw_from_etf_growth_for_spending

                percent_pension = (pension_recv / gross_total_inflow) * 100 if gross_total_inflow > 0 else 0
                percent_super = (draw_super / gross_total_inflow) * 100 if gross_total_inflow > 0 else 0
                percent_etf = (cash_from_etf_for_spending / gross_total_inflow) * 100 if gross_total_inflow > 0 else 0

                # Set plotting values for retirement phase
                actual_planned_spending = planned_spending # This is the adjusted planned spending
                actual_current_health_cost = current_health_cost
                actual_sudden_health_cost = sudden_health_cost
                actual_age_care_cost = age_care_cost
                actual_unexpected_expense = unexpected_expense
                actual_asfa = asfa_nom # This is the base ASFA amount


            # Append yearly data
            total_super_for_reporting = super_you_acc + super_spouse_acc + super_pension_bucket_you + super_pension_bucket_spouse
            total_etf_for_reporting = etf_cash_bucket + etf_growth_bucket

            sim_data.append([
                year + 1, total_super_for_reporting,
                total_etf_for_reporting, total_assets,
                pension_assets, pension_income, actual_asfa, actual_planned_spending, pension_recv,
                draw_super, draw_etf, tax, total_income, percent_pension, percent_super, percent_etf,
                lower_asset_thr, upper_asset_thr, total_super_for_reporting, actual_unexpected_expense,
                actual_age_care_cost, actual_current_health_cost, actual_sudden_health_cost,
                cash_from_etf_for_spending
            ])
        all_sims.append(sim_data)

    dfs = [pd.DataFrame(sim, columns=[
        "Yr", "TotalSuper", "ETF", "TotAst",
        "PenAst", "PenInc", "ASFA", "PlannedSpend", "PenRcv",
        "DrawSup", "DrawETF", "Tax", "TotInc", "%Pen", "%Sup", "%ETF",
        "CL_Low", "CL_Up", "SupBal", "Unexp", "AgeCareCost",
        "CurrentHealthCost", "SuddenHealthCost", "CashFromETF_forSpending"
    ]) for sim in all_sims]
    return dfs

# ---- Main Simulation ----
def run_combined_model():
    # To fix NameError for 'years_accumulation', explicitly reference it from the global scope
    years_accumulation_for_plotting = globals()['accumulation_years']

    # Run Trust Simulation
    trust_dfs = simulate_trust_module(sim_years=total_years, child_current_age=child_initial_age)
    trust_summary = pd.concat(trust_dfs).groupby("Yr").quantile([0.1, 0.5, 0.9]).unstack()
    trust_median_df = trust_summary.xs(0.5, level=1, axis=1)
    trust_low_df = trust_summary.xs(0.1, level=1, axis=1)
    trust_high_df = trust_summary.xs(0.9, level=1, axis=1)

    # Run Retirement Simulation
    retirement_dfs = run_retirement_model_bucket(spending_floor_percentage_of_asfa=0.8,
                                                years_in_cash_bucket=2,
                                                years_in_medium_bucket=5)

    retirement_summary = pd.concat(retirement_dfs).groupby("Yr").quantile([0.1, 0.5, 0.9]).unstack()
    retirement_median_df = retirement_summary.xs(0.5, level=1, axis=1)

    # --- Plot Trust Growth ---
    plt.figure(figsize=(12, 6))
    plt.plot(trust_median_df.index, trust_median_df['TrustBalance'], label='Trust Median', color='black')
    plt.fill_between(trust_median_df.index, trust_low_df['TrustBalance'], trust_high_df['TrustBalance'],
                     color='gray', alpha=0.3, label='10th-90th Percentile')
    plt.xlabel('Year')
    plt.ylabel('Trust Balance ($)')
    plt.title('Trust ETF Growth with Dividend Reinvestment (After Tax)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Generate Summary DataFrame ---
    all_sim_df = pd.concat(retirement_dfs)
    summary_df = all_sim_df.groupby('Yr').quantile([0.1, 0.5, 0.9]).unstack()

    # Define columns to display in summary for median values
    median_df_for_plot = summary_df.xs(0.5, level=1, axis=1)
    median_df_for_plot = pd.merge(median_df_for_plot, trust_median_df[['TrustBalance']], left_index=True, right_index=True, how='left')
    columns_to_display_median = ['TotAst', 'TotalSuper', 'ETF', 'TrustBalance', 'PenRcv', 'DrawSup', 'DrawETF', 'TotInc', 'ASFA',
                                 'PlannedSpend', 'CurrentHealthCost', 'SuddenHealthCost', 'AgeCareCost', 'Tax', 'CashFromETF_forSpending','CL_Low', 'CL_Up'] # Added 'CL_Low' and 'CL_Up']

    # Extract median (0.5 quantile) for relevant columns

    print("\n--- Retirement Simulation Summary (Median Values) ---")
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)
    formatted_df = median_df_for_plot[columns_to_display_median].round(0).astype(int)
    print(formatted_df.to_string())
    pd.reset_option('display.width')
    pd.reset_option('display.max_columns')

    # --- ADD THIS SECTION FOR 10TH PERCENTILE ---
    low_df_for_plot = summary_df.xs(0.1, level=1, axis=1)
    low_df_for_plot = pd.merge(low_df_for_plot, trust_low_df[['TrustBalance']], left_index=True, right_index=True, how='left') # Assuming trust_low_df is available from trust simulation
    columns_to_display_low = ['TotAst', 'TotalSuper', 'ETF', 'TrustBalance', 'PenRcv', 'DrawSup', 'DrawETF', 'TotInc', 'ASFA',
                              'PlannedSpend', 'CurrentHealthCost', 'SuddenHealthCost', 'AgeCareCost', 'Tax', 'CashFromETF_forSpending','CL_Low', 'CL_Up'] # Added 'CL_Low' and 'CL_Up']

    print("\n--- Retirement Simulation Summary (10th Percentile Values) ---")
    formatted_low_df = low_df_for_plot[columns_to_display_low].round(0).astype(int)
    print(formatted_low_df.to_string())


    # --- ADD THIS SECTION FOR 90TH PERCENTILE ---
    high_df_for_plot = summary_df.xs(0.9, level=1, axis=1)
    high_df_for_plot = pd.merge(high_df_for_plot, trust_high_df[['TrustBalance']], left_index=True, right_index=True, how='left') # Assuming trust_high_df is available from trust simulation
    columns_to_display_high = ['TotAst', 'TotalSuper', 'ETF', 'TrustBalance', 'PenRcv', 'DrawSup', 'DrawETF', 'TotInc', 'ASFA',
                               'PlannedSpend', 'CurrentHealthCost', 'SuddenHealthCost', 'AgeCareCost', 'Tax', 'CashFromETF_forSpending','CL_Low', 'CL_Up'] # Added 'CL_Low' and 'CL_Up']

    print("\n--- Retirement Simulation Summary (90th Percentile Values) ---")
    formatted_high_df = high_df_for_plot[columns_to_display_high].round(0).astype(int)
    print(formatted_high_df.to_string())


    # Calculate overall success rate
    success_rate = (sum(1 for sim_df in retirement_dfs if sim_df['TotAst'].iloc[-1] > 0) / n_simulations) * 100
    print(f"\nOverall Success Rate (Assets last until end of modeling period): {success_rate:.2f}%")


    # --- Plot 1: Median Asset Balances Over Time ---
    plt.figure(figsize=(12, 6))
    plt.plot(median_df_for_plot.index, median_df_for_plot['TotAst'], label='Median Total Assets', color='blue')
    plt.plot(median_df_for_plot.index, median_df_for_plot['TotalSuper'], label='Median Super Balance', color='green', linestyle='--')
    plt.plot(median_df_for_plot.index, median_df_for_plot['ETF'], label='Median ETF Balance', color='orange', linestyle=':')
    plt.plot(trust_median_df.index, trust_median_df['TrustBalance'], label='Trust Balance', color='purple')

    plt.fill_between(summary_df.index, summary_df[('TotAst', 0.1)], summary_df[('TotAst', 0.9)],
                     color='lightblue', alpha=0.3, label='10th-90th Percentile (Total Assets)')
    plt.xlabel('Year')
    plt.ylabel('Amount ($)')
    plt.title('Median Asset Balances Over Time (Guardrail + Bucket Strategy)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # --- Plot 2: Median Income Sources and Outflows ---
    plt.figure(figsize=(12, 7))
    income_sources_and_outflows_to_plot = ['PenRcv', 'DrawSup', 'CashFromETF_forSpending', 'TotInc', 'ASFA', 'PlannedSpend',
                                           'CurrentHealthCost', 'SuddenHealthCost', 'AgeCareCost']

    for col in income_sources_and_outflows_to_plot:
        plt.plot(median_df_for_plot.index, median_df_for_plot[col], label=col)

    plt.fill_between(summary_df.index, summary_df[('TotInc', 0.1)], summary_df[('TotInc', 0.9)],
                     color='lightcoral', alpha=0.3, label='10th-90th Percentile (Total Income)')

    plt.xlabel('Year')
    plt.ylabel('Amount ($)')
    plt.title('Median Income and Expenses (Guardrail + Bucket Strategy)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Retirement Income Components (Stacked Area Chart) ---
    # Filter data for retirement years only using the correctly referenced variable
    retirement_median_income_df = median_df_for_plot[median_df_for_plot.index > years_accumulation_for_plotting]

    # Stacked area plot for income components
    plt.figure(figsize=(12, 7))
    plt.stackplot(retirement_median_income_df.index,
                  retirement_median_income_df['PenRcv'],
                  retirement_median_income_df['DrawSup'],
                  retirement_median_income_df['CashFromETF_forSpending'],
                  labels=['Age Pension', 'Super Drawdowns', 'ETF Drawdowns'],
                  alpha=0.8)

    plt.plot(retirement_median_income_df.index, retirement_median_income_df['TotInc'],
             color='black', linestyle='--', label='Total Income (Median)')

    plt.xlabel('Year')
    plt.ylabel('Amount ($)')
    plt.title('Median Retirement Income Components Over Time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    return trust_dfs, retirement_dfs

if __name__ == "__main__":
    trust_outputs, retirement_outputs = run_combined_model()