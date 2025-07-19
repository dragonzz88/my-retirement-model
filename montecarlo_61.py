import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility
n_simulations = 3000  # Number of Monte Carlo simulations

def run_retirement_model_bucket(initial_etf_amount=100000,
                                annual_self_contribution_you=20000,
                                asfa_modest_real=48000, # ASFA Modest in real terms (initial year)
                                initial_living_expenses_real=30000, # Living expenses in accumulation phase, real terms
                                spending_floor_percentage_of_asfa=0.6,
                                years_in_cash_bucket=2, # Number of years of expenses to hold in cash bucket
                                years_in_medium_bucket=5): # Number of years of expenses to hold in medium bucket

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
    # Add correlation for cash bucket if it's independently managed
    # For now, let's keep it between super and growth ETF
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

        # Superannuation for retirement will become the "medium-term" bucket conceptually
        # It needs to be managed for drawdowns as well as growth.
        # We'll consider super_you_ret and super_spouse_ret together as the Super Pension Bucket
        super_pension_bucket_you = 0
        super_pension_bucket_spouse = 0

        salary = initial_salary_you
        life_you = int(np.random.normal(90, 5))
        life_spouse = int(np.random.normal(90, 5))
        survivor_mode = False
        retirement_start_assets = None  # Will be set once at the start of retirement
        
        sim_data = []

        for year in range(total_years):
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

            # Random shocks
            unexpected_expense = np.random.choice([0, np.random.randint(10000, 50000)]) if np.random.rand() < 0.1 else 0
            age_care_cost = np.random.randint(10000, 40000) if (year >= years_accumulation + 20 and np.random.rand() < 0.3) else 0

            # Survivorship
            alive_you = age_you < life_you
            alive_spouse = age_spouse < life_spouse
            if not survivor_mode:
                if not alive_you and alive_spouse:
                    # Only transfer super_acc and super_pension_bucket_you if you actually had any
                    if super_you_acc > 0 or super_pension_bucket_you > 0:
                        super_spouse_acc += super_you_acc + super_pension_bucket_you # Transfer super to spouse
                    super_you_acc = super_pension_bucket_you = 0
                    survivor_mode = True
                elif not alive_spouse and alive_you:
                    # Only transfer super_acc and super_pension_bucket_spouse if spouse actually had any
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
                initial_planned_spending = asfa_modest_real * inflation_factor # Use ASFA as initial guide
                
                # Calculate initial cash bucket target
                cash_target = initial_planned_spending * years_in_cash_bucket
                
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
                        # If no super available for initial cash transfer, cash bucket won't be fully filled
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
                current_living_expenses = initial_living_expenses_real * inflation_factor

                tax_on_salary = calc_personal_tax(gross_salary, medicare_levy_threshold)
                # Tax on dividends from ETF growth bucket
                tax_on_dividend = calc_personal_tax(etf_dividend, medicare_levy_threshold)
                net_salary = gross_salary - tax_on_salary
                net_dividend = etf_dividend - tax_on_dividend

                etf_growth_bucket += etf_growth_value + net_dividend # Dividends reinvested into growth bucket
                etf_growth_cost_base += net_dividend # Cost base increases with reinvested dividends

                disposable_income = net_salary - current_living_expenses
                if disposable_income > 0:
                    etf_growth_bucket += disposable_income
                    etf_growth_cost_base += disposable_income

                salary = gross_salary
                total_assets = (super_you_acc + super_spouse_acc) + etf_growth_bucket + etf_cash_bucket
                pension_assets = pension_income = pension_recv = draw_super = draw_etf = tax = 0
                total_income = net_salary + net_dividend + max(disposable_income, 0)
                percent_pension = percent_super = percent_etf = 0
                cash_from_etf_for_spending = 0 # Not used in accumulation but in output

            # --- Retirement Phase Logic ---
            else:
                # Initialize drawdown variables for this year to prevent UnboundLocalError
                draw_from_cash_bucket = 0
                # draw_from_super_additional is not used directly in final calculations of available cash,
                # but draw_super is.
                draw_from_etf_growth = 0
                
                current_planned_spending = asfa_modest_real * inflation_factor # Base for guardrail
                
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
                
                total_assets = (super_you_acc + super_spouse_acc + super_pension_bucket_you + super_pension_bucket_spouse) + etf_cash_bucket + etf_growth_bucket
                
                # Ensure total_assets is not NaN if any component became NaN
                if np.isnan(total_assets):
                    total_assets = 0 # Or handle as an extinction event

                # --- Guardrail Spending Strategy ---
                # Ensure retirement_start_assets is not None before using it
                if retirement_start_assets is None:
                    retirement_start_assets = total_assets # Fallback in case of unexpected None (shouldn't happen with current flow)

                adj = (total_assets / (retirement_start_assets * inflation_factor)) - 1
                adj = max(-0.2, min(adj, 0.5))
                planned_spending = current_planned_spending * (1 + adj)
                minimum_spending_floor_nom = asfa_modest_real * spending_floor_percentage_of_asfa * inflation_factor
                planned_spending = max(planned_spending, minimum_spending_floor_nom)

                retirement_benchmark_assets = retirement_start_assets * ((1 + inflation_rate)**(year - years_accumulation))
                if total_assets < (0.85 * retirement_benchmark_assets):
                    planned_spending = minimum_spending_floor_nom

                # --- Sophisticated Age Pension Calculation ---
                # Determine relevant parameters based on survivor mode
                if survivor_mode:
                    pension_max = pension_nom_single
                    lower_asset_thr = lower_thr_single
                    upper_asset_thr = upper_thr_single
                    income_free = income_free_area_single
                    deeming_threshold_apply = initial_deeming_threshold_single * inflation_factor # Inflate per individual threshold
                else:
                    pension_max = pension_nom_couple
                    lower_asset_thr = lower_thr_couple
                    upper_asset_thr = upper_thr_couple
                    income_free = income_free_area_couple
                    deeming_threshold_apply = initial_deeming_threshold_couple * inflation_factor # Inflate couple threshold

                # Age Pension Asset Test
                # Note: Principal home is exempt from asset test but not explicitly modeled here.
                # Assuming 'total_assets' already excludes the principal home if applicable.
                if total_assets <= lower_asset_thr:
                    pension_assets = pension_max
                elif total_assets >= upper_asset_thr:
                    pension_assets = 0
                else:
                    # For every $1,000 above the lower threshold, pension reduces by asset_taper_rate
                    # (effectively $3 per fortnight, or $78 per year for every $1000).
                    # The asset_taper_rate is per $1000, so we divide the difference by 1000.
                    pension_assets = max(pension_max - ((total_assets - lower_asset_thr) / 1000) * asset_taper_rate, 0)
                
                # Age Pension Income Test (using deeming rates)
                # All financial investments and super in pension phase are deemed.
                # Here, we assume 'total_assets' (which includes ETF and Super Pension) is subject to deeming.
                # More sophisticated would break down assets into financial vs. non-financial.
                
                deemed_income = 0
                if total_assets > 0: # Only deem if there are assets to deem
                    # Deeming calculation based on relevant threshold (single or couple)
                    deemed_income_lower_tier = min(total_assets, deeming_threshold_apply) * deeming_rate_lower
                    deemed_income_upper_tier = max(0, total_assets - deeming_threshold_apply) * deeming_rate_upper
                    deemed_income = deemed_income_lower_tier + deemed_income_upper_tier

                # Assessable income for income test (includes deemed income and other actual income)
                # For now, other actual income is implicitly captured through deemed income from assets,
                # and super drawdowns are not deemed as they are from a pension balance (tax-free in retirement).
                # If there were other actual income sources (e.g., from employment or rental property), they'd be added here.
                assessable_income = deemed_income
                
                # Apply income free area
                income_over_free_area = max(0, assessable_income - income_free)
                
                # Pension reduces by 50 cents for every dollar over the income free area
                pension_income = max(pension_max - (income_over_free_area * 0.50), 0)

                # Final Age Pension received is the lower of the asset test and income test outcomes
                pension_recv = min(pension_assets, pension_income)

                # --- Bucket Strategy Drawdown and Replenishment Logic ---
                draw_super = 0
                draw_etf = 0
                
                total_super_pension_bucket = super_pension_bucket_you + super_pension_bucket_spouse
                # Ensure total_super_pension_bucket is not negative if individual buckets became negative due to large draws
                total_super_pension_bucket = max(0, total_super_pension_bucket)

                # Calculate minimum super drawdown (required by law)
                age_at_retirement = 68
                current_pension_age = age_at_retirement + (year - years_accumulation)
                min_rate = get_min_drawdown_rate(current_pension_age)
                min_super_drawdown_required = min_rate * total_super_pension_bucket

                # Total cash outflow target (before final tax on CG)
                provisional_tax_on_dividends = calc_personal_tax(etf_dividend, medicare_levy_threshold)
                total_cash_outflow_target = planned_spending + unexpected_expense + age_care_cost + provisional_tax_on_dividends

                # Available cash from non-investment sources for spending
                other_available_income = pension_recv

                # --- Replenish Cash Bucket first if needed ---
                cash_bucket_target = planned_spending * years_in_cash_bucket
                
                # Replenish from Super Pension Bucket
                if etf_cash_bucket < cash_bucket_target and total_super_pension_bucket > 0:
                    amount_to_replenish_cash = min(cash_bucket_target - etf_cash_bucket, total_super_pension_bucket)
                    etf_cash_bucket += amount_to_replenish_cash
                    
                    # Deduct proportionally from individual super buckets
                    if total_super_pension_bucket > 0: # Avoid division by zero
                        ratio_you_replenish = super_pension_bucket_you / total_super_pension_bucket
                        super_pension_bucket_you -= amount_to_replenish_cash * ratio_you_replenish
                        super_pension_bucket_spouse -= amount_to_replenish_cash * (1 - ratio_you_replenish)
                        # Ensure no negative balances
                        super_pension_bucket_you = max(0, super_pension_bucket_you)
                        super_pension_bucket_spouse = max(0, super_pension_bucket_spouse)


                # Replenish from ETF Growth Bucket if cash bucket still low and Super is depleted or insufficient
                if etf_cash_bucket < cash_bucket_target and etf_growth_bucket > 0:
                    amount_to_replenish_cash = min(cash_bucket_target - etf_cash_bucket, etf_growth_bucket)
                    etf_cash_bucket += amount_to_replenish_cash
                    
                    # Track cost base for replenishment from growth bucket
                    # Use the current etf_growth_bucket value before replenishment to calculate ratio
                    etf_growth_bucket_before_replenish = etf_growth_bucket
                    if etf_growth_bucket_before_replenish > 0:
                        etf_sell_ratio_replenish = amount_to_replenish_cash / etf_growth_bucket_before_replenish
                        etf_growth_cost_base -= etf_growth_cost_base * etf_sell_ratio_replenish
                    etf_growth_bucket -= amount_to_replenish_cash # Deduct after ratio calculation

                # --- Draw from Buckets for Spending ---
                cash_from_buckets_needed = max(0, total_cash_outflow_target - other_available_income)
                
                # 1. Draw from Cash Bucket (Bucket 1)
                draw_from_cash_bucket = min(cash_from_buckets_needed, etf_cash_bucket)
                etf_cash_bucket -= draw_from_cash_bucket
                cash_from_buckets_needed -= draw_from_cash_bucket

                # 2. Draw minimum required from Super (Bucket 2) and prioritize it if cash bucket is empty
                if min_super_drawdown_required > 0 and total_super_pension_bucket > 0:
                    additional_super_draw_for_min = min(min_super_drawdown_required, total_super_pension_bucket)
                    draw_super += additional_super_draw_for_min
                    cash_from_buckets_needed = max(0, cash_from_buckets_needed - additional_super_draw_for_min)

                # 3. If still needed, draw from Super Pension Bucket (Bucket 2) beyond minimum
                # Recalculate total_super_pension_bucket after min draw for this check
                current_total_super_pension_bucket = super_pension_bucket_you + super_pension_bucket_spouse
                if cash_from_buckets_needed > 0 and current_total_super_pension_bucket > 0: # Check if there's money left in super
                    draw_from_super_additional = min(cash_from_buckets_needed, current_total_super_pension_bucket)
                    draw_super += draw_from_super_additional
                    cash_from_buckets_needed -= draw_from_super_additional

                # 4. If still needed, draw from ETF Growth Bucket (Bucket 3)
                if cash_from_buckets_needed > 0 and etf_growth_bucket > 0:
                    draw_from_etf_growth = min(cash_from_buckets_needed, etf_growth_bucket)
                    draw_etf += draw_from_etf_growth
                    cash_from_buckets_needed -= draw_from_etf_growth

                # Capital Gains Tax on ETF withdrawals for spending
                capital_gain = 0
                # Calculate gain only on the amount drawn *directly* from etf_growth_bucket for spending
                # This is the `draw_from_etf_growth` part from step 4, and potentially `additional_etf_growth_draw_from_rem` later.
                # For simplicity, let's track the *net* amount from growth used for spending
                
                # The total `draw_etf` here is the sum of `draw_from_etf_growth` and any `additional_cash_draw` from the `etf_cash_bucket`
                # and `additional_etf_growth_draw_from_rem`.
                # We need to compute CG only on sales from the `etf_growth_bucket` directly.
                # The replenishment step already adjusted cost base, so this calculation is for the final spending draw.
                # The `actual_draw_from_etf_growth_for_spending` is the best candidate.
                # We will calculate this after the shortfall handling.
                # Tax calculation before final deductions (as it's based on income/gains *before* spend)
                # For now, let's keep it based on `etf_dividend` and `discounted_gain` that will be calculated later
                tax = calc_personal_tax(etf_dividend, medicare_levy_threshold) # Only dividend tax for now

                final_total_outflow_required = planned_spending + unexpected_expense + age_care_cost + tax
                current_available_cash = pension_recv + draw_super + draw_from_cash_bucket + draw_from_etf_growth + net_dividend # Include dividend for gross income
                # Note: `net_dividend` was added to `etf_growth_bucket` already, so it's not a direct cash inflow here for spending.
                # It should be `etf_dividend` for income calculation if it's considered for spending, not `net_dividend`.
                # Let's consider `net_dividend` as part of assets, and if needed for spending, it will be drawn from `etf_growth_bucket` implicitly.
                # For the `current_available_cash`, let's just stick to the actual draws.
                current_available_cash = pension_recv + draw_super + draw_from_cash_bucket + draw_from_etf_growth

                # Check for remaining shortfall after initial draw decisions and re-adjust
                shortfall = max(0, final_total_outflow_required - current_available_cash)
                
                additional_cash_draw_from_cash_bucket_for_shortfall = 0
                additional_super_draw_from_rem = 0
                additional_etf_growth_draw_from_rem = 0

                if shortfall > 0:
                    # Try to cover shortfall from any remaining cash bucket
                    additional_cash_draw_from_cash_bucket_for_shortfall = min(shortfall, etf_cash_bucket)
                    etf_cash_bucket -= additional_cash_draw_from_cash_bucket_for_shortfall
                    shortfall -= additional_cash_draw_from_cash_bucket_for_shortfall
                    # draw_etf += additional_cash_draw_from_cash_bucket_for_shortfall # This is already captured in draw_etf if it was initially from cash bucket.
                                                                                        # If it means 'funds from ETF for spending', it's okay.
                    if shortfall > 0:
                        # Try to cover shortfall from remaining super pension bucket
                        additional_super_draw_from_rem = min(shortfall, total_super_pension_bucket - draw_super) # Use total super available to draw from
                        draw_super += additional_super_draw_from_rem
                        shortfall -= additional_super_draw_from_rem

                        if shortfall > 0:
                            # Try to cover shortfall from remaining ETF growth bucket
                            additional_etf_growth_draw_from_rem = min(shortfall, etf_growth_bucket)
                            draw_etf += additional_etf_growth_draw_from_rem # Add to total ETF drawn for spending
                            shortfall -= additional_etf_growth_draw_from_rem

                # Now calculate the final capital gain and tax based on *actual* sales from growth bucket
                # This needs to be precise.
                # The `draw_etf` here is the sum of `draw_from_etf_growth`
                # and `additional_etf_growth_draw_from_rem`.
                # The actual amount of ETF growth bucket sold for spending (after all adjustments)
                actual_draw_from_etf_growth_for_spending = draw_from_etf_growth + additional_etf_growth_draw_from_rem

                capital_gain = 0
                if actual_draw_from_etf_growth_for_spending > 0:
                    # To calculate capital gain, we need the average cost basis per unit/dollar *before* the sale
                    # The etf_growth_bucket and etf_growth_cost_base are already adjusted by replenishment.
                    # We need the value just *before* these final draws for spending.
                    etf_growth_bucket_value_pre_final_draw = etf_growth_bucket + actual_draw_from_etf_growth_for_spending
                    etf_growth_cost_base_pre_final_draw = etf_growth_cost_base # This cost base is what remains after replenishment draws

                    if etf_growth_bucket_value_pre_final_draw > 0:
                        avg_cost_per_dollar = etf_growth_cost_base_pre_final_draw / etf_growth_bucket_value_pre_final_draw
                        cost_sold = actual_draw_from_etf_growth_for_spending * avg_cost_per_dollar
                        capital_gain = max(actual_draw_from_etf_growth_for_spending - cost_sold, 0)
                
                discounted_gain = capital_gain * 0.5
                tax = calc_personal_tax(discounted_gain + etf_dividend, medicare_levy_threshold) # Recalculate tax with CGT

                # Deduct final draws from assets
                # Only deduct the part coming directly from growth (not from cash bucket, which was already handled)
                etf_growth_bucket -= actual_draw_from_etf_growth_for_spending
                etf_growth_cost_base -= etf_growth_cost_base * (actual_draw_from_etf_growth_for_spending / (etf_growth_bucket + actual_draw_from_etf_growth_for_spending)) if (etf_growth_bucket + actual_draw_from_etf_growth_for_spending) > 0 else 0
                
                # Deduct from individual super pension buckets proportionally
                total_draw_from_super_actual = draw_super
                
                # Ensure we don't divide by zero for super ratios
                current_total_super_pension_bucket_before_draw = super_pension_bucket_you + super_pension_bucket_spouse
                if current_total_super_pension_bucket_before_draw > 0:
                    ratio_you = super_pension_bucket_you / current_total_super_pension_bucket_before_draw
                    super_pension_bucket_you -= total_draw_from_super_actual * ratio_you
                    super_pension_bucket_spouse -= total_draw_from_super_actual * (1 - ratio_you)
                
                # Ensure no negative balances after drawing
                super_pension_bucket_you = max(0, super_pension_bucket_you)
                super_pension_bucket_spouse = max(0, super_pension_bucket_spouse)
                etf_cash_bucket = max(0, etf_cash_bucket)
                etf_growth_bucket = max(0, etf_growth_bucket)
                etf_growth_cost_base = max(0, etf_growth_cost_base) # Cost base can't be negative


                # Final calculation of total income received (what was actually successfully drawn for spending)
                gross_total_inflow = pension_recv + draw_super + draw_from_cash_bucket + actual_draw_from_etf_growth_for_spending + etf_dividend # Include dividend for gross income
                
                # Any surplus cash from pension/super draws (that wasn't spent) can be put into cash bucket
                surplus_cash = max(0, gross_total_inflow - final_total_outflow_required)
                etf_cash_bucket += surplus_cash # Reinvest any surplus back into the cash bucket

                total_income = min(gross_total_inflow, final_total_outflow_required) # What was actually available/spent
                
                # For reporting, cash from ETF is from both cash bucket and growth bucket
                cash_from_etf_for_spending = draw_from_cash_bucket + actual_draw_from_etf_growth_for_spending

                # Recalculate percentages based on the actual total inflow
                percent_pension = (pension_recv / gross_total_inflow) * 100 if gross_total_inflow > 0 else 0
                percent_super = (draw_super / gross_total_inflow) * 100 if gross_total_inflow > 0 else 0
                percent_etf = (cash_from_etf_for_spending / gross_total_inflow) * 100 if gross_total_inflow > 0 else 0


            # Append yearly data
            total_super_for_reporting = super_you_acc + super_spouse_acc + super_pension_bucket_you + super_pension_bucket_spouse
            total_etf_for_reporting = etf_cash_bucket + etf_growth_bucket
            
            sim_data.append([
                year + 1, total_super_for_reporting, # Only one total super value
                total_etf_for_reporting, total_assets,
                pension_assets, pension_income, asfa_nom, pension_recv,
                draw_super, draw_etf, tax, total_income, percent_pension, percent_super, percent_etf,
                lower_asset_thr, upper_asset_thr, total_super_for_reporting, unexpected_expense, age_care_cost,
                cash_from_etf_for_spending
            ])
        all_sims.append(sim_data)

    dfs = [pd.DataFrame(sim, columns=[
        "Yr", "TotalSuper", "ETF", "TotAst",
        "PenAst", "PenInc", "ASFA", "PenRcv",
        "DrawSup", "DrawETF", "Tax", "TotInc", "%Pen", "%Sup", "%ETF",
        "CL_Low", "CL_Up", "SupBal", "Unexp", "CareCost",
        "CashFromETF_forSpending"
    ]) for sim in all_sims]
    return dfs

# ---- Main Execution ----
if __name__ == "__main__":
    all_simulation_dataframes = run_retirement_model_bucket(spending_floor_percentage_of_asfa=0.8,
                                                            years_in_cash_bucket=2,
                                                            years_in_medium_bucket=5) # Parameters for bucket strategy

    # Set pandas display option to suppress scientific notation for floats
    pd.options.display.float_format = '{:,.0f}'.format # Display floats with 0 decimal places and comma separator

    # Display year-by-year data for the first simulation
    print("--- Year-by-Year Data for First Simulation (Total Super) ---")
    print(all_simulation_dataframes[0].to_string()) # .to_string() for full display without truncation
    print("\n")

    # Reset option to default if other parts of code rely on it or for general good practice
    # pd.reset_option('display.float_format')

    summary = pd.concat(all_simulation_dataframes).groupby("Yr").quantile([0.1, 0.5, 0.9]).unstack()
    median_df_for_plot = summary.xs(0.5, level=1, axis=1)

    # Plot 1: Asset Balances
    plt.figure(figsize=(12, 7))
    assets_to_plot = ['TotalSuper', 'ETF', 'TotAst']
    for col in assets_to_plot:
        plt.plot(median_df_for_plot.index, median_df_for_plot[col], label=col)
    plt.fill_between(summary.index, summary[('TotAst', 0.1)], summary[('TotAst', 0.9)], color='gray', alpha=0.2, label='10th-90th Percentile (TotAst)')
    plt.xlabel('Year')
    plt.ylabel('Amount ($)')
    plt.title('Median Asset Balances Over Time (Guardrail + Bucket Strategy)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Income Sources and Outflows
    plt.figure(figsize=(12, 7))
    income_sources_and_outflows_to_plot = ['PenRcv', 'DrawSup', 'DrawETF', 'TotInc', 'ASFA']
    for col in income_sources_and_outflows_to_plot:
        plt.plot(median_df_for_plot.index, median_df_for_plot[col], label=col)
    plt.fill_between(summary.index, summary[('TotInc', 0.1)], summary[('TotInc', 0.9)], color='gray', alpha=0.2, label='10th-90th Percentile (TotInc)')
    plt.xlabel('Year')
    plt.ylabel('Amount ($)')
    plt.title('Median Income and Expenses (Guardrail + Bucket Strategy)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: Percentage of Income Sources
    plt.figure(3, figsize=(12, 6))
    percent_sources_to_plot = ['%Pen', '%Sup', '%ETF']
    for col in percent_sources_to_plot:
        plt.plot(median_df_for_plot.index, median_df_for_plot[col], label=col)
    plt.xlabel('Year')
    plt.ylabel('Percentage (%)')
    plt.title('Median Percentage of Income Sources (Guardrail + Bucket Strategy)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()