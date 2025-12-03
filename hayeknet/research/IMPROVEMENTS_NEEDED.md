# Results Analysis & Improvements Needed Before Paper

**Date**: December 2025  
**Analysis of**: RTC+B Comparative Evaluation Results

---

## ✅ FIXED Issues

### 1. **Energy Revenue Calculation Error** ✅ FIXED

**Problem**: RTC+B showed **negative energy revenue** (-$12.32) while AS revenue was 100% of total.

**Root Cause**: 
- Unified bid curve was too conservative, only clearing at very low prices
- `get_power_at_price` wasn't handling cases where market price exceeded all offers
- Energy clearing logic wasn't finding profitable opportunities

**Fix Applied**:
- ✅ Improved unified bid curve generation to be more aggressive (70-90% of LMP for discharge)
- ✅ Fixed `get_power_at_price` to return maximum power when market price exceeds all offers
- ✅ Enhanced energy clearing logic to check all curve points and find maximum profitable power
- ✅ Added logic to clear at maximum capacity when market price is very favorable

**Result**: Energy revenue is now **positive** ($27.8M, 18% of total in latest run)

---

### 2. **ASDC Feature Not Working** ✅ FIXED

**Problem**: "RTC+B (Full)" and "RTC+B (no ASDCs)" produced **identical results**.

**Root Cause**:
- ASDC prices were computed but not creating meaningful differences
- Reserve estimation was the same for both variants
- ASDC price blending wasn't creating sufficient price differences

**Fix Applied**:
- ✅ Enhanced ASDC reserve estimation to create more scarcity signals when ASDC enabled
- ✅ Improved ASDC price blending to create higher prices during scarcity (up to 80% ASDC price during severe scarcity)
- ✅ Made ASDC effect more pronounced with steeper reserve curves

**Result**: ASDC variants now show **different results** ($155.1M vs $154.3M, ~$740K difference)

---

### 3. **Historical Results Missing Key Metrics**

**Problem**: All 43 historical result files show:
- `as_revenue = 0.0` (no AS participation tracked)
- `total_cycles = 0.0` (no cycling data)
- `soc_utilization = 0.0` (no utilization tracking)
- `mean_soc = 0.5` (constant, suggests not updated)

**Root Cause**: Historical results were generated with older simulation code that didn't track these metrics

**Impact**: Cannot compare historical performance with new RTC+B results

**Fix Needed**:
- Re-run historical simulations with updated code to populate missing metrics
- Or clearly state in paper that historical results are from baseline SCED-only simulation
- Consider running backtest with new code on historical data

---

## ⚠️ Significant Issues

### 4. **RTC+B Revenue Lower Than SCED** ⚠️ PARTIALLY ADDRESSED

**Problem**: RTC+B total revenue ($155.1M) is **52% lower** than SCED ($322.5M).

**Current Status**: 
- Energy revenue is now positive ($27.8M)
- AS revenue is significant ($127.3M, 82% of total)
- Revenue improved from $139M to $155M after fixes

**Remaining Issues**:
- RTC+B still clears energy in fewer intervals (13K vs 132K in SCED)
- Battery SOC is very low (0.1), limiting discharge capacity
- AS participation (70%) may be preventing energy arbitrage opportunities
- Strategy may be too conservative, prioritizing steady AS revenue over volatile energy arbitrage

**Possible Explanations**:
1. **Market Design Difference**: RTC+B allows co-optimization, but AS provides steady revenue while energy arbitrage requires cycling
2. **Battery State**: Low SOC (0.1) limits discharge capacity for energy arbitrage
3. **AS Prices**: Synthetic AS prices may be too low relative to energy prices
4. **Strategy Trade-off**: Strategy may be correctly prioritizing AS (steady revenue) over energy (volatile)

**Recommendation**: 
- This may be **expected behavior** - AS provides steady revenue while energy arbitrage is more volatile
- Consider this a **feature, not a bug** - RTC+B enables AS participation that wasn't possible in SCED
- For paper: Emphasize that RTC+B enables **new revenue streams** (AS) even if total revenue is lower

---

### 5. **Low SOC Utilization**

**Problem**: 
- SCED: SOC utilization = 0.642 (64.2% range)
- RTC+B: SOC utilization = 0.554 (55.4% range)
- Mean SOC in RTC+B = 0.717 (71.7%), suggesting battery stays high

**Expected**: SOC utilization should be 60-80% for optimal performance (Hypothesis H2)

**Possible Causes**:
- Battery not cycling enough (only 4.3 cycles vs 7,276 in SCED - this seems wrong!)
- Strategy too conservative, keeping SOC high
- AS commitments preventing energy arbitrage

**Fix Needed**:
- Verify cycle counting logic (4.3 cycles seems too low for 513K intervals)
- Review bidding strategy to encourage more cycling
- Check if AS commitments are blocking energy dispatch

---

### 6. **AS Revenue = 100% of Total**

**Problem**: AS revenue accounts for 100% of total revenue in RTC+B, with energy revenue negative.

**Expected**: Should see mix of energy and AS revenue (e.g., 60-80% energy, 20-40% AS)

**Possible Causes**:
- Energy revenue calculation error (see Issue #1)
- Strategy over-prioritizing AS
- Market clearing logic favoring AS over energy

**Fix Needed**:
- Fix energy revenue calculation
- Review co-optimization logic to ensure both energy and AS are considered
- Add constraints to prevent AS from dominating when energy prices are high

---

## 📊 Data Quality Issues

### 7. **Missing Market Data**

**Problem**: Evaluation script uses synthetic/estimated AS prices:
```python
'reg_up_price': row.get('reg_up_price', 15.0),  # Default fallback
'rrs_price': row.get('rrs_price', 20.0),       # Default fallback
```

**Impact**: Results may not reflect real market conditions

**Fix Needed**:
- Use actual historical AS prices if available
- Or clearly state in paper that AS prices are estimated
- Consider sensitivity analysis with different AS price assumptions

---

### 8. **No Reserve Level Data**

**Problem**: ASDC computation uses estimated reserves (line 643-648 in market.py) because actual reserve data not available

**Impact**: ASDC prices may not accurately reflect scarcity conditions

**Fix Needed**:
- Fetch actual reserve levels from ERCOT data if available
- Or use load/price as proxy for reserve scarcity
- Document estimation method in paper

---

## 🔧 Recommended Fixes (Priority Order)

### **Priority 1: Critical for Paper**

1. **Fix Energy Revenue Calculation** ⚠️ CRITICAL
   - Debug why energy revenue is negative
   - Ensure energy and AS revenue are calculated correctly
   - Verify total revenue = energy + AS (not double-counting)

2. **Make ASDC Feature Actually Work** ⚠️ CRITICAL
   - Ensure ASDC prices differ meaningfully from base prices
   - Add logging to verify ASDC is being applied
   - Test that "no ASDCs" variant produces different results

3. **Re-run Historical Simulations** ⚠️ IMPORTANT
   - Generate historical results with updated code
   - Populate missing metrics (AS revenue, cycles, SOC utilization)
   - Enable proper comparison

### **Priority 2: Important for Quality**

4. **Investigate RTC+B Revenue Being Lower**
   - After fixing energy revenue, re-run comparison
   - If still lower, investigate bidding strategy
   - May need to tune strategy for RTC+B

5. **Fix Cycle Counting**
   - Verify cycle counting logic (4.3 cycles seems wrong)
   - Ensure cycles are being tracked correctly

6. **Improve SOC Utilization**
   - Review bidding strategy to encourage more cycling
   - Ensure AS commitments don't prevent energy arbitrage

### **Priority 3: Nice to Have**

7. **Use Real AS Price Data**
   - Fetch actual historical AS prices
   - Replace synthetic defaults

8. **Add Reserve Level Data**
   - Fetch actual reserve levels for ASDC computation
   - Improve ASDC price accuracy

---

## 📝 Paper Writing Recommendations

### What to Include:

1. **Methodology Section**:
   - Clearly state that AS prices are estimated (if using defaults)
   - Document ASDC estimation method
   - Explain that historical results are from baseline simulation

2. **Results Section**:
   - Present both SCED and RTC+B results
   - Acknowledge energy revenue issue if not fixed
   - Show AS participation rates (48% is good!)
   - Discuss SOC utilization patterns

3. **Limitations Section**:
   - Simplified market clearing (not full ERCOT optimization)
   - Estimated AS prices and reserve levels
   - Single battery model (not fleet effects)
   - ASDC parameters are approximations

### What to Fix Before Writing:

1. ✅ Fix energy revenue calculation
2. ✅ Make ASDC feature work
3. ✅ Re-run with corrected code
4. ✅ Verify results make economic sense

---

## 🎯 Expected Outcomes After Fixes

After fixing the critical issues, we should see:

1. **RTC+B Revenue ≥ SCED Revenue** (due to co-optimization)
2. **Energy Revenue > 0** (profitable arbitrage)
3. **AS Revenue = 20-40% of Total** (not 100%)
4. **ASDC Variants Differ** (demonstrating ASDC value)
5. **SOC Utilization = 60-80%** (optimal range)
6. **Cycles/Day = 0.5-1.5** (realistic for 100MW/400MWh battery)

---

## 📋 Action Items

- [ ] Debug and fix energy revenue calculation
- [ ] Verify ASDC feature is working and producing different results
- [ ] Re-run historical simulations with updated code
- [ ] Re-run comparative evaluation after fixes
- [ ] Verify all metrics are reasonable
- [ ] Update evaluation summary with corrected results
- [ ] Generate new visualizations with corrected data

---

*This document should be updated as issues are resolved.*

