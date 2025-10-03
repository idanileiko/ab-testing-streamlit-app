import streamlit as st
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="A/B Testing Analysis for Continuous Variables",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 A/B Testing Statistical Analysis for Continuous Variables")
st.write("Paste your data below (one value per line, or comma/space separated)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sample 1")
    data1_input = st.text_area("Enter data for Sample 1:", height=200, key="data1")
    
with col2:
    st.subheader("Sample 2")
    data2_input = st.text_area("Enter data for Sample 2:", height=200, key="data2")

# Option for equal variance assumption
assume_equal_var = st.checkbox("Assume equal variances (Student's t-test)", value=True)
if not assume_equal_var:
    st.caption("Welch's t-test will be used (does not assume equal variances)")

alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)

if st.button("Run T-Test", type="primary"):
    try:
        # Parse data
        def parse_data(text):
            # Replace common separators with spaces
            text = text.replace(',', ' ').replace('\n', ' ').replace('\t', ' ')
            # Split and convert to float
            values = [float(x) for x in text.split() if x.strip()]
            return np.array(values)
        
        data1 = parse_data(data1_input)
        data2 = parse_data(data2_input)
        
        if len(data1) < 2 or len(data2) < 2:
            st.error("Each sample needs at least 2 data points")
        else:
            # Run t-test
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=assume_equal_var)
            
            # Calculate confidence interval for difference in means
            mean_diff = np.mean(data1) - np.mean(data2)
            se_diff = np.sqrt(np.var(data1, ddof=1)/len(data1) + np.var(data2, ddof=1)/len(data2))
            
            if assume_equal_var:
                df = len(data1) + len(data2) - 2
            else:
                # Welch-Satterthwaite equation
                v1 = np.var(data1, ddof=1) / len(data1)
                v2 = np.var(data2, ddof=1) / len(data2)
                df = (v1 + v2)**2 / (v1**2/(len(data1)-1) + v2**2/(len(data2)-1))
            
            t_crit = stats.t.ppf(1 - alpha/2, df)
            ci_lower = mean_diff - t_crit * se_diff
            ci_upper = mean_diff + t_crit * se_diff
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data1)-1)*np.var(data1, ddof=1) + 
                                  (len(data2)-1)*np.var(data2, ddof=1)) / (len(data1) + len(data2) - 2))
            cohens_d = mean_diff / pooled_std

            if abs(cohens_d) < 0.2:
                effect_interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            # Results & Interpretation
            st.markdown("### Takeaways:")
            
            if p_value < alpha:
                st.success("**There IS a statistically significant difference between the two groups.**")
                st.write(f"Sample 1's average ({np.mean(data1):.2f}) is statistically different from Sample 2's average ({np.mean(data2):.2f}).")
                
                difference_direction = "higher" if mean_diff > 0 else "lower"
                percent_diff = (mean_diff / np.mean(data2)) * 100
                st.write(f"Sample 1 is **{abs(mean_diff):.2f} points {difference_direction}** than Sample 2 (a **{abs(percent_diff):.1f}%** difference).")
                
                st.write(f"The probability this difference happened by random chance is only **{p_value*100:.2f}%** (very unlikely).")
                
                # Practical significance
                if abs(cohens_d) < 0.2:
                    st.warning("**However**, the difference is **tiny** (effect size is negligible).")
                elif abs(cohens_d) < 0.5:
                    st.info("The difference is **small** but real. Consider whether this size of difference matters for your situation.")
                elif abs(cohens_d) < 0.8:
                    st.success("The difference is **medium-sized** and likely meaningful in practice.")
                else:
                    st.success("The difference is **large** and definitely meaningful in practice.")
                    
            else:
                st.error("**No significant difference detected.**")
                st.write(f"Sample 1's average ({np.mean(data1):.2f}) and Sample 2's average ({np.mean(data2):.2f}) are not statistically different.")
                
                st.write(f"The difference you see ({abs(mean_diff):.2f} points) could easily happen by random chance (probability: **{p_value*100:.1f}%**).")
                
                st.write("**Important:** This doesn't prove they're identical, just that you don't have enough evidence to say they're different.")
            
            # Technical details in expander
            with st.expander("📊 Technical Details (for the stats nerds)"):
                st.write(f"**t-statistic:** {t_stat:.4f}")
                st.write(f"**Degrees of freedom:** {df:.1f}")
                st.write(f"**p-value:** {p_value:.4f}")
                st.write(f"**Cohen's d:** {cohens_d:.4f} (effect size is **{effect_interpretation}**)")
                st.write(f"**Test type:** {'Student\'s t-test (equal variances)' if assume_equal_var else 'Welch\'s t-test (unequal variances)'}")
                st.write(f"**Significance level:** {alpha}")
                st.write(f"{int((1-alpha)*100)}% Confidence Interval", f"[{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Plot distributions
            st.subheader("Distribution Comparison")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram with shared bins across both samples
            all_data = np.concatenate([data1, data2])
            bin_edges = np.linspace(all_data.min(), all_data.max(), 20)
            
            ax1.hist(data1, bins=bin_edges, alpha=0.5, label='Sample 1', color='#1f77b4', edgecolor='black')
            ax1.hist(data2, bins=bin_edges, alpha=0.5, label='Sample 2', color='#ff7f0e', edgecolor='black')
            ax1.axvline(np.mean(data1), color='#1f77b4', linestyle='--', linewidth=2, label='Sample 1 Mean')
            ax1.axvline(np.mean(data2), color='#ff7f0e', linestyle='--', linewidth=2, label='Sample 2 Mean')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Overlapping Histograms')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plots
            box_data = [data1, data2]
            bp = ax2.boxplot(box_data, labels=['Sample 1', 'Sample 2'], patch_artist=True,
                           medianprops=dict(color='black', linewidth=2))
            bp['boxes'][0].set_facecolor('#1f77b4')
            bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][1].set_facecolor('#ff7f0e')
            bp['boxes'][1].set_alpha(0.6)
            ax2.set_ylabel('Value')
            ax2.set_title('Box Plots (showing spread and outliers)')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)

            # Calculate descriptive statistics
            st.subheader("Descriptive Statistics")
            
            stats_df = pd.DataFrame({
                'Sample 1': [len(data1), np.mean(data1), np.std(data1, ddof=1), 
                            np.min(data1), np.max(data1)],
                'Sample 2': [len(data2), np.mean(data2), np.std(data2, ddof=1),
                            np.min(data2), np.max(data2)]
            }, index=['n', 'Mean', 'Std Dev', 'Min', 'Max'])
            
            st.dataframe(stats_df.style.format("{:.4f}"))
            
    except ValueError as e:
        st.error(f"Error parsing data: {e}")
        st.info("Make sure your data contains only numbers, separated by spaces, commas, or line breaks")
    except Exception as e:
        st.error(f"Error: {e}")

# Instructions
with st.expander("How to use this app"):
    st.write("""
    1. **Paste your data** into the two text boxes (Sample 1 and Sample 2)
    2. Data can be formatted as:
       - One number per line
       - Comma-separated: `1.2, 3.4, 5.6`
       - Space-separated: `1.2 3.4 5.6`
    3. **Choose variance assumption:**
       - Equal variances (Student's t-test): Use when samples have similar spread
       - Unequal variances (Welch's t-test): Safer default, especially with different sample sizes
    4. **Set significance level** (typically 0.05)
    5. Click **Run T-Test**
    
    **What you get:**
    - Descriptive statistics for both samples
    - t-statistic and p-value
    - Confidence interval for the difference in means
    - Clear interpretation of results
    - Effect size (Cohen's d) to assess practical significance
    """)
