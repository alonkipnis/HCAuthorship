from scipy.stats import (distributions)
import pandas as pd
import numpy as np



def _unequal_var_ttest_denom(v1, n1, v2, n2):
    vn1 = v1 / n1
    vn2 = v2 / n2
    with np.errstate(divide='ignore', invalid='ignore'):
        df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

    # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
    # Hence it doesn't matter what df is as long as it's not NaN.
    df = np.where(np.isnan(df), 1, df)
    denom = np.sqrt(vn1 + vn2)
    return df, denom

def _ttest_ind_from_stats(mean1, mean2, denom, df, alternative):

    d = mean1 - mean2
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(d, denom)
    t, prob = _ttest_finish(df, t, alternative)

    return (t, prob)

def _ttest_finish(df, t, alternative):
    """Common code between all 3 t-test functions."""
    if alternative == 'less':
        prob = distributions.t.cdf(t, df)
    elif alternative == 'greater':
        prob = distributions.t.sf(t, df)
    elif alternative == 'two-sided':
        prob = 2 * distributions.t.sf(np.abs(t), df)
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")

    if t.ndim == 0:
        t = t[()]

    return t, prob


table = pd.read_csv("~/Data/Gutenberg/HCAuthorship/gutenberg/all_results.csv")
table['accuracy'] = table['accuracy'].apply(lambda x : np.round(x,3)) 
table['std'] = table['std'].apply(lambda x : np.round(x,4)) 
table['dataset'] = table['dataset'].str.extract(r"([0-9]+)").astype(int)
print(table.pivot(index='clf_name', columns = 'dataset', values = ['std', 'accuracy']))


pvals = []

print()
clf1 = 'freq_table_HC'
clf2 = 'freq_table_HC_org'
print(f"Comparing {clf1} and {clf2}:")
for vs in [250, 1000, 3000] :
    std1 = table[(table.dataset == vs) & (table.clf_name == clf1)]['std'].values[0]
    std2 = table[(table.dataset == vs) & (table.clf_name == clf2)]['std'].values[0]
    mean1 = table[(table.dataset == vs) & (table.clf_name == clf1)]['accuracy'].values[0]
    mean2 = table[(table.dataset == vs) & (table.clf_name == clf2)]['accuracy'].values[0]

    df, denom = _unequal_var_ttest_denom(std1**2, 10, std2**2, 10)
    _, pval = _ttest_ind_from_stats(mean1, mean2, denom, df ,alternative = 'greater')
    pvals += [pval]
    print(f"Vocab size = {vs}, Pval = {pval}")

print("Fishser's combinatin test:")
print(distributions.chi2.sf(-2*np.log(pvals).sum(), df=6))