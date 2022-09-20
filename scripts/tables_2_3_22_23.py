#!/usr/bin/env python
# coding: utf-8

##############################################################################
# Compute Tables 2, 3, 22 and 23 of the HPOBench paper
##############################################################################

import re
import numpy as np
import pandas as pd
import scipy.stats

whitespace_pattern = re.compile(r'\s+')


def parse_latex(latex):
    """Parse results table for further computation of p-values."""
    
    for i in range(len(latex)):
        latex[i] = latex[i].replace(b'\\underline'.decode('ascii'), '')
        latex[i] = latex[i].replace(b'\\textbf'.decode('ascii'), '')
        latex[i] = latex[i].replace(b'textbf'.decode('ascii'), '')
        latex[i] = latex[i].replace('\\\\\\n', '')
        latex[i] = latex[i].replace('{', '').replace('}', '').replace('&', '').replace('\\', '').strip()
        latex[i] = latex[i].split()
        
    results = pd.DataFrame(latex)
    results.columns = results.iloc[0].to_list()
    results = results.drop(0, axis=0)
    if 'optimizers' in results.columns:
        results = results.set_index('optimizers')
    elif 'optimizer' in results.columns:
        results = results.set_index('optimizer')
    else:
        raise ValueError()
    results[results == '-'] = np.NaN
    results = results.astype(float)
    return results


concats_community_benchmarks = {}
for fraction in ('1', '10', '100'):
    results_mf_sf = []
    for fidelity in ('mf', 'sf'):
        results = []
        
        fname = 'results_tables/community_benchmarks/all_%s_%s.tex' % (fidelity, fraction)
        try:
            with open(fname, 'rt') as fh:
                lines = fh.readlines()
                line_toprule = -1
                line_bottomrule = -1
                lines = [l for l in lines if not re.sub(whitespace_pattern, '', l).startswith('%')]
                lines = [l for l in lines if len(re.sub(whitespace_pattern, '', l)) > 0]
                lines = [line for line in lines if 'midrule' not in line]
                
                for i in range(len(lines)):
                    if 'toprule' in lines[i]:
                        assert line_toprule == -1, line_toprule
                        line_toprule = i
                    elif 'end{tabular}' in lines[i]:
                        assert line_bottomrule == -1, line_bottomrule
                        line_bottomrule = i
                assert line_toprule != -1, line_toprule
                assert line_bottomrule != -1, line_bottomrule
                
                #lines = ('\\n'.join(lines[line_toprule+1:line_bottomrule-1]))
                lines = lines[line_toprule+1:line_bottomrule]
                res = parse_latex(lines)
                assert res.shape[0] == 22, (res.shape, res.index)
                results.append(res)
        except:
            print(fname)
            raise
        results_mf_sf.append(pd.concat(results, axis=0))
    results_mf_sf = pd.concat(results_mf_sf, axis=1)
    #print(results_mf_sf.shape, results_mf_sf.columns, results_mf_sf.index)
    concats_community_benchmarks[fraction] = results_mf_sf


concats_new = {}
for fraction in ('0.01', '0.1', '1.0'):
    results_mf_sf = []
    for fidelity in ('mf', 'sf'):
        results = []
        for bench in ('lr', 'svm', 'xgb', 'rf', 'nn'):
            fname = 'results_tables/new_benchmarks/%s_all_%s_%s.tex' % (bench, fidelity, fraction)
            try:
                with open(fname, 'rt') as fh:
                    lines = fh.readlines()
                    line_toprule = -1
                    line_bottomrule = -1
                    lines = [l for l in lines if not re.sub(whitespace_pattern, '', l).startswith('%')]
                    lines = [l for l in lines if len(re.sub(whitespace_pattern, '', l)) > 0]
                    for i in range(len(lines)):
                        if 'toprule' in lines[i]:
                            assert line_toprule == -1
                            line_toprule = i
                        elif 'bottomrule' in lines[i]:
                            assert line_bottomrule == -1
                            line_bottomrule = i
                    assert line_toprule != -1
                    assert line_bottomrule != -1
                    lines = [line for line in lines if 'midrule' not in line]
                    #lines = ('\\n'.join(lines[line_toprule+1:line_bottomrule-1]))
                    lines = lines[line_toprule+1:line_bottomrule-1]
                    res = parse_latex(lines)
                    results.append(res)
            except:
                print(fname)
                raise
        results_mf_sf.append(pd.concat(results, axis=0))
    results_mf_sf = pd.concat(results_mf_sf, axis=1)
    #print(results_mf_sf.shape, results_mf_sf.columns, results_mf_sf.index)
    concats_new[fraction] = results_mf_sf


def compute_sign_test(baseline, competitor, concat):
    """Compute the sign test according to Demsar, 2006.
    
    We use the sign test because different benchmarks measure different metrics,
    making them incommensurable."""
    baseline_wins_against_competitor = np.sum(concat[baseline].to_numpy() < concat[competitor].to_numpy())
    competitor_wins_against_baseline = np.sum(concat[competitor].to_numpy() < concat[baseline].to_numpy())
    baseline_and_competitor_tie = np.sum(concat[baseline].to_numpy() == concat[competitor].to_numpy())
    remainder = int(baseline_and_competitor_tie / 2)

    wtl = '%d/%d/%d' % (competitor_wins_against_baseline, baseline_and_competitor_tie, baseline_wins_against_competitor)

    p = scipy.stats.binom_test(
        (baseline_wins_against_competitor + remainder, competitor_wins_against_baseline + remainder),
        alternative='less'
    )
    return wtl, p

##############################################################################
# RQ 1 for cummunity benchmarks - Table 2

# blackbox optimizers
wtls = []
ps = []
competitors = ('de', 'smacbo', 'smacrf', 'hebo', 'tpe')
for competitor in competitors:
    wtl, p = compute_sign_test('random', competitor, concats_community_benchmarks['100'])
    wtls.append(wtl)
    if p < (0.05 / len(competitors)):
        p = '$\mathbf{\\underline{%.5f}}$' % p
    elif p < 0.05:
        p = '$\\underline{%.5f}$' % p
    else:
        p = '$%.5f$' % p
    ps.append(p)
print("RQ1 - Table 2 - Blackbox optimizers")
print(' & '.join(['$%s$' % wtl for wtl in wtls]))
print(' & '.join(ps))
print()

# multi-fidelity optimizers
wtls = []
ps = []
competitors = ('bohb', 'dehb', 'smachb', 'dragonfly')
for competitor in competitors:
    wtl, p = compute_sign_test('hb', competitor, concats_community_benchmarks['100'])
    wtls.append(wtl)
    if p < (0.05 / len(competitors)):
        p = '$\mathbf{\\underline{%.5f}}$' % p
    elif p < 0.05:
        p = '$\\underline{%.5f}$' % p
    else:
        p = '$%.5f$' % p
    ps.append(p)
print("RQ1 - Table 2 - multi-fidelity optimizers")
print(' & '.join(['$%s$' % wtl for wtl in wtls]))
print(' & '.join(ps))
print()

##############################################################################
# RQ 1 for the new benchmarks - Table 22

# blackbox optimizers
wtls = []
ps = []
competitors = ('de', 'smacbo', 'smacrf', 'hebo', 'tpe')
for competitor in competitors:
    wtl, p = compute_sign_test('random', competitor, concats_new['1.0'])
    wtls.append(wtl)
    if p < (0.05 / len(competitors)):
        p = '$\mathbf{\\underline{%.5f}}$' % p
    elif p < 0.05:
        p = '$\\underline{%.5f}$' % p
    else:
        p = '$%.5f$' % p
    ps.append(p)
print("RQ1 - Table 22 - blackbox optimizers")
print(' & '.join(['$%s$' % wtl for wtl in wtls]))
print(' & '.join(ps))
print()

# multi-fidelity optimizers
wtls = []
ps = []
competitors = ('bohb', 'dehb', 'smachb', 'dragonfly')
for competitor in competitors:
    wtl, p = compute_sign_test('hb', competitor, concats_new['1.0'])
    wtls.append(wtl)
    if p < (0.05 / len(competitors)):
        p = '$\mathbf{\\underline{%.5f}}$' % p
    elif p < 0.05:
        p = '$\\underline{%.5f}$' % p
    else:
        p = '$%.5f$' % p
    ps.append(p)
print("RQ1 - Table 22 - multi-fidelity optimizers")
print(' & '.join(['$%s$' % wtl for wtl in wtls]))
print(' & '.join(ps))
print()

##############################################################################
# RQ 2 for community benchmarks - Table 3

print("RQ2 - Table 3")
for fraction in ('100', '10', '1',  ):
    wtls = []
    ps = []
    for baseline, competitor in (('random', 'hb'), ('de', 'dehb'), ('tpe', 'bohb'), ('smacrf', 'smachb')):
        wtl, p = compute_sign_test(baseline, competitor, concats_community_benchmarks[fraction])
        wtls.append(wtl)
        if p < 0.05:
            p = '$\mathbf{%.5f}$' % p
        else:
            p = '$%.5f$' % p
        ps.append(p)
    print(' & '.join(['$%s$' % wtl for wtl in wtls]))
    print(' & '.join(ps))
print()


##############################################################################
# RQ 2 for new benchmarks - Table 23

print("RQ2 - Table 23")
for fraction in ('1.0', '0.1', '0.01'):
    wtls = []
    ps = []
    for baseline, competitor in (('random', 'hb'), ('de', 'dehb'), ('tpe', 'bohb'), ('smacrf', 'smachb')):
        
        # renaming a few columns
        concats = concats_new[fraction]
        for to_replace, replace_with in (
            ('randomsearch', 'random'),
            ('hpbandster_hb_eta_3', 'hb'),
            ('hpbandster_tpe', 'tpe'),
            ('hpbandster_bohb_eta_3', 'bohb'),
            ('smac_sf', 'smacrf'),
            ('smac_hb_eta_3', 'smachb')
        ):
            if to_replace in concats:
                concats[replace_with] = concats[to_replace]
                concats = concats.drop(to_replace, axis=1)
        
        try:
            wtl, p = compute_sign_test(baseline, competitor, concats)
            wtls.append(wtl)
            if p < 0.05:
                p = '$\mathbf{%.5f}$' % p
            else:
                p = '$%.5f$' % p
            ps.append(p)
        except KeyError:
            print(baseline, competitor, concat.columns, fraction)
            raise
    print(' & '.join(['$%s$' % wtl for wtl in wtls]))
    print(' & '.join(ps))
print()
