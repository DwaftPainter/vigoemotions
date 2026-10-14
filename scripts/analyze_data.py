"""Generate aggregate EDA, fixed split IDs, and train-only dependencies."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vigo-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, LABELS, load_data, targets, style_masks, write_json, environment


def main():
    frames, excluded = load_data()
    tables, figures = ROOT/'reports/tables', ROOT/'reports/figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (ROOT/'splits').mkdir(exist_ok=True)
    excluded[['id']].to_csv(ROOT/'splits/excluded_train.csv', index=False)
    summary, lengths, cardinalities, styles = [], [], [], []
    freq = pd.DataFrame({'label_id': range(27), 'label': LABELS})
    for split, df in frames.items():
        df[['id']].to_csv(ROOT/'splits'/f'{split}.csv', index=False)
        y = targets(df)
        freq[f'{split}_count'] = y.sum(axis=0)
        freq[f'{split}_prevalence'] = y.mean(axis=0)
        card = y.sum(axis=1)
        summary.append(dict(split=split, samples=len(df),
            cardinality_mean=card.mean(), cardinality_max=int(card.max()),
            neutral_only=int((card == 0).sum()),
            neutral_mixed=int(df.label_ids.map(lambda v: 27 in v and len(set(v)) > 1).sum()),
            repeated_label_rows=int(df.label_ids.map(lambda v: len(v) != len(set(v))).sum())))
        for k, count in zip(*np.unique(card, return_counts=True)):
            cardinalities.append(dict(split=split, cardinality=k, count=count))
        for name, values in [('whitespace_tokens', df.normalized.str.split().str.len()),
                             ('characters', df.normalized.str.len())]:
            lengths.append(dict(split=split, unit=name, mean=values.mean(),
                median=values.median(), p95=values.quantile(.95), p99=values.quantile(.99),
                minimum=values.min(), maximum=values.max()))
        for name, mask in style_masks(df).items():
            styles.append(dict(split=split, style=name, count=int(mask.sum()), proportion=mask.mean()))
    # Relative tiers from train only: nine labels per tier, stable label-ID tie break.
    ranking = freq.sort_values(['train_count', 'label_id'], ascending=[False, True]).index
    freq['tier'] = 'tail'
    freq.loc[ranking[:9], 'tier'] = 'head'
    freq.loc[ranking[9:18], 'tier'] = 'mid'
    freq['val_drift_pp'] = 100*(freq.val_prevalence-freq.train_prevalence)
    freq['test_drift_pp'] = 100*(freq.test_prevalence-freq.train_prevalence)
    freq.to_csv(tables/'label_frequency.csv', index=False)
    pd.DataFrame(summary).to_csv(tables/'dataset_summary.csv', index=False)
    pd.DataFrame(lengths).to_csv(tables/'text_lengths.csv', index=False)
    pd.DataFrame(cardinalities).to_csv(tables/'cardinality.csv', index=False)
    pd.DataFrame(styles).to_csv(tables/'text_styles.csv', index=False)
    y = targets(frames['train'])
    co = y.T @ y
    corr = np.corrcoef(y, rowvar=False)
    for name, matrix in [('cooccurrence', co), ('correlation', corr)]:
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(tables/f'train_{name}.csv')
        fig, ax = plt.subplots(figsize=(11, 9))
        image = ax.imshow(matrix, cmap='RdBu_r' if name == 'correlation' else 'Blues',
                          **({'vmin': -1, 'vmax': 1} if name == 'correlation' else {}))
        ax.set_xticks(range(27), LABELS, rotation=90, fontsize=8)
        ax.set_yticks(range(27), LABELS, fontsize=8)
        ax.set_title(f'Train-only label {name} (n={len(y):,})')
        fig.colorbar(image, ax=ax); fig.tight_layout()
        fig.savefig(figures/f'train_{name}.png', dpi=150); plt.close(fig)
    pairs = [dict(label_a=LABELS[a], label_b=LABELS[b], count=int(co[a,b]),
                  phi=float(corr[a,b])) for a in range(27) for b in range(a+1,27)]
    pd.DataFrame(pairs).sort_values('count', ascending=False).to_csv(tables/'train_top_pairs.csv', index=False)
    ordered = freq.sort_values('train_count')
    fig, ax = plt.subplots(figsize=(10,8))
    ax.barh(ordered.label, ordered.train_count,
            color=ordered.tier.map({'head':'#167d9a','mid':'#e9a23b','tail':'#ce5863'}))
    ax.set(xlabel='Positive training instances', title='Training frequency tiers: 9 head / 9 mid / 9 tail labels')
    fig.tight_layout(); fig.savefig(figures/'label_frequency.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for split, df in frames.items():
        for ax, values, title in zip(axes,
                [targets(df).sum(axis=1), df.normalized.str.split().str.len(), df.normalized.str.len()],
                ['Emotion cardinality |Y|', 'Whitespace tokens', 'Characters']):
            ax.hist(values, bins=np.arange(11)-.5 if ax is axes[0] else 40,
                    alpha=.4, density=True, label=split)
            ax.set(title=title, ylabel='Density')
    axes[2].legend(); fig.tight_layout(); fig.savefig(figures/'distributions.png', dpi=150); plt.close(fig)
    write_json(ROOT/'reports/eda_environment.json', environment())
    print(pd.DataFrame(summary).to_string(index=False))
    print('Excluded training IDs:', excluded.id.tolist())


if __name__ == '__main__':
    main()
