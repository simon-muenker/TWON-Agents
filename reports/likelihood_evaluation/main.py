import json

import pandas
import seaborn

PATH_PREFIX: str = "data/reply_likelihood.en"

data = (
    pandas.concat(
        {
            sample: pandas.json_normalize(json.loads(open(f"{PATH_PREFIX}.{sample}.json").read()))
            for sample in ["paper", "retry_1", "retry_2", "retry_3"]
        }
    )
    .T
)

(
    data
    .agg(["mean", "std"], axis="columns")
    .round(3)
    .to_csv("reports/reply_likelihood.en.mean_std.csv")
)


agg_data = (
    data
    .set_index(
        data.index.str.split(".", expand=True)
        .rename(("", "split", "class", "metric"))
    )
    .droplevel(level=0, axis=0)
    .reset_index()
    .droplevel(level=1, axis=1)
    .pipe(lambda _df: _df[_df["metric"].isin(["precision", "recall", "f1-score"])])
)
agg_data.to_csv("reports/reply_likelihood.en.agg.csv")

print(agg_data.melt(id_vars=["split", "class", "metric"]))

grid = seaborn.FacetGrid(
    agg_data.melt(id_vars=["split", "class", "metric"]),
    col="class",
    row="metric",
    hue="split",
    ylim=(0.960, 0.990),
    sharey=True
)

grid.map_dataframe(seaborn.pointplot, x="variable", y="value", dodge=True)
grid.add_legend()

grid.savefig("reports/reply_likelihood.en.agg.pdf")