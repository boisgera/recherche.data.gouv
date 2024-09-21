import marimo

__generated_with = "0.8.15"
app = marimo.App(width="medium")


@app.cell
def __():
    # Python Standard Library
    import json
    import math
    import os
    import pathlib

    # Third-Party Libraries
    import joblib
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests

    # Marimo
    import marimo as mo
    return joblib, json, math, mo, np, os, pathlib, pd, plt, requests


@app.cell
def __(joblib, json, mo, os, pathlib, requests):
    # Dataverse search API doc: https://guides.dataverse.org/en/latest/api/search.html

    def fetch(
        dataverse="entrepot.recherche.data.gouv.fr", # test: demo.dataverse.org
    ):
        # Nota: experimentally, the data is sorted by date, we use that property
        # Load local data (if any)
        data = []
        output = f"{dataverse}.json"
        if pathlib.Path(output).exists():
            with open(output, "rt", encoding="utf-8") as file:
                data = json.load(file)
                
        URL = f"https://{dataverse}/api/search"
        N=1000  # max number of items on a single search API request
        params = {"q": "*"}
        response = requests.get(URL, params=params)
        total_count = response.json()["data"]["total_count"]

        @joblib.delayed
        def get_chunck(start):
            params = {"q": "*", "start": start, "per_page": N}
            response = requests.get(URL, params=params)
            return response.json()["data"]["items"]

        parallel = joblib.Parallel(n_jobs=os.cpu_count(), return_as="generator")

        starts = range(len(data), total_count, N)
        output_generator = parallel(get_chunck(start) for start in starts)

        for i, datum in mo.status.progress_bar(
            collection=zip(starts, output_generator),
            total=max(len(starts), 1), # harmless workaround
            title=dataverse,
            subtitle=f'Data saved as "{dataverse}.json"',
        ):
            data.extend(datum)

        with open(f"{dataverse}.json", mode="wt", encoding="utf-8") as output:
            json.dump(data, output)

    fetch()
    return fetch,


@app.cell
def __(fetch):
    fetch("demo.dataverse.org")
    return


@app.cell
def __(pd):
    pd.read_json("entrepot.recherche.data.gouv.fr.json")
    return


@app.cell
def __(pd):
    pd.read_json("entrepot.recherche.data.gouv.fr.json").type.value_counts()
    return


@app.cell
def __(pd):
    # TODO:
    # - [X] split raw into dataverse, dataset and (datafile), clean up each table of the unused stuff,
    # - [ ] DRY the code
    # - [ ] Drop the type (redundant) from each table
    #    
    def drop_unused_columns(df):
        keys = [k for k in df.columns if (~df[k].isnull()).any()] 
        return df[keys]

    def split_raw(df):
        dv = df[df.type=="dataverse"]
        dv = dv.reset_index(drop=True)
        dv = drop_unused_columns(dv)

        ds = df[df.type=="dataset"]
        ds = ds.reset_index(drop=True)
        ds = drop_unused_columns(ds)

        df_ = df[df.type=="file"]
        df_ = df_.reset_index(drop=True)
        df_ = drop_unused_columns(df_)
        
        return dv, ds, df_

    dv, ds, df = split_raw(pd.read_json("entrepot.recherche.data.gouv.fr.json"))
    return df, drop_unused_columns, ds, dv, split_raw


@app.cell
def __(dv):
    dv
    return


@app.cell
def __(ds):
    ds
    return


@app.cell
def __(df):
    df
    return


@app.cell
def __(mo):
    mo.md("## Data Files")
    return


@app.cell
def __(df, mo):
    mo.md(f"Total size: {round(df.size_in_bytes.sum() / 1024**3)} GiB")
    return


@app.cell
def __(df):
    datatypes = df.file_content_type.value_counts()
    datatypes.to_dict()
    return datatypes,


@app.cell
def __(df, np, pd):
    df_size = pd.DataFrame(df.groupby("file_content_type")["size_in_bytes"].sum())
    df_size = df_size.sort_values("size_in_bytes", ascending=False)
    df_size["size_in_GiB"] = np.round(df_size.size_in_bytes / 1024**3, 1)
    df_size = df_size[["size_in_GiB"]]
    df_size
    return df_size,


@app.cell
def __(df):
    df_by_size = df.sort_values("size_in_bytes", ascending=False).reset_index(drop=True)
    df_by_size["size_in_GiB"] = round(df_by_size["size_in_bytes"] / 1024**3, 1)
    df_by_size
    #df_df_sorted[["name", "description", "dataset_name", "url", "file_type", "size_in_GiB"]]
    return df_by_size,


@app.cell
def __():
    # TODO:
    # Analyze datatypes, group some of them (archives, images, etc), make pie chart, etc.
    return


@app.cell
def __(mo):
    mo.md(
        """
        TODO:

          - Organize / make explicit the dataverse / dataset / datafile relationships.
          - How? To begin with, identify unique ids for dvs, ds and df (the same?).
            - identifier for dataverse
            - global_id (DOI) for datasets (up: identifier_of_dataverse)
            - file_id or better file_persistent_id for files (up : dataset_persistent_id)
          - Have dv -> [ds] and ds -> [df] dictionaries
        """
    )
    return


@app.cell
def __(df_df, df_ds, df_dv):
    df_dv.loc[:, "datasets"] = None

    for i, entry in df_ds.iterrows():
        if parent_id := entry.identifier_of_dataverse:
            dv_entry = df_dv[df_dv["identifier"] == parent_id]
            if dv_entry.iloc[0]["datasets"] is None:
                df_dv.at[dv_entry.index[0], "datasets"] = []
            df_dv.at[dv_entry.index[0], "datasets"].append(entry.global_id)

    df_ds.loc[:, "datafiles"] = None

    for i, file_entry in df_df.iterrows():
        if parent_id := file_entry.dataset_persistent_id:
            ds_entry = df_ds[df_ds["global_id"] == parent_id]
            if ds_entry.iloc[0]["datafiles"] is None:
                df_ds.at[ds_entry.index[0], "datafiles"] = []
            df_ds.at[ds_entry.index[0], "datafiles"].append(file_entry.file_persistent_id)
    return ds_entry, dv_entry, entry, file_entry, i, parent_id


@app.cell
def __(df_dv, mo, pd):
    undefined_ds = pd.isnull(df_dv["datasets"]).mean()
    mo.md(f"""
    About {round(undefined_ds*100.0, 1)}% of dataverses have no associated datasets. 

    **TODO:** investigate the phantom dataverses! Study the populated ones, in terms of number of data[sets/files], file size, etc.

    **TODO:** add `size_in_bytes` in datasets and dataverses dataframes (will be a nice summary and will allow to sort this stuff)

    """)
    return undefined_ds,


@app.cell
def __(df_ds, mo, pd):
    assert all(pd.isnull(df_ds["identifier_of_dataverse"]) == False)
    mo.md("Every dataset is registered into a dataverse 👍")
    return


@app.cell
def __(df_ds, mo, pd):
    undefined_df = pd.isnull(df_ds["datafiles"]).mean()
    mo.md(f"""About {round(undefined_df*100.0, 1)}% of datasets have no associated datafiles.
    """)
    return undefined_df,


@app.cell
def __(df_df, mo, pd):
    assert (pd.isnull(df_df["dataset_persistent_id"]).any() == False)
    mo.md("Every datafile is registered into a datasets 👍")
    return


@app.cell
def __(df_df, mo):
    _size_bytes = df_df["size_in_bytes"].sum()
    mo.md(f"{round(_size_bytes / 1024**3, 1)}GiB")
    return


@app.cell
def __(df_df):
    df_df["dataset_persistent_id"]
    return


@app.cell
def __(df_df, df_ds):
    # Dispatch datafiles sizes into datasets
    # 🪲 Buggy: all the size is allocated to the same dataset entry!
    df_ds["size_in_bytes"] = 0

    _s = set()

    for _, _file_entry in df_df.iterrows():
        dataset_id = _file_entry.dataset_persistent_id
        _s.add(dataset_id)
        _selection = df_ds[df_ds["global_id"]==dataset_id]
        _index = _selection.index[0]
        # print(_file_entry["size_in_bytes"])
        df_ds.at[_index, "size_in_bytes"] += _file_entry["size_in_bytes"]

    _s
    return dataset_id,


@app.cell
def __(df_ds, mo):
    _size_bytes = df_ds["size_in_bytes"].sum()
    mo.md(f"{round(_size_bytes / 1024**3, 1)}GiB")
    return


@app.cell
def __(df_ds):
    df_ds.sort_values("size_in_bytes", ascending=False)
    return


@app.cell
def __(df_ds, df_dv):
    df_dv["size_in_bytes"] = 0

    for _, _dataset_entry in df_ds.iterrows():
        dataverse_id = _dataset_entry.identifier_of_dataverse
        _selection = df_dv[df_dv["identifier"]==dataverse_id]
        _index = _selection.index[0]
        # print(_file_entry["size_in_bytes"])
        df_dv.at[_index, "size_in_bytes"] += _dataset_entry["size_in_bytes"]
    return dataverse_id,


@app.cell
def __(df_dv, mo):
    _size_bytes = df_dv["size_in_bytes"].sum()
    mo.md(f"{round(_size_bytes / 1024**3, 1)}GiB")
    return


@app.cell
def __(df_dv):
    df_dv.sort_values("size_in_bytes", ascending=False)
    return


@app.cell
def __(df_df, np, plt):
    plt.plot(np.array(df_df.sort_values("size_in_bytes", ascending=False).size_in_bytes) / 1024**3)
    ax = plt.gca()
    ax.set_yscale("log")
    plt.yticks([1, 1/1024, 1/1024**2], ["1 GiB", "1 MiB", "1 KiB"])
    plt.grid(True)
    plt.ylabel("File size")
    plt.xlabel("# file")
    return ax,


@app.cell
def __(df_ds, np, plt):
    plt.plot(np.array(df_ds.sort_values("size_in_bytes", ascending=False).size_in_bytes) / 1024**3)
    _ax = plt.gca()
    _ax.set_yscale("log")
    plt.yticks([1, 1/1024, 1/1024**2], ["1 GiB", "1 MiB", "1 KiB"])
    plt.grid(True)
    plt.ylabel("File size")
    plt.xlabel("# dataset")
    return


@app.cell
def __(df_dv, np, plt):
    plt.plot(np.array(df_dv.sort_values("size_in_bytes", ascending=False).size_in_bytes) / 1024**3)
    _ax = plt.gca()
    _ax.set_yscale("log")
    plt.yticks([1, 1/1024, 1/1024**2], ["1 GiB", "1 MiB", "1 KiB"])
    plt.grid(True)
    plt.ylabel("File size")
    plt.xlabel("# dataverse")
    return


@app.cell
def __(df_dv):
    df_dv[df_dv["size_in_bytes"] >= 1024**3].sort_values("size_in_bytes", ascending=False)
    return


@app.cell
def __(df_dv):
    dv_generique = df_dv[df_dv["identifier"] == "espacegenerique"]
    dv_generique
    return dv_generique,


@app.cell
def __(dv_generique):
    print(dv_generique.iloc[0].description)
    return


@app.cell
def __(dv_generique, mo):
    mo.md(f"Volume données génériques: {round(dv_generique.iloc[0].size_in_bytes/1024**3, 1)} GiB")
    return


@app.cell
def __(mo):
    mo.md("""Plusieurs dataverses par 'espace institutionnel'??? Ex ici: <https://entrepot.recherche.data.gouv.fr/dataverse/univ-cotedazur?q=&types=dataverses&sort=dateSort&order=desc&page=1>""")
    return


@app.cell
def __(mo):
    mo.md("""API access to files is simpler with the API than the site (AFAICT). I WANT TO KNOW WHERE THE FILES ARE HOSTED.""")
    return


@app.cell
def __(df_df):
    df_df_ts = df_df.sort_values("published_at")
    return df_df_ts,


@app.cell
def __(df_df, df_df_ts, np):
    df_df_ts["cum_size_in_bytes"] = None
    df_df_ts["cum_size_in_bytes"] = np.array(df_df["size_in_bytes"].cumsum())
    return


@app.cell
def __(df_df_ts):
    df_df_ts
    return


@app.cell
def __(df_df_ts):
    _ts = df_df_ts["published_at"].iloc[0]
    _ts
    return


@app.cell
def __(df_df_ts, np, plt):
    sizes = np.array(df_df_ts["cum_size_in_bytes"] / 1024**3)

    tss = [ts for ts in df_df_ts["published_at"]]
    dates = list([ts.to_numpy() for ts in tss])
    plt.ylabel("size in GiB")
    plt.plot(dates, sizes)
    return dates, sizes, tss


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
