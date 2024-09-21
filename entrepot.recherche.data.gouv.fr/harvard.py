import marimo

__generated_with = "0.8.15"
app = marimo.App(width="medium")


@app.cell
def __():
    # Python Standard Library
    import json
    import math
    import os
    import time

    # Third-Party Libraries
    import joblib
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests

    # Marimo
    import marimo as mo
    return joblib, json, math, mo, np, os, pd, plt, requests, time


@app.cell
def __(requests):
    URL = "https://dataverse.harvard.edu/api/search"
    session = requests.Session()
    response = session.get(URL, params={"q": "*"})
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.5",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "DNT": "1",
        "Host": "dataverse.harvard.edu",
        "Pragma": "no-cache",
        "Priority": "u=0, i",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-GPC": "1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
    }
    print(response.status_code, response.text)
    print(response.headers)
    # OK, given the "x-amz-waf-action": "challenge" stuff, I guess that there is some anto-bot
    # protection, see e.g. https://docs.aws.amazon.com/waf/latest/APIReference/API_ChallengeAction.html

    return URL, headers, response, session


@app.cell
def __(response):
    response.text
    return


@app.cell
def __(joblib, json, mo, os, requests):
    def fetch(
        URL="https://dataverse.harvard.edu/api/search",
        N=1000,  # number of items on a single search API request
    ):
        params = {"q": "*"}
        response = requests.get(URL, params=params)
        total_count = response.json()["data"]["total_count"]

        @joblib.delayed
        def get_chunck(start):
            params = {"q": "*", "start": start, "per_page": N}
            response = requests.get(URL, params=params)
            return response.json()["data"]["items"]

        parallel = joblib.Parallel(n_jobs=os.cpu_count(), return_as="generator")

        starts = range(0, total_count, N)

        output_generator = parallel(get_chunck(start) for start in starts)

        items = []
        for i, datum in mo.status.progress_bar(
            collection=zip(starts, output_generator),
            total=len(starts),
            title="https://entrepot.recherche.data.gouv.fr",
            subtitle='Data saved as "harvard.json"',
        ):
            items.extend(datum)

        with open("items.json", mode="wt", encoding="utf-8") as output:
            json.dump(items, output)
        return items


    fetch_button = mo.ui.button(
        value=False, on_click=lambda value: True, label="fetch"
    )

    fetch()
    return fetch, fetch_button


@app.cell
def __(mo):
    mo.md(r"""-----""")
    return


@app.cell
def __(pd):
    df = pd.read_json("harvard.json")
    df
    return df,


@app.cell
def __(df):
    df.type.value_counts()
    return


@app.cell
def __(df):
    df_dataverse = df[df.type=="dataverse"]
    df_dataverse
    return df_dataverse,


@app.cell
def __(df_dataverse, pd):
    def def_ratios(df):
        n = len(df)
        ratios = [(key, (n - pd.isnull(df[key]).sum()) / n) for key in df.columns]
        ratios = sorted(ratios, key=lambda x: 1-x[1])
        return dict(ratios)
    dv_ratios = def_ratios(df_dataverse)
    dv_used_keys = [key for (key, ratio) in dv_ratios.items() if ratio > 0]
    df_dv = df_dataverse[dv_used_keys]
    df_dv
    return def_ratios, df_dv, dv_ratios, dv_used_keys


@app.cell
def __(df_dv):
    omics_dv = df_dv[df_dv.name == "Omics Dataverse"]
    omics_dv
    return omics_dv,


@app.cell
def __(mo, omics_dv):
    mo.md(omics_dv.description[0])
    return


@app.cell
def __(mo, omics_dv):
    mo.md(f"url: <{omics_dv.url[0]}>")
    return


@app.cell
def __(mo):
    mo.md(r"""-----""")
    return


@app.cell
def __(df):
    df_dataset = df[df.type=="dataset"]
    df_dataset
    return df_dataset,


@app.cell
def __(def_ratios, df_dataset):
    ds_ratios = def_ratios(df_dataset)
    ds_ratios
    return ds_ratios,


@app.cell
def __(ds_ratios):
    ds_used_keys = [key for (key, ratio) in ds_ratios.items() if ratio > 0]
    ds_used_keys
    return ds_used_keys,


@app.cell
def __(df_dataset, ds_used_keys):
    df_ds = df_dataset[ds_used_keys]
    df_ds
    return df_ds,


@app.cell
def __(mo):
    mo.md("""------""")
    return


@app.cell
def __(df):
    df_datafile = df[df.type=="file"]
    df_datafile
    return df_datafile,


@app.cell
def __(def_ratios, df_datafile):
    df_ratios = def_ratios(df_datafile)
    df_ratios
    return df_ratios,


@app.cell
def __(df_ratios):
    df_used_keys = [key for (key, ratio) in df_ratios.items() if ratio > 0]
    df_used_keys
    return df_used_keys,


@app.cell
def __(df_datafile, df_used_keys):
    df_df = df_datafile[df_used_keys]
    df_df
    return df_df,


@app.cell
def __(df_df, mo):
    mo.md(f"Total size: {round(df_df.size_in_bytes.sum() / 1024**3)} GiB")
    return


@app.cell
def __(df_df):
    datatypes = df_df.file_content_type.value_counts()
    datatypes.to_dict()
    return datatypes,


@app.cell
def __(df_df, np, pd):
    df_size = pd.DataFrame(df_df.groupby("file_content_type")["size_in_bytes"].sum())
    df_size = df_size.sort_values("size_in_bytes", ascending=False)
    df_size["size_in_GiB"] = np.round(df_size.size_in_bytes / 1024**3, 1)
    df_size = df_size[["size_in_GiB"]]
    df_size
    return df_size,


@app.cell
def __(df_df):
    df_images = df_df[df_df.file_content_type.str.startswith("image")] # VERY slow
    df_images
    return df_images,


@app.cell
def __(df_images, mo):
    mo.md(f"Images: {round(df_images.size_in_bytes.sum() / 1024**3)} GiB")
    return


@app.cell
def __(df_df):
    df_df_sorted = df_df.sort_values("size_in_bytes", ascending=False)
    df_df_sorted["size_in_GiB"] = round(df_df_sorted["size_in_bytes"] / 1024**3)
    df_df_sorted[["name", "description", "dataset_name", "url", "file_type", "size_in_GiB"]]
    return df_df_sorted,


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


if __name__ == "__main__":
    app.run()
