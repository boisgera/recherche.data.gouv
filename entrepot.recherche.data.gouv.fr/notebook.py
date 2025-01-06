import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    # Python Standard Library
    import json
    import math
    import os
    import pathlib

    # Third-Party Libraries
    import altair as alt
    import joblib
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests

    # Marimo
    import marimo as mo
    return (
        alt,
        joblib,
        json,
        math,
        mo,
        np,
        os,
        pa,
        pathlib,
        pd,
        plt,
        pq,
        requests,
    )


@app.cell
def _():
    DATAVERSES = [
        "demo.dataverse.org",
        "demo.recherche.data.gouv.fr",
        "entrepot.recherche.data.gouv.fr"
        # NOTA: the harvard official repo is protected from access
    ]
    return (DATAVERSES,)


@app.cell
def _(DATAVERSES, joblib, json, mo, os, pathlib, requests):
    # Dataverse search API doc: https://guides.dataverse.org/en/latest/api/search.html

    def fetch(
        dataverse="entrepot.recherche.data.gouv.fr",  # test: demo.dataverse.org
    ):
        # Nota: experimentally, the data is sorted by date, we could use that property
        # Question: is the existing data immutable? I am afraid not (version update?)
        # so we may still need to fetch all the existing data (which we do not do ATM).

        # Load local data (if any)
        data = []
        output = f"data/{dataverse}.json"
        if pathlib.Path(output).exists():
            with open(output, "rt", encoding="utf-8") as file:
                data = json.load(file)

        URL = f"https://{dataverse}/api/search"
        N = 1000  # max number of items on a single search API request
        params = {"q": "*"}
        response = requests.get(URL, params=params)
        total_count = response.json()["data"]["total_count"]

        @joblib.delayed
        def get_chunck(start):
            params = {"q": "*", "start": start, "per_page": N}
            response = requests.get(URL, params=params)
            return response.json()["data"]["items"]

        starts = range(len(data), total_count, N)
        parallel = joblib.Parallel(
            n_jobs=max(1, min(len(starts), os.cpu_count())),
            return_as="generator",
        )

        output_generator = parallel(get_chunck(start) for start in starts)

        for i, datum in mo.status.progress_bar(
            collection=zip(starts, output_generator),
            total=max(len(starts), 1),  # harmless workaround
            title=dataverse,
            subtitle=f"{output}",
        ):
            data.extend(datum)

        with open(output, mode="wt", encoding="utf-8") as output:
            json.dump(data, output)

    for _dv in DATAVERSES:
        fetch(_dv)

    fetch_done = True
    return fetch, fetch_done


@app.cell
def _(fetch_done, pd):
    fetch_done

    master = pd.read_json("data/entrepot.recherche.data.gouv.fr.json")
    master.type.value_counts()
    return (master,)


@app.cell
def _(master):
    def drop_unused_columns(df):
        keys = [k for k in df.columns if (~df[k].isnull()).any()] 
        return df[keys]

    def split_raw(dataframe):
        types = ["dataverse", "dataset", "file"]
        dataframes = []
        for type_ in types:
            df = dataframe[dataframe.type==type_]
            df = df.reset_index(drop=True)
            df = drop_unused_columns(df)
            df = df.drop("type", axis=1)
            dataframes.append(df)
        return tuple(dataframes)

    dv, ds, df = split_raw(master)
    return df, drop_unused_columns, ds, dv, split_raw


@app.cell
def _(dv):
    dv
    return


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    _df = df.copy(deep=True)
    _df = _df[["file_type", "file_content_type", "size_in_bytes"]].copy()

    #def renamer(name):
    #    if name == "file_type":
    #        return "type"
    #    elif name == "file_content_type":
    #        return "MIME type"
    #    elif name == "size_in_bytes":
    #        return "size (B)"
    #    else:        
    #        return name

    #_df = _df.rename(columns=renamer)
    _df["count"] = 1
    _df["count_in_percents"] = 100.0 * _df["count"] / len(_df)
    _df["size_in_gibibytes"] = _df["size_in_bytes"] / 1024**3
    _df["size_in_percents"] = 100.0 * _df["size_in_bytes"] / _df["size_in_bytes"].sum()

    def acc(types):
        return sorted(set(list(types)))

    _df = _df.groupby(["file_type"], as_index=False, dropna=True).agg({
        "file_content_type": acc,
        "count": "sum",
        "count_in_percents": "sum",
        "size_in_gibibytes": "sum",
        "size_in_percents": "sum",
        "size_in_bytes": "sum"
    })

    _df = _df.sort_values(by="size_in_percents", ascending=False, na_position="last")
    # _df = _df[_df["size (%)"] >= 1.0]
    _df["count_en_percents"] = _df["count_in_percents"].round(1)
    _df["size_in_percents"] = _df["size_in_percents"].round(1)
    _df["size_in_gibibytes"] = _df["size_in_gibibytes"].round()
    _df = _df.sort_values("count", ascending=False)
    by_type =_df.reset_index(drop=True)
    by_type
    return acc, by_type


@app.cell
def _(alt, df, mo):
    _df = df.copy(deep=True)
    _df["year"] = _df["published_at"].dt.year
    _df = _df[["year", "size_in_bytes"]]
    _df = _df.groupby("year", as_index=False)
    _df = _df.sum()
    _df
    _df["size_in_gibibytes"] = (_df["size_in_bytes"] / 1024**3).round(1)
    _df = _df.drop("size_in_bytes", axis=1)
    _df

    mo.hstack([
        _df,
        mo.ui.altair_chart(alt.Chart(_df).mark_bar().encode(
            x="year:O",
            y='size_in_gibibytes',
        ))
    ])
    return


@app.cell
def _(alt, df, mo):
    _df = df.copy(deep=True)
    _df["year"] = _df["published_at"].dt.year 
    _df["count"] = 1
    _df = _df[["year", "count"]]
    _df = _df.groupby("year", as_index=False)
    _df = _df.sum()
    _df
    #_df["size (GiB)"] = (_df["size_in_bytes"] / 1024**3).round(1)
    # _df = _df.drop("size_in_bytes", axis=1)
    _df

    mo.hstack([
        _df,
        mo.ui.altair_chart(alt.Chart(_df).mark_bar().encode(
            x="year:O",
            y='count:Q',
        ))
    ])
    return


@app.cell
def _(df):
    _df = df.copy(deep=True)
    _df["year"] = _df["published_at"].dt.year
    #_df["type"] = _df["file_type"]
    _df["size_in_gibibytes"] = _df["size_in_bytes"] / 1024**3
    _df["count"] = 1
    _df = _df[["year", "file_type", "count", "size_in_gibibytes"]] 
    _df = _df.groupby(["year", "file_type"]).sum()
    compo = _df.reset_index()
    compo
    return (compo,)


@app.cell
def _(alt, compo):
    alt.Chart(compo[compo["count"]>=1000]).mark_bar().encode(
        x='year:O',
        y='count',
        color='file_type'
    )
    return


@app.cell
def _(alt, compo):
    alt.Chart(compo[compo["size_in_gibibytes"]>=20.0]).mark_bar().encode(
        x='year:O',
        y='size_in_gibibytes',
        color='file_type'
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Subjects""")
    return


@app.cell
def _(ds):
    _df = ds[["name", "keywords", "subjects"]]
    all_keywords = sorted(set(sum(list(_df["keywords"][_df["keywords"].notnull()]), [])))
    all_keywords
    return (all_keywords,)


@app.cell
def _(ds):
    _df = ds[["name", "keywords", "subjects"]]
    all_subjects = sorted(set(sum(list(_df["subjects"][_df["subjects"].notnull()]), [])))
    all_subjects
    return (all_subjects,)


@app.cell
def _(all_subjects, ds, pd):
    _df = ds[["name", "keywords", "subjects"]]
    subject_count = {key: 0 for key in all_subjects}
    for subject in all_subjects:
        for subjects in _df["subjects"]:
            if subject in subjects:
                subject_count[subject] += 1
    data = []
    for k, c in subject_count.items():
        data.append({"Topic": k, "Datasets (#)": c})
    subjects = pd.DataFrame(data)
    subjects = subjects.sort_values("Datasets (#)", ascending=False)
    subjects = subjects.reset_index(drop=True)
    subjects = subjects.set_index("Topic")

    subjects.plot.pie(y="Datasets (#)", legend=False, figsize=(10,5))
    return c, data, k, subject, subject_count, subjects


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Licenses

        No info whatsoever in the metadata??? No "license" field in the search?
        In the GUI we get the "Etalab" stuff (for all?). How does it goes for Harvard?


        I see at the very least public domain stuff (CC0) in the Harvard repo. Anything else?

        > Waiver: Harvard Dataverse strongly encourages use of a Creative Commons Zero (CC0) waiver for all public datasets, but dataset owners > can specify other terms of use and restrict access to data.

        But you can't programatically get the license? How come? WTF?

        Nota: dans la creation, API de dépôt, on peut bien choisir la license, cf <https://guides.dataverse.org/en/latest/api/native-api.html#>. Et dans le GUI? Rien a priori.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Organisation Info? Spaces?

        Espaces institutionnels ("Spaces") pas présent dans les métadonnées ? Concept différent de dataverse (oui a priori?) ?
        """
    )
    return


@app.cell
def _(ds):
    _df = ds[["publisher", "name_of_dataverse", "identifier_of_dataverse"]]
    _df
    return


@app.cell
def _(ds):
    all(ds["publisher"] == ds["name_of_dataverse"]) # redundant intel.
    return


@app.cell
def _(ds):
    _df = ds.groupby("publisher").size().to_frame()
    _df = _df.reset_index()
    _df = _df.rename(columns={0: "datasets (#)"})
    _df = _df[["publisher", "datasets (#)"]]
    _df = _df.sort_values("datasets (#)", ascending=False)
    _df = _df.reset_index(drop=True)

    _df
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Hierarchy

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
def _(df, ds, dv):
    dv.loc[:, "datasets"] = None

    for i, entry in ds.iterrows():
        if parent_id := entry.identifier_of_dataverse:
            dv_entry = dv[dv["identifier"] == parent_id]
            if dv_entry.iloc[0]["datasets"] is None:
                dv.at[dv_entry.index[0], "datasets"] = []
            dv.at[dv_entry.index[0], "datasets"].append(entry.global_id)

    ds.loc[:, "datafiles"] = None

    for i, file_entry in df.iterrows():
        if parent_id := file_entry.dataset_persistent_id:
            ds_entry = ds[ds["global_id"] == parent_id]
            if ds_entry.iloc[0]["datafiles"] is None:
                ds.at[ds_entry.index[0], "datafiles"] = []
            ds.at[ds_entry.index[0], "datafiles"].append(file_entry.file_persistent_id)

    hierarchy = True
    return ds_entry, dv_entry, entry, file_entry, hierarchy, i, parent_id


@app.cell
def _(dv, hierarchy, mo, pd):
    hierarchy

    undefined_ds = pd.isnull(dv["datasets"]).mean()
    mo.md(f"""
    About {round(undefined_ds*100.0, 1)}% of dataverses have no associated datasets. 

    **TODO:** investigate the phantom dataverses! Study the populated ones, in terms of number of data[sets/files], file size, etc.

    **TODO:** add `size_in_bytes` in datasets and dataverses dataframes (will be a nice summary and will allow to sort this stuff)

    """)
    return (undefined_ds,)


@app.cell
def _(ds, mo, pd):
    assert all(pd.isnull(ds["identifier_of_dataverse"]) == False)
    mo.md("Every dataset is registered into a dataverse 👍")
    return


@app.cell
def _(ds, hierarchy, mo, pd):
    hierarchy

    undefined_df = pd.isnull(ds["datafiles"]).mean()
    mo.md(f"""About {round(undefined_df*100.0, 1)}% of datasets have no associated datafiles. 😔
    """)
    return (undefined_df,)


@app.cell
def _(df, mo, pd):
    assert (pd.isnull(df["dataset_persistent_id"]).any() == False)
    mo.md("Every datafile is registered into a dataset 👍")
    return


@app.cell
def _(df, mo):
    _size_bytes = df["size_in_bytes"].sum()
    mo.md(f"{round(_size_bytes / 1024**3, 1)} GiB")
    return


@app.cell
def _(df):
    df["dataset_persistent_id"]
    return


@app.cell
def _(mo):
    mo.md(r"""## Easy size data analysis""")
    return


@app.cell
def _(df):
    # df.rename(columns={"size_in_bytes": "size (B)"}, inplace=True)
    df["size_in_gibibytes"] = df["size_in_bytes"] / 1024**3
    df.sort_values("size_in_bytes", ascending=False)
    return


@app.cell
def _(df, ds):
    # Dispatch datafiles sizes into datasets
    # 🪲 Buggy: all the size is allocated to the same dataset entry!
    ds["size_in_bytes"] = 0

    _s = set()

    for _, _file_entry in df.iterrows():
        dataset_id = _file_entry.dataset_persistent_id
        _s.add(dataset_id)
        _selection = ds[ds["global_id"]==dataset_id]
        _index = _selection.index[0]
        # print(_file_entry["size_in_bytes"])
        ds.at[_index, "size_in_bytes"] += _file_entry["size_in_bytes"]


    ds["size_in_gibibytes"] = ds["size_in_bytes"] / 1024**3
    _s
    return (dataset_id,)


@app.cell
def _(ds, mo):
    _size_bytes = ds["size_in_bytes"].sum()
    mo.md(f"{round(_size_bytes / 1024**3, 1)} GiB")
    return


@app.cell
def _(ds):
    ds.sort_values("size_in_bytes", ascending=False)
    return


@app.cell
def _(ds, dv):
    dv["size_in_bytes"] = 0

    for _, _dataset_entry in ds.iterrows():
        dataverse_id = _dataset_entry.identifier_of_dataverse
        _selection = dv[dv["identifier"]==dataverse_id]
        _index = _selection.index[0]
        # print(_file_entry["size_in_bytes"])
        dv.at[_index, "size_in_bytes"] += _dataset_entry["size_in_bytes"]

    dv["size_in_gibibytes"] = dv["size_in_bytes"] / 1024**3
    return (dataverse_id,)


@app.cell
def _(dv, mo):
    _size_bytes = dv["size_in_bytes"].sum()
    mo.md(f"{round(_size_bytes / 1024**3, 1)} GiB")
    return


@app.cell
def _(dv):
    dv.sort_values("size_in_bytes", ascending=False)
    return


@app.cell
def _(df, np, plt):
    plt.plot(np.array(df.sort_values("size_in_bytes", ascending=False)["size_in_gibibytes"]))
    ax = plt.gca()
    ax.set_yscale("log")
    plt.yticks([1024, 1, 1/1024, 1/1024**2], ["1 TiB", "1 GiB", "1 MiB", "1 KiB"])
    plt.grid(True)
    plt.ylabel("File size")
    plt.xlabel("# file")
    return (ax,)


@app.cell
def _(ds, np, plt):
    plt.plot(np.array(ds.sort_values("size_in_bytes", ascending=False)["size_in_gibibytes"]))
    _ax = plt.gca()
    _ax.set_yscale("log")
    plt.yticks([1024, 1, 1/1024, 1/1024**2], ["1 TiB", "1 GiB", "1 MiB", "1 KiB"])
    plt.grid(True)
    plt.ylabel("File size")
    plt.xlabel("# dataset")
    return


@app.cell
def _(dv, np, plt):
    plt.plot(np.array(dv.sort_values("size_in_bytes", ascending=False)["size_in_gibibytes"]))
    _ax = plt.gca()
    _ax.set_yscale("log")
    plt.yticks([1024, 1, 1/1024, 1/1024**2], ["1 TiB", "1 GiB", "1 MiB", "1 KiB"])
    plt.grid(True)
    plt.ylabel("File size")
    plt.xlabel("# dataverse")
    return


@app.cell
def _(dv):
    dv[dv["size_in_gibibytes"] >= 1.0].sort_values("size_in_bytes", ascending=False)
    return


@app.cell
def _(dv):
    dv_generique = dv[dv["identifier"] == "espacegenerique"]
    dv_generique
    return (dv_generique,)


@app.cell
def _(dv_generique):
    print(dv_generique.iloc[0].description)
    return


@app.cell
def _(dv_generique, mo):
    mo.md(f"Volume données génériques: {round(dv_generique.iloc[0]["size_in_bytes"]/1024**3, 1)} GiB")
    return


@app.cell
def _(mo):
    mo.md("""Plusieurs dataverses par 'espace institutionnel'??? Ex ici: <https://entrepot.recherche.data.gouv.fr/dataverse/univ-cotedazur?q=&types=dataverses&sort=dateSort&order=desc&page=1>""")
    return


@app.cell
def _(mo):
    mo.md("""API access to files is simpler with the API than the site (AFAICT). I WANT TO KNOW WHERE THE FILES ARE HOSTED.""")
    return


if __name__ == "__main__":
    app.run()
