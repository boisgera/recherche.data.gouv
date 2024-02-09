<https://entrepot.recherche.data.gouv.fr>

Fédération d'entrepôts institutionnels des données de la recherche,
à utiliser à défaut d'entrepôts disciplinaires adaptés. Présence
d'un entrepôt racine/générique à utiliser en l'absence d'entrepôt 
institutionnel.

Cf. <https://recherche.data.gouv.fr/fr/guide/ou-deposer-et-publier-ses-donnees>
pour la stratégie de recherche de dépôt de données disciplinaires.

Solution technique
--------------------------------------------------------------------------------

Exploite <https://dataverse.org/> issu (et utilisé par) Harvard 
(https://dataverse.harvard.edu/).


API
--------------------------------------------------------------------------------

Doc (dataverse.org): <https://guides.dataverse.org/en/latest/api/>

Notamment pour la recherche : <https://guides.dataverse.org/en/latest/api/search.html>

Exemple d'API Rest avec retour en JSON : <https://demo.dataverse.org/api/search?q=trees>

Sur un "vrai" server : <https://dataverse.harvard.edu/api/search?q=trees>

Sur <entrepot.recherche.data.gouv.fr> : <https://entrepot.recherche.data.gouv.fr/api/search?q=trees>

TODO: dans un notebook, avec pandas pour les tableaux(?) 
  - Taille dataset: (5.5 To)
  - Composition par type de données (non typées (1er !!!), images, ZIP, TSV/CSV, HDF, Source code, etc.)
  - Taille dépôt générique (~nulle)

Classement des fichiers par taille plutôt que par compte? (complément)

Question: croissance des données en volume en fonction du temps ?
