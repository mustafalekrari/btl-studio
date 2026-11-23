# Données

Les images ne sont pas versionnées (taille et confidentialité des photos d'intérieur).

| Dossier | Contenu | Cible |
| --- | --- | --- |
| `scenes/` | Photos de pièces (salons, chambres, salles à manger), JPG/PNG | ~40 images |
| `furniture/` | Photos produits Home BTL, une par meuble | 15–20 images |

Convention de nommage :
- `scenes/salon_001.jpg`, `scenes/chambre_001.jpg`, …
- `furniture/<categorie>_<nom>.png`, ex. `furniture/canape_milano.png`

Une fiche produit optionnelle peut accompagner chaque meuble (`furniture/<nom>.json`) :

```json
{"nom": "Canapé Milano", "categorie": "canapé", "dimensions_cm": [220, 90, 85], "matiere": "velours", "couleur": "beige", "prix_mad": 8900}
```

Elle est injectée dans le prompt du LLM pour éviter les hallucinations sur le produit.
