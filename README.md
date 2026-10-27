# Magnificat — Sancta Lactatio

Retour vidéo webcam sanctifié pour un spectacle technocritique autour de
Saint-François, la lactation des meufs trans et le Miracle que c'est.

## Fonctions

- **Retour plein écran sans déformation** : la vidéo remplit l'écran en
  conservant son ratio — bandes noires sur les côtés si nécessaire.
- **NudeNet** : affiche toutes les infos qu'il peut obtenir dans un panneau
  latéral (traduites en français, tétons inclus), avec boîtes sur la vidéo.
- **YOLO-World** : repère les vaches (`cow`), les pis (`udder`), les biberons
  (`baby bottle`) et les tétons (`nipple`). Plus il en voit, plus la barre de
  **« Lait divin »** monte en haut de l'écran.
- **Slider « Foi »** (fenêtre secondaire) : biaise le modèle pour qu'il en voie
  partout — à 100 %, toutes les détections deviennent du « lait divin ».
- **MediaPipe Hands** : deux doigts levés (index + majeur) déclenchent un
  **hallelujah sonore** (synthétisé, aucun fichier requis) + grand « HALLELUJAH ».
- **Auréoles dorées** autour des têtes des comédiennes (l'arrière de l'anneau
  passe derrière le crâne) et inscription **SANCTA** au-dessus de leurs têtes.

## Installation

Installer dans un environnement dédié pour éviter que mediapipe ne rétrograde
`numpy`/`protobuf` pour tes autres projets (deepface, retina-face, etc.) :

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Les versions sont épinglées (`numpy<2`, `opencv<5`) pour éviter le long
« backtracking » du résolveur pip avec mediapipe.

Notes :

- Le premier lancement télécharge `yolov8s-worldv2.pt` (24 Mo) puis le met en cache.
- Les messages `inference_feedback_manager`, XNNPACK ou `torch.jit.load` au
  démarrage sont des avertissements internes bénins de MediaPipe/torch.

## Lancement

```bash
python magnificat.py                # webcam 0
python magnificat.py --camera=1     # autre webcam
python magnificat.py --test         # pattern de test sans webcam
```

Échap ou fermeture de fenêtre pour quitter. La fenêtre « Foi » contient le
slider de biais YOLO-World.

## Notes spectacle

- Le biais « Foi » ne modifie pas le modèle : il réévalue les détections
  existantes en gonflant leur confiance — le modèle « en voit partout »
  sans perdre ses détections normales, c'est le commentaire technocritique.
- Si un analyseur est absent (pas de modèle, pas de webcam), l'app démarre
  quand même et affiche ce qui est disponible.
