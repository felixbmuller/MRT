
# Data

## Original Paper

- 15 joints
- 6,000 training mocap
- 800 test mocap
- 209 test mupots

datapoints
- input shape: (3, 15, 45)
- output shape: (3, 46, 45)
- (persons, frames, no_joints*3)

- `input_[:,1:15,:]-input_[:,:14,:]` = taking delta
- code at some locations implies that persons is shape[1], but it must be shape 0

## BAM poses

- 

# Setup Python Env outside Docker
- conda create -n mrt python=3.7.3 
- conda install -y pip numpy
- conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit==9.0 -c pytorch
- pip install torch-dct transforms3d matplotlib pygame pyopengl open3d==0.15.2 trimesh
- conda install numba
- conda install -c conda-forge jupyterlab

TODO:
- reread datasets & results section, find out which is done with which data

# Meetings

## Meeting 06/23

Anmerkungen/Kann man irgendwann einbauen
- MRT hat überhaupt keine data augmentation -> interessant für meine Arbeit
- Welchen Abstand vom Ursprung haben die Posen? Kann das Modell akkurate Vorhersagen für Posen
  machen, die weit vom Ursprung entfernt sind? (vmtl Normalisierung zum Ursprung notwendig ->
  interessant für meine Arbeit)
- Code so alt -> kann nur auf T80s trainieren (nur 11GB GPU-RAM, Batchsize 2-3) -> später auf
  mehreren Maschinen gleichzeitig trainieren, für Entwicklung kein Problem

Konkrete Aufgaben:
1. BAM Datensatz preprocessen, damit man ihn mit MRT benutzen kann
  - erstmal ohne overlap splitten
  - cave: 25 statt 15hz
2. Herausfinden, wie ich am besten missing persons maskiere:
  - infinity (besser für transformer) vs 0
  - Wie funktioniert der Kontext genau?
  - cave: self-attention funktioniert mit infty nicht
  - keine missing joints von einzelnen Skeletten, einfach filtern

## Meeting 06/30

Fragen
- Warum sind die Personen in BAM poses sparses? I.e. unvollständige/abwesende Personen zwischen
  anwesenden Ist einfach zusammenschieben eine gute Lösung? Weil 18 unique personen?
- Auf histogramm schauen, ist das realistisch?
- Ist es okay personen komplett zu maskieren, die innerhalb der 3sec kommen/gehen?
- Auswahl an joints okay?