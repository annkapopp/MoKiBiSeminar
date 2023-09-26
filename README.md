# MoKiBiSeminar: Continual Test-Time Domain Adaptation

## Kurzanleitung zum Ausführen des Codes:

### Basismodell trainieren
Um ein Basismodell zu trainieren, muss `train_base_mode.py` ausgeführt werden. Anpassungen der Hyperparameter und des
verwendeten Datensatzes können in der main-Methode vorgenommen werden.

### Basismodell testen
Zum Testen eines Basismodells wird `test_base_model.py` ausgeführt. Hierfür muss in der main-Methode der richtige
Datensatz auskommentiert werden und der Name des zu testenden Modells eingegeben werden.
Dieses Skript eine .pkl-Datei im `results`-Ordner mit dem gleichen Namen wie das Modell und der Endung "_test".
In dieser Datei stehen nur die Bildnamen mit den jeweiligen Vorhersagen und Targets.

### Auswertung der Vorhersagen
Um die Vorhersagen eines Modells auf Testdaten auszuwerten, wird `evaluation.py` benötigt. Hier wird in der main-Methode
der Name der entsprechende .pkl-Datei eingegeben, welche die Vorhersagen des Modells beinhaltet. Beim Ausführen werden
verschiedene Dateien im Ordner `results` erzeugt, welche Metriken beinhalten. Diese wurden berechnet für jedes Bild, 
für jede Klasse ("_label") und als Durchschnitt über die kompletten Testdaten ("_avg").
