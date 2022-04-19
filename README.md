# bitcoin_prediction

Φακελος maintenance: Ενας φακελος που θα χρησιμοποιειται για επανεκπαυδευση του μοντελου σε ολα τα δεδομενα μετα απο ενα χρονικο διαστημα

Φακελος model_exp1: Ο φακελος του μοντελου (δεν αλλαζουμε τιποτα)

data_collector.py: Συλλογη δεδομενων
yfdata.py: Συλλογη δεδομενων bitcoin 
rolling.py: Bοηθητικη συναρτηση για τον υπολογισμο του z-score με moving averages

============= Τα βασικα τωρα ==============

predictor.py: Ενα object που κανει predict με δωθεν LAG (Το LAG για αυτο το πειραμα ειναι σταθερο και δεν θελουμε να αλλαξει)

Επιστρεφει:

- signal: Αν θα παει πανω η κατω
- prediction: Η προβλεψη για το σημερινο close
- prediction date: Ημερομηνια προβλεψης (Η σημερινη)
- previous close: προηγουμενο κλεισιμο
- previous date: Ημερομηνια προηγουμενου κλεισηματος (Η χθεσινη)

monitor.py: Ενα απλο script που καλει την predictor και κανει προβλεψη


[How to work with github markdown tables](https://www.pluralsight.com/guides/working-tables-github-markdown)

#### Αποτελέσματα ερευνών σε $ για `model_exp1_alt`

| Signal Date   | Result     | Signal ( Up/Down ) | Yesterday Close | 
|---------------|------------|---------|---------| 
| 17/April/2022 | 39910.7 $  | Up      |           |   
| 18/April/2022 | 39941.23 $ | Up      | 39716.95 $ |
| 19/April/2022 |            |         |           |
| 20/April/2022 |            |         |           |
| 21/April/2022 |            |         |           |
| 22/April/2022 |            |         |           |
| 23/April/2022 |            |         |           |
| 24/April/2022 |            |         |           |
| 25/April/2022 |            |         |           |
| 26/April/2022 |            |         |           |
| 27/April/2022 |            |         |           |
| 28/April/2022 |            |         |           |


#### Αποτελέσματα ερευνών για `model_exp1`

| Signal Date | Result | Signal ( Up/Down ) | Yesterday Close | 
|---------------|------------|---------|---------| 
| 18/April/2022 | 39960.76 $ | UP | 39716.95 $ |
| 19/April/2022 | | | |
| 20/April/2022 | | | |
| 21/April/2022 | | | |
| 22/April/2022 | | | |
| 23/April/2022 | | | |
| 24/April/2022 | | | |
| 25/April/2022 | | | |
| 26/April/2022 | | | |
| 27/April/2022 | | | |
| 28/April/2022 | | | |
