
# BD: Trash Sorting using CBAM

README ir veidots ar Gemini.

Šis projekts ir bakalaura darba izstrādes daļa, kurā tiek pētīta atkritumu šķirošana, izmantojot YOLO/Ultralytics modeli, kas papildināts ar **CBAM (Convolutional Block Attention Module)** mehānismu.

---

## ⚠️ Svarīga piezīme pirms darba uzsākšanas
Kods pašlaik nav "plug-and-play". Lai to palaistu, jāņem vērā:
* **Absolūtie failu ceļi:** Kodā ir izmantoti lokālie *hardcoded* failu ceļi (`filepaths`), kas pirms palaišanas ir jānomaina uz jūsu sistēmai atbilstošiem.
* **Datu kopa:** Nepieciešams patstāvīgi lejupielādēt attiecīgo Roboflow datu kopu. BD/WasteProject/datasets/waste_data_4
* **Ultralytics modifikācija:** Lokāli ir jāinstalē `ultralytics` bibliotēka un manuāli jāveic izmaiņas tās pirmkodā, lai integrētie CBAM bloku arhitektūra strādātu.

---

## 📂 Repozitorija struktūra un saturs

Projekta svarīgākās daļas un rezultāti ir apskatāmi šādās mapēs:

* **Trenēšanas rezultāti:** Atrodami mapē `BD/results`. 
* **Modeļu arhitektūras:** Projektā tiek salīdzinātas trīs dažādas pieejas:
  1. **Baseline** (Standarta YOLO modelis bez modifikācijām)
  2. **CBAM in Head** (Uzmanības mehānisms integrēts modeļa "galvā")
  3. **CBAM in Backbone** (Uzmanības mehānisms integrēts modeļa "mugurkaulā")

---

