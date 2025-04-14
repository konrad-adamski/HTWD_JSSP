# 🚀 Schnellanleitung: HiGHS unter Windows kompilieren (MSVC + CMake GUI)

## 🧰 Voraussetzungen

- Visual Studio 2022 (Community Edition reicht)
  - Workload: ✅ „Desktopentwicklung mit C++“
- CMake GUI (https://cmake.org/download/)
- Git (optional)

---

## 📦 1. HiGHS-Quellcode herunterladen

```bash
git clone https://github.com/coin-or/HiGHS.git
```

Oder ZIP von GitHub herunterladen und entpacken, z. B. nach `C:\tools\HiGHS`.

---

## 📁 2. Build-Ordner erstellen

Manuell im Explorer oder via Konsole:

```bash
mkdir C:\tools\HiGHS\build
```

---

## 🛠️ 3. CMake GUI verwenden

- **Where is the source code:**  
  `C:/tools/HiGHS`

- **Where to build the binaries:**  
  `C:/tools/HiGHS/build`

Dann:

1. Klick auf **„Configure“**
2. Generator wählen:  
   - ✅ Visual Studio 17 2022  
   - ✅ Plattform: `x64`
3. Danach: **„Generate“**
4. Dann: **„Open Project“**

---

## 🧠 4. In Visual Studio kompilieren

1. Öffne `C:\tools\HiGHS\build\HIGHS.sln`
2. Oben einstellen:
   - ✅ Konfiguration: `Release`
   - ✅ Plattform: `x64`
3. Menü → **„Erstellen → Projektmappe erstellen“**  
   (oder `Strg + Umschalt + B`)

---

## ✅ 5. highs.exe ausführen

Nach erfolgreichem Build findest du:

```text
C:\tools\HiGHS\build\bin\Release\highs.exe
```

Testen:

```powershell
cd C:\tools\HiGHS\build\bin\Release
.\highs.exe
```

---

## 🧩 6. highs.exe zum Windows-PATH hinzufügen

1. Windows-Suche → **„Umgebungsvariablen bearbeiten“** öffnen
2. Unter **Systemvariablen** → `Path` → **Bearbeiten**
3. Klick auf **„Neu“** → Pfad eintragen:

```
C:\tools\HiGHS\build\bin\Release
```

4. Mit OK bestätigen
5. Konsole oder VS Code **neu starten**

---

## 🐍 7. Verwendung in Python (PuLP)

Falls du `PuLP` nutzt:

```python
solver = pulp.HiGHS_CMD(msg=True, timeLimit=300)
```

Oder direkt mit Pfadangabe:

```python
solver = pulp.HiGHS_CMD(
    path="C:/tools/HiGHS/build/bin/Release/highs.exe",
    msg=True,
    timeLimit=300
)
```

---

🎉 Fertig! Du hast HiGHS unter Windows erfolgreich kompiliert und eingebunden.
