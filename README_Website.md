# CRISP-DM Interactive Regression Website

This project now includes a **GitHub Pages-ready static site**.

## Project structure

- `docs/` → GitHub Pages site source
  - `index.html`
  - `style.css`
  - `app.js`
  - `data/modified_data.csv`
  - `data/Human-AI Interaction.csv`
- `data/` → canonical dataset folder for local/non-pages workflows
- `app.py` → optional Streamlit version (still runnable locally)

## Deploy to GitHub Pages

1. Push this repository to GitHub.
2. In repository settings, open **Pages**.
3. Set source to **Deploy from a branch**.
4. Choose your branch (e.g., `main`) and folder **`/docs`**.
5. Save and wait for deployment.

The website will be hosted from `docs/` and datasets are loaded from `docs/data/`.

## Optional local Streamlit run

```bash
pip install -r requirements.txt
streamlit run app.py
```
