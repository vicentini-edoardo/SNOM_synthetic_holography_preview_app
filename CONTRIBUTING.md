# Contributing

## Local Setup

1. Install Python 3.10+.
2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Launch the app:

```bash
python3 main.py
```

## Development Notes

- `processing.py` is the processing backend and should stay independent from the GUI toolkit where possible.
- `viewer.py` owns the PySide6 interface and Matplotlib embedding.
- Keep generated files, caches, and local environments out of version control.

## Before Opening A Pull Request

- Run:

```bash
python3 -m py_compile main.py viewer.py processing.py
```

- Verify the GUI starts.
- Verify at least one real dataset still loads and exports correctly.
- If you touch the UI, update `GUI.png` when the screenshot is no longer representative.

## GitHub Publication Checklist

- Add a repository description and topics.
- Choose and add a license explicitly.
- Confirm `README.md` reflects the current UI and install process.
- Remove local artifacts before pushing.
- Make sure no private sample data is committed.
