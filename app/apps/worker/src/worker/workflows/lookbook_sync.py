"""Lookbook-Sync: haelt LOOKBOOK.md im Repo automatisch aktuell.

Liest neue Fotos aus ``schrank/`` und ``look/``, erstellt Thumbnails nach ``thumbs/``,
analysiert jedes Teil per Vision-Modell, verlinkt pro Look passende Schrank-Teile und
aktualisiert die deutsche LOOKBOOK.md sowie lookbook-state.json im Repo.
"""

import base64
import io
import json
import os
from typing import Literal

import mistralai.workflows as workflows
import structlog
from mistralai.client.models import ChatCompletionRequest, ImageURLChunk, TextChunk, UserMessage
from mistralai.workflows import activity
from mistralai.workflows.plugins.mistralai import chat_parse_to_model
from pydantic import BaseModel, Field

logger = structlog.get_logger("lookbook")

VISION_MODEL = "mistral-small-latest"
KATEGORIEN = "oberteil, hose, jacke, schuh, accessoire, sonstiges"
THUMB_SIZE = 512
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".avif"}


class LookbookInput(BaseModel):
    dry_run: bool = False


class LookbookOutput(BaseModel):
    dry_run: bool
    new_photos: int = 0
    new_looks: int = 0
    schrank_items: int = 0
    look_items: int = 0
    commit_sha: str | None = None
    message: str = ""


class ItemAnalysis(BaseModel):
    name: str
    kategorie: Literal["oberteil", "hose", "jacke", "schuh", "accessoire", "sonstiges"]
    farben: list[str]
    stil: str
    beschreibung: str


class LookMatches(BaseModel):
    schrank_pfade: list[str] = Field(default_factory=list)
    begruendung: str = ""


class PhotoInfo(BaseModel):
    path: str
    kind: Literal["schrank", "look"]
    analysis: ItemAnalysis
    thumb_b64: str


class ScanResult(BaseModel):
    repo: str
    branch: str
    head_sha: str
    state: dict = Field(default_factory=dict)
    new_photos: list[str] = Field(default_factory=list)


class StoredItem(BaseModel):
    path: str
    kind: str
    thumb: str
    analysis: ItemAnalysis


class CommitPayload(BaseModel):
    dry_run: bool
    branch: str
    head_sha: str
    lookbook_md: str
    site_html: str
    state_json: str
    files: dict[str, str] = Field(default_factory=dict)


def _github_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("github_token") or ""
    if not token:
        raise RuntimeError("GITHUB_TOKEN ist nicht gesetzt (Workspace-Secret github_token fehlt).")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _repo() -> str:
    return os.environ.get("GITHUB_REPO", "Julian-Lukas/lookbook")


@activity()
async def scan_repo(dry_run: bool) -> ScanResult:
    import httpx

    async with httpx.AsyncClient(timeout=60, headers=_github_headers()) as client:
        repo = _repo()
        resp = await client.get(f"https://api.github.com/repos/{repo}")
        resp.raise_for_status()
        branch = resp.json()["default_branch"]

        resp = await client.get(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1")
        resp.raise_for_status()
        tree = resp.json()["tree"]
        head_sha = next((e["sha"] for e in tree if e.get("sha") and e["path"] == branch), None)
        if head_sha is None:
            resp = await client.get(f"https://api.github.com/repos/{repo}/git/ref/heads/{branch}")
            resp.raise_for_status()
            head_sha = resp.json()["object"]["sha"]

        all_paths = [e["path"] for e in tree if e["type"] == "blob"]

        state: dict = {}
        if "lookbook-state.json" in all_paths:
            resp = await client.get(
                f"https://api.github.com/repos/{repo}/contents/lookbook-state.json",
                headers={"Accept": "application/vnd.github.raw"},
            )
            if resp.status_code == 200:
                state = resp.json()

        known = set(state.get("items", {}))
        exts = IMAGE_EXTENSIONS
        new_photos = [
            p
            for p in all_paths
            if (p.startswith("schrank/") or p.startswith("look/") or p.startswith("looks/"))
            and os.path.splitext(p)[1].lower() in exts
            and p not in known
        ]
        logger.info("scan abgeschlossen", repo=repo, neue_bilder=len(new_photos), dry_run=dry_run)
        return ScanResult(repo=repo, branch=branch, head_sha=head_sha, state=state, new_photos=new_photos)


@activity()
async def process_photo(path: str) -> PhotoInfo:
    import httpx

    kind = "look" if path.startswith(("look/", "looks/")) else "schrank"
    async with httpx.AsyncClient(timeout=120, headers=_github_headers()) as client:
        resp = await client.get(
            f"https://api.github.com/repos/{_repo()}/contents/{path}",
            headers={"Accept": "application/vnd.github.raw"},
        )
        resp.raise_for_status()
        raw = resp.content

    thumb_b64 = _make_thumbnail(raw)
    data_uri = f"data:image/jpeg;base64,{thumb_b64}"

    prompt = (
        "Analysiere das Foto eines Kleidungsstücks oder Accessoires. "
        f"Kategorie ist genau einer von: {KATEGORIEN}. "
        "Antworte auf Deutsch. 'name' ist ein kurzer, prägnanter Name (max. 5 Wörter). "
        "'farben' sind die Hauptfarben als deutsche Farbnamen. "
        "'stil' beschreibt den Stil (z. B. casual, elegant, sportlich, streetwear, business). "
        "'beschreibung' ist ein Satz zum Teil."
    )
    if kind == "look":
        prompt = (
            "Analysiere das Foto eines kompletten Outfits / einer Look-Idee. "
            f"Kategorie ist genau einer von: {KATEGORIEN} (meist 'sonstiges'). "
            "Antworte auf Deutsch. 'name' ist ein kurzer Name für den Look (max. 5 Wörter). "
            "'farben' sind die Hauptfarben als deutsche Farbnamen. "
            "'stil' beschreibt den Stil. 'beschreibung' fasst den Look in einem Satz zusammen."
        )

    request = ChatCompletionRequest(
        model=VISION_MODEL,
        messages=[
            UserMessage(
                content=[
                    ImageURLChunk(image_url=data_uri),
                    TextChunk(text=prompt),
                ]
            )
        ],
        max_tokens=400,
    )
    analysis = await chat_parse_to_model(ItemAnalysis, request)
    logger.info("foto analysiert", path=path, name=analysis.name, kategorie=analysis.kategorie)
    return PhotoInfo(path=path, kind=kind, analysis=analysis, thumb_b64=thumb_b64)


@activity()
async def match_look_items(look: StoredItem, schrank_items: list[StoredItem]) -> LookMatches:
    if not schrank_items:
        return LookMatches()
    katalog = "\n".join(
        f"- {s.path}: {s.analysis.name} ({s.analysis.kategorie}, Farben: {', '.join(s.analysis.farben)}, Stil: {s.analysis.stil})"
        for s in schrank_items
    )
    prompt = (
        "Hier ist eine Look-Idee:\n"
        f"Name: {look.analysis.name}\n"
        f"Beschreibung: {look.analysis.beschreibung}\n"
        f"Stil: {look.analysis.stil}\nFarben: {', '.join(look.analysis.farben)}\n\n"
        "Und der Kleiderschrank:\n"
        f"{katalog}\n\n"
        "Welche Teile aus dem Schrank passen zu diesem Look (maximal 4)? "
        "Gib in 'schrank_pfade' genau die Pfade aus dem Katalog an, die am besten passen. "
        "Begründe kurz auf Deutsch in 'begruendung'."
    )
    request = ChatCompletionRequest(
        model=VISION_MODEL,
        messages=[UserMessage(content=prompt)],
        max_tokens=300,
    )
    matches = await chat_parse_to_model(LookMatches, request)
    gueltig = {s.path for s in schrank_items}
    matches.schrank_pfade = [p for p in matches.schrank_pfade if p in gueltig][:4]
    return matches


@activity(retry_policy_max_attempts=2)
async def commit_results(payload: CommitPayload) -> str:
    import httpx

    files = dict(payload.files)
    files["LOOKBOOK.md"] = base64.b64encode(payload.lookbook_md.encode()).decode()
    files["index.html"] = base64.b64encode(payload.site_html.encode()).decode()
    files["lookbook-state.json"] = base64.b64encode(payload.state_json.encode()).decode()

    if payload.dry_run:
        logger.info("dry_run: kein Commit", dateien=len(files))
        return ""

    api = f"https://api.github.com/repos/{_repo()}"
    async with httpx.AsyncClient(timeout=120, headers=_github_headers()) as client:
        blobs: dict[str, str] = {}
        for path, content_b64 in files.items():
            resp = await client.post(
                f"{api}/git/blobs",
                json={"content": content_b64, "encoding": "base64"},
            )
            resp.raise_for_status()
            blobs[path] = resp.json()["sha"]

        tree_items = [
            {"path": path, "mode": "100644", "type": "blob", "sha": sha} for path, sha in blobs.items()
        ]
        resp = await client.post(f"{api}/git/trees", json={"base_tree": payload.head_sha, "tree": tree_items})
        resp.raise_for_status()
        tree_sha = resp.json()["sha"]

        resp = await client.post(
            f"{api}/git/commits",
            json={
                "message": "lookbook-sync: Thumbnails, Analysen und LOOKBOOK.md aktualisiert",
                "tree": tree_sha,
                "parents": [payload.head_sha],
            },
        )
        resp.raise_for_status()
        commit_sha = resp.json()["sha"]

        resp = await client.patch(
            f"{api}/git/refs/heads/{payload.branch}",
            json={"sha": commit_sha},
        )
        resp.raise_for_status()
        logger.info("commit gepusht", sha=commit_sha, branch=payload.branch)
        return commit_sha


def _make_thumbnail(raw: bytes) -> str:
    from PIL import Image

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _thumb_name(path: str) -> str:
    base = os.path.basename(path)
    return "thumbs/" + os.path.splitext(base)[0] + ".jpg"


@workflows.workflow.define(name="lookbook_sync")
class LookbookSync:
    @workflows.workflow.entrypoint
    async def run(self, data: LookbookInput) -> LookbookOutput:
        scan = await scan_repo(data.dry_run)

        stored: dict[str, StoredItem] = {
            path: StoredItem.model_validate(item)
            for path, item in scan.state.get("items", {}).items()
        }

        new_schrank: list[PhotoInfo] = []
        new_looks: list[PhotoInfo] = []
        for path in scan.new_photos:
            info = await process_photo(path)
            (new_looks if info.kind == "look" else new_schrank).append(info)

        files: dict[str, str] = {}
        for info in new_schrank + new_looks:
            thumb_path = _thumb_name(info.path)
            files[thumb_path] = info.thumb_b64
            stored[info.path] = StoredItem(
                path=info.path, kind=info.kind, thumb=thumb_path, analysis=info.analysis
            )

        schrank_items = [s for s in stored.values() if s.kind == "schrank"]
        look_items = [s for s in stored.values() if s.kind == "look"]

        look_matches: dict[str, LookMatches] = {}
        for look in look_items:
            is_new = look.path in {i.path for i in new_looks}
            cached = scan.state.get("matches", {}).get(look.path)
            if is_new or not cached:
                look_matches[look.path] = await match_look_items(look, schrank_items)
            else:
                look_matches[look.path] = LookMatches.model_validate(cached)

        lookbook_md = _render_markdown(scan.repo, schrank_items, look_items, look_matches)
        site_html = _render_site(scan.repo, schrank_items, look_items, look_matches)

        state = {
            "version": 1,
            "items": {path: item.model_dump() for path, item in stored.items()},
            "matches": {path: m.model_dump() for path, m in look_matches.items()},
        }
        state_json = json.dumps(state, ensure_ascii=False, indent=2) + "\n"

        commit_sha = await commit_results(
            CommitPayload(
                dry_run=data.dry_run,
                branch=scan.branch,
                head_sha=scan.head_sha,
                lookbook_md=lookbook_md,
                site_html=site_html,
                state_json=state_json,
                files=files,
            )
        )

        if data.dry_run:
            nachricht = (
                f"Dry-Run: {len(new_schrank)} neue Schrank-Teile und {len(new_looks)} neue Looks "
                f"analysiert, nichts ins Repo geschrieben."
            )
        else:
            nachricht = (
                f"Lookbook aktualisiert: {len(new_schrank)} neue Schrank-Teile, "
                f"{len(new_looks)} neue Looks, Commit {commit_sha[:7]}."
            )
        return LookbookOutput(
            dry_run=data.dry_run,
            new_photos=len(new_schrank) + len(new_looks),
            new_looks=len(new_looks),
            schrank_items=len(schrank_items),
            look_items=len(look_items),
            commit_sha=commit_sha or None,
            message=nachricht,
        )


def _render_site(
    repo: str,
    schrank_items: list[StoredItem],
    look_items: list[StoredItem],
    look_matches: dict[str, LookMatches],
) -> str:
    import html

    def esc(s: str) -> str:
        return html.escape(s, quote=True)

    def karte(item: StoredItem, footer: str = "") -> str:
        a = item.analysis
        badgerl = " ".join(f'<span class="badge">{esc(f)}</span>' for f in a.farben)
        return (
            f'<div class="card">'
            f'<a href="{esc(item.path)}" target="_blank"><img src="{esc(item.thumb)}" alt="{esc(a.name)}" loading="lazy"></a>'
            f'<div class="card-body"><h3>{esc(a.name)}</h3>'
            f'<span class="kat">{esc(a.kategorie)}</span>{badgerl}{footer}</div></div>'
        )

    schrank_html = ""
    by_kind: dict[str, list[StoredItem]] = {}
    for item in sorted(schrank_items, key=lambda s: s.path):
        by_kind.setdefault(item.analysis.kategorie, []).append(item)
    for kategorie in ["oberteil", "hose", "jacke", "schuh", "accessoire", "sonstiges"]:
        items = by_kind.get(kategorie, [])
        if not items:
            continue
        schrank_html += f'<h2>{esc(kategorie.capitalize())} <small>({len(items)})</small></h2><div class="grid">'
        schrank_html += "".join(karte(i) for i in items)
        schrank_html += "</div>"
    if not schrank_html:
        schrank_html = '<p class="empty">Noch keine Fotos in <code>schrank/</code>.</p>'

    looks_html = ""
    for look in sorted(look_items, key=lambda s: s.path):
        matches = look_matches.get(look.path)
        footer = ""
        if matches and matches.schrank_pfade:
            treffer = []
            for path in matches.schrank_pfade[:4]:
                item = next((s for s in schrank_items if s.path == path), None)
                if item:
                    treffer.append(
                        f'<a class="match" href="{esc(item.thumb)}" target="_blank">'
                        f'<img src="{esc(item.thumb)}" alt="{esc(item.analysis.name)}">'
                        f'<span>{esc(item.analysis.name)}</span></a>'
                    )
            footer = (
                f'<div class="matches"><h4>Passt dazu aus deinem Schrank</h4>'
                f'<div class="match-row">{"".join(treffer)}</div>'
                + (f'<p class="why">{esc(matches.begruendung)}</p>' if matches.begruendung else "")
                + "</div>"
            )
        else:
            footer = '<div class="matches"><h4>Passt dazu aus deinem Schrank</h4><p class="why">Noch keine Treffer im Schrank.</p></div>'
        looks_html += karte(look, footer)
    if not look_items:
        looks_html = '<p class="empty">Noch keine Look-Ideen in <code>look/</code> — lade Fotos hoch und der nächste Sync erzeugt sie hier.</p>'
    looks_html = f'<div class="grid looks">{looks_html}</div>'

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mein Lookbook</title>
<style>
:root {{ --bg:#0f1115; --panel:#171a21; --card:#1d2129; --text:#e8eaf0; --muted:#9aa3b2; --accent:#ff7a59; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--text); }}
header {{ padding:2rem 1rem 0; text-align:center; }}
header h1 {{ margin:0 0 .25rem; font-size:2rem; }}
header p {{ margin:0 0 1rem; color:var(--muted); }}
.tabs {{ display:flex; justify-content:center; gap:.5rem; padding:1rem; position:sticky; top:0; background:var(--bg); z-index:5; border-bottom:1px solid #23262e; }}
.tabs button {{ border:none; border-radius:999px; padding:.6rem 1.6rem; font-size:1rem; cursor:pointer; background:var(--panel); color:var(--muted); }}
.tabs button.active {{ background:var(--accent); color:#141414; font-weight:600; }}
main {{ max-width:1100px; margin:0 auto; padding:1rem; }}
section {{ display:none; }}
section.active {{ display:block; }}
h2 {{ margin:1.5rem 0 .75rem; font-size:1.25rem; }}
h2 small {{ color:var(--muted); font-weight:400; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:1rem; }}
.card {{ background:var(--card); border-radius:14px; overflow:hidden; border:1px solid #23262e; }}
.card img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; }}
.card-body {{ padding: .8rem 1rem 1rem; }}
.card-body h3 {{ margin:0 0 .4rem; font-size:1rem; }}
.kat {{ display:inline-block; background:var(--accent); color:#141414; font-size:.72rem; font-weight:600; padding:.15rem .6rem; border-radius:999px; margin-right:.25rem; text-transform:uppercase; }}
.badge {{ display:inline-block; background:var(--panel); color:var(--muted); font-size:.72rem; padding:.15rem .55rem; border-radius:999px; margin:.25rem .15rem 0; }}
.matches {{ margin-top:.75rem; border-top:1px solid #23262e; padding-top:.6rem; }}
.matches h4 {{ margin:0 0 .5rem; font-size:.85rem; color:var(--muted); }}
.match-row {{ display:flex; gap:.5rem; flex-wrap:wrap; }}
.match {{ text-decoration:none; color:var(--text); font-size:.72rem; }}
.match img {{ width:56px; height:56px; border-radius:10px; object-fit:cover; display:block; }}
.match span {{ display:block; max-width:76px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-top:.2rem; color:var(--muted); }}
.why {{ color:var(--muted); font-size:.8rem; font-style:italic; margin:.5rem 0 0; }}
.empty {{ color:var(--muted); text-align:center; padding:2rem; }}
footer {{ text-align:center; color:var(--muted); font-size:.8rem; padding:2rem; }}
</style>
</head>
<body>
<header>
<h1>Mein Lookbook</h1>
<p>Automatisch generiert vom <code>lookbook_sync</code> Workflow · Originale auf <a href="https://github.com/{repo}">GitHub</a></p>
</header>
<nav class="tabs"><button id="tab-schrank" class="active" onclick="show('schrank')">Schrank ({len(schrank_items)})</button><button id="tab-looks" onclick="show('looks')">Looks ({len(look_items)})</button></nav>
<main>
<section id="sec-schrank" class="active">{schrank_html}</section>
<section id="sec-looks">{looks_html}</section>
</main>
<footer>Letzter Sync: workflow <code>lookbook_sync</code></footer>
<script>
function show(tab) {{
  document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
  document.getElementById('sec-' + tab).classList.add('active');
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  history.replaceState(null, '', '#' + tab);
}}
if (location.hash === '#looks') show('looks');
</script>
</body>
</html>
"""


def _render_markdown(
    repo: str,
    schrank_items: list[StoredItem],
    look_items: list[StoredItem],
    look_matches: dict[str, LookMatches],
) -> str:
    lines = ["# Mein digitales Lookbook", ""]
    lines.append(
        "Automatisch generiert vom `lookbook_sync` Workflow. "
        "Originale liegen in [schrank/](schrank/) und [look/](look/)."
    )
    lines += ["", "---", "", "## Mein Schrank", ""]

    by_kind: dict[str, list[StoredItem]] = {}
    for item in schrank_items:
        by_kind.setdefault(item.analysis.kategorie, []).append(item)

    for kategorie in ["oberteil", "hose", "jacke", "schuh", "accessoire", "sonstiges"]:
        items = sorted(by_kind.get(kategorie, []), key=lambda s: s.path)
        if not items:
            continue
        lines += [f"### {kategorie.capitalize()}", ""]
        for item in items:
            a = item.analysis
            lines.append(f"#### {a.name}")
            lines.append("")
            lines.append(
                f'<a href="https://github.com/{repo}/blob/main/{item.path}">'
                f'<img src="{item.thumb}" alt="{a.name}" width="256"></a>'
            )
            lines.append("")
            lines.append(f"**Kategorie:** {a.kategorie} · **Farben:** {', '.join(a.farben)}")
            lines.append("")
            lines.append(f"[Original ansehen]({item.path})")
            lines.append("")

    lines += ["---", "", "## Look-Ideen", ""]
    if not look_items:
        lines.append("Noch keine Looks in `look/` hochgeladen.")
        lines.append("")
    for look in sorted(look_items, key=lambda s: s.path):
        a = look.analysis
        lines.append(f"### {a.name}")
        lines.append("")
        lines.append(
            f'<a href="https://github.com/{repo}/blob/main/{look.path}">'
            f'<img src="{look.thumb}" alt="{a.name}" width="256"></a>'
        )
        lines.append("")
        lines.append(f"**Farben:** {', '.join(a.farben)}")
        lines.append("")
        matches = look_matches.get(look.path)
        if matches and matches.schrank_pfade:
            lines.append("**Passt dazu aus deinem Schrank:**")
            lines.append("")
            for path in matches.schrank_pfade:
                item = next((s for s in schrank_items if s.path == path), None)
                if item:
                    lines.append(
                        f"- [{item.analysis.name}]({item.thumb}) "
                        f"([Original]({item.path}))"
                    )
            if matches.begruendung:
                lines.append("")
                lines.append(f"_{matches.begruendung}_")
        else:
            lines.append("**Passt dazu aus deinem Schrank:** noch keine Treffer im Schrank.")
        lines += ["", "---", ""]
    return "\n".join(lines) + "\n"
