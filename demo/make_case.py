"""Build the demo case file in demo/case-moreau/.

An invented Lausanne tenancy: the boiler broke, the tenant had it repaired herself, won a
settlement at the conciliation authority, and two months later received notice to quit. The
documents come in the forms a law firm actually receives:

  01_Contrat_de_bail.pdf         a born-digital PDF (text layer)
  02_Resiliation_scan.pdf        a scanned letter + official form, image only (Nemotron Parse OCR)
  03_Facture_Rochat_photo.jpg    a phone photo of the plumber's bill (Nemotron Parse OCR)
  04_Enregistrement_cliente.wav  the client telling her story (Magpie TTS; the app transcribes it with ASR)
  05_Notes_entretien.txt         the lawyer's notes from the first meeting

Run from the repository root (needs the Magpie TTS NIM on :50052 for the recording):
    uv run --with reportlab python demo/make_case.py
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent / "case-moreau"
FONTS = Path("/usr/share/fonts/truetype/dejavu")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


# ---------------------------------------------------------------- 01 lease (text PDF)

LEASE = [
    ("h", "CONTRAT DE BAIL À LOYER POUR LOCAUX D'HABITATION"),
    ("p", "Entre les soussignés :"),
    ("p", "<b>Bailleur :</b> SI Béthusy-Soleil SA, représentée par sa gérance, Gérance Lac &amp; Jura SA, "
          "Rue du Petit-Chêne 18, 1003 Lausanne (ci-après « le bailleur »)."),
    ("p", "<b>Locataire :</b> Madame Claire Moreau, née le 14 mai 1986, de nationalité suisse "
          "(ci-après « la locataire »)."),
    ("s", "Art. 1 — Objet du bail"),
    ("p", "Le bailleur remet à bail à la locataire un appartement de 3,5 pièces au 3e étage de l'immeuble sis "
          "Avenue de Béthusy 42, 1005 Lausanne, comprenant : hall, séjour, deux chambres, cuisine agencée, "
          "salle de bains/WC, balcon, ainsi que la cave n° 7. Les locaux sont destinés exclusivement à "
          "l'habitation de la locataire et de sa famille."),
    ("s", "Art. 2 — Durée et résiliation"),
    ("p", "Le bail commence le 1er avril 2019 et prend fin le 31 mars 2020. Sauf avis de résiliation donné "
          "par écrit au moins trois mois à l'avance pour cette échéance, il se renouvelle tacitement aux "
          "mêmes conditions d'année en année, soit jusqu'au 31 mars de chaque année."),
    ("p", "La résiliation par le bailleur doit être notifiée au moyen de la formule officielle agréée par le "
          "canton de Vaud. Les échéances et délais fixés au présent article sont seuls applicables ; les "
          "termes usuels locaux ne s'appliquent pas."),
    ("s", "Art. 3 — Loyer et frais accessoires"),
    ("p", "Le loyer mensuel net est fixé à CHF 1'850.–. Il s'y ajoute un acompte mensuel de CHF 180.– pour "
          "le chauffage et l'eau chaude, faisant l'objet d'un décompte annuel au 30 juin. Le loyer et les "
          "acomptes, soit CHF 2'030.– au total, sont payables d'avance le premier de chaque mois."),
    ("s", "Art. 4 — Garantie"),
    ("p", "La locataire constitue une garantie de CHF 5'550.–, correspondant à trois mois de loyer net, "
          "déposée sur un compte bancaire bloqué au nom de la locataire."),
    ("s", "Art. 5 — Chauffage et eau chaude"),
    ("p", "L'appartement est équipé d'une chaudière murale individuelle au gaz qui assure le chauffage et la "
          "production d'eau chaude. Le bailleur en commande l'entretien annuel ; les frais d'entretien sont "
          "compris dans les frais accessoires. Les réparations de la chaudière sont à la charge du bailleur, "
          "à l'exception des menus travaux de nettoyage et d'entretien, à la charge de la locataire, dont le "
          "coût n'excède pas CHF 150.– par intervention."),
    ("s", "Art. 6 — Défauts"),
    ("p", "La locataire signale sans retard à la gérance tout défaut de la chose louée qu'elle ne doit pas "
          "réparer elle-même. En cas d'urgence en dehors des heures de bureau, elle s'adresse au service de "
          "piquet indiqué au tableau d'affichage de l'immeuble."),
    ("s", "Art. 7 — Usage de la chose louée"),
    ("p", "La locataire use de la chose louée avec le soin nécessaire et avec égards pour les voisins. Elle "
          "respecte le règlement de maison, notamment les heures de repos de 22 h à 7 h. La détention de "
          "chiens est soumise à l'accord écrit préalable du bailleur ; les petits animaux domestiques sont "
          "autorisés."),
    ("s", "Art. 8 — Sous-location"),
    ("p", "Toute sous-location, totale ou partielle, requiert le consentement écrit préalable du bailleur, "
          "auquel la locataire communique les conditions de la sous-location."),
    ("s", "Art. 9 — Travaux et visites"),
    ("p", "Le bailleur peut exécuter les travaux nécessaires à l'entretien de l'immeuble. Il annonce les "
          "visites et travaux en temps utile, en principe cinq jours à l'avance, sauf urgence. La locataire "
          "autorise la visite des locaux en vue de leur relocation pendant les trois mois précédant la fin "
          "du bail."),
    ("s", "Art. 10 — État des lieux"),
    ("p", "Un état des lieux d'entrée a été établi contradictoirement le 1er avril 2019 et signé par les "
          "parties. Il fait partie intégrante du présent contrat. Un état des lieux de sortie sera établi à "
          "la restitution des locaux."),
    ("s", "Art. 11 — Buanderie"),
    ("p", "L'usage de la buanderie commune est réglé par le tableau de répartition affiché dans celle-ci. "
          "La locataire dispose du mardi et du vendredi."),
    ("s", "Art. 12 — Dispositions complémentaires"),
    ("p", "Les Règles et usages locatifs du canton de Vaud (RULV) font partie intégrante du présent contrat, "
          "sous réserve de l'art. 2 ci-dessus. Pour le surplus, les dispositions du Code des obligations "
          "(art. 253 ss CO) sont applicables. For : Lausanne."),
    ("p", "Fait en deux exemplaires, à Lausanne, le 12 mars 2019."),
    ("sig", "Pour le bailleur : Gérance Lac &amp; Jura SA — O. Chappuis<br/>La locataire : C. Moreau"),
]


def lease_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    pdfmetrics.registerFont(TTFont("Serif", str(FONTS / "DejaVuSerif.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Bold", str(FONTS / "DejaVuSerif-Bold.ttf")))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold")
    body = ParagraphStyle("b", fontName="Serif", fontSize=10, leading=14.5, spaceAfter=6, alignment=4)
    styles = {
        "h": ParagraphStyle("h", parent=body, fontName="Serif-Bold", fontSize=13, alignment=1, spaceAfter=16),
        "s": ParagraphStyle("s", parent=body, fontName="Serif-Bold", fontSize=10.5, spaceBefore=8, spaceAfter=4),
        "p": body,
        "sig": ParagraphStyle("sig", parent=body, spaceBefore=24, alignment=0),
    }

    def footer(canvas, doc):
        canvas.setFont("Serif", 8)
        canvas.drawString(2.2 * cm, 1.3 * cm, "Bail Av. de Béthusy 42, 3e étage — Moreau")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.3 * cm, f"Page {doc.page}")

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2.2 * cm, bottomMargin=2.2 * cm, title="Contrat de bail — Moreau",
                            author="Gérance Lac & Jura SA")
    flow = []
    for kind, text in LEASE:
        flow.append(Paragraph(text, styles[kind]))
        if kind == "h":
            flow.append(Spacer(1, 4))
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)


# ---------------------------------------------------------------- page rendering helpers

W, H = 1240, 1754  # A4 at 150 dpi


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return [*lines, line] if line else lines


def paragraph(draw, xy, text, f, width, leading, fill=(20, 20, 20)) -> int:
    x, y = xy
    for line in wrap(draw, text, f, width):
        draw.text((x, y), line, font=f, fill=fill)
        y += leading
    return y


def scanned(page: Image.Image, angle: float) -> Image.Image:
    """Grey, slightly skewed, a little blurred and speckled, like an office scanner."""
    g = page.convert("L").rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    g = g.filter(ImageFilter.GaussianBlur(0.7))
    a = np.asarray(g, dtype=np.float32)
    a = a * 0.93 + 8 + np.random.default_rng(1).normal(0, 7, a.shape)
    a[:, :14] *= 0.55  # dark scanner edge
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- 02 termination (scanned PDF)

def termination_pages() -> list[Image.Image]:
    serif, bold, small = font("DejaVuSerif.ttf", 23), font("DejaVuSerif-Bold.ttf", 23), font("DejaVuSans.ttf", 18)
    left, width = 130, W - 260

    p1 = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(p1)
    d.text((left, 110), "Gérance Lac & Jura SA", font=font("DejaVuSerif-Bold.ttf", 34), fill=(20, 20, 20))
    d.text((left, 158), "Rue du Petit-Chêne 18  ·  1003 Lausanne  ·  021 000 00 00", font=small, fill=(60, 60, 60))
    d.line((left, 195, W - left, 195), fill=(40, 40, 40), width=2)
    d.text((left, 250), "RECOMMANDÉ", font=bold, fill=(20, 20, 20))
    y = 330
    for line in ("Madame", "Claire Moreau", "Avenue de Béthusy 42", "1005 Lausanne"):
        d.text((700, y), line, font=serif, fill=(20, 20, 20))
        y += 34
    d.text((left, 520), "Lausanne, le 31 août 2026", font=serif, fill=(20, 20, 20))
    d.text((left, 580), "Concerne : appartement de 3,5 pièces, 3e étage, Avenue de Béthusy 42 —",
           font=bold, fill=(20, 20, 20))
    d.text((left, 612), "résiliation de bail", font=bold, fill=(20, 20, 20))
    y = 690
    for text in (
        "Madame,",
        "Au nom et pour le compte de notre mandante, SI Béthusy-Soleil SA, nous vous notifions par la "
        "présente la résiliation du bail de l'appartement cité en marge pour le 31 décembre 2026. Vous "
        "trouverez ci-joint la formule officielle dûment remplie.",
        "Suite aux nombreuses difficultés rencontrées ces derniers mois dans nos relations, notre mandante "
        "ne souhaite pas poursuivre le rapport de bail avec vous.",
        "Nous vous prions de prendre contact avec notre service technique afin de fixer la date de l'état "
        "des lieux de sortie, et de nous restituer l'ensemble des clés à cette occasion.",
        "Nous vous prions d'agréer, Madame, nos salutations distinguées.",
    ):
        y = paragraph(d, (left, y), text, serif, width, 35) + 22
    d.text((700, y + 20), "Gérance Lac & Jura SA", font=serif, fill=(20, 20, 20))
    d.text((700, y + 60), "O. Chappuis", font=font("DejaVuSansMono-Oblique.ttf", 30), fill=(25, 45, 110))
    d.text((700, y + 105), "Olivier Chappuis, gérant", font=serif, fill=(20, 20, 20))
    d.text((left, H - 150), "Annexe : formule officielle de résiliation", font=small, fill=(60, 60, 60))
    # the tenant's note in blue ink, written in the space under "RECOMMANDÉ"
    note = Image.new("RGBA", (560, 60), (0, 0, 0, 0))
    ImageDraw.Draw(note).text((0, 8), "Retiré à la poste le 2.9.2026 - CM", font=font("DejaVuSansMono-Oblique.ttf", 26),
                              fill=(25, 45, 140, 255))
    note = note.rotate(3, expand=True, resample=Image.BICUBIC)
    p1.paste(note, (left - 10, 410), note)

    p2 = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(p2)
    d.text((left, 100), "AVIS DE RÉSILIATION DE BAIL", font=font("DejaVuSerif-Bold.ttf", 34), fill=(20, 20, 20))
    d.text((left, 150), "pour locaux d'habitation — formule officielle (art. 266l al. 2 CO)", font=serif,
           fill=(20, 20, 20))
    d.text((left, 185), "Canton de Vaud", font=serif, fill=(20, 20, 20))
    y = 260
    fields = [
        ("Bailleur", "SI Béthusy-Soleil SA, p.a. Gérance Lac & Jura SA, Rue du Petit-Chêne 18, 1003 Lausanne"),
        ("Locataire", "Madame Claire Moreau"),
        ("Chose louée", "Appartement de 3,5 pièces, 3e étage, cave n° 7, Avenue de Béthusy 42, 1005 Lausanne"),
        ("Le bail est résilié pour le", "31 décembre 2026"),
        ("Motif du congé", "Communiqué sur demande (art. 271 al. 2 CO)"),
        ("Lieu et date", "Lausanne, le 31 août 2026"),
    ]
    for label, value in fields:
        n = len(wrap(d, value, bold, width - 32))
        d.rectangle((left, y, W - left, y + 62 + 30 * n), outline=(40, 40, 40), width=2)
        d.text((left + 16, y + 10), label, font=small, fill=(60, 60, 60))
        paragraph(d, (left + 16, y + 38), value, bold, width - 32, 30)
        y += 78 + 30 * n
    y += 20
    d.text((left, y), "Indications à l'intention du locataire", font=bold, fill=(20, 20, 20))
    y += 48
    for text in (
        "1. Si le locataire entend contester le congé, il doit saisir l'autorité de conciliation dans les "
        "30 jours qui suivent la réception du congé (art. 273 al. 1 CO).",
        "2. Le locataire peut demander la prolongation du bail ; la requête doit être adressée à l'autorité "
        "de conciliation dans les 30 jours qui suivent la réception du congé (art. 273 al. 2 CO).",
        "3. Autorité compétente : Commission de conciliation en matière de baux à loyer du district de "
        "Lausanne.",
    ):
        y = paragraph(d, (left, y), text, font("DejaVuSerif.ttf", 21), width, 31) + 14
    d.text((700, H - 260), "Pour le bailleur :", font=serif, fill=(20, 20, 20))
    d.text((700, H - 215), "O. Chappuis", font=font("DejaVuSansMono-Oblique.ttf", 30), fill=(25, 45, 110))
    return [scanned(p1, 0.6), scanned(p2, -0.4)]


# ---------------------------------------------------------------- 03 plumber's bill (phone photo)

ITEMS = [
    ("Déplacement urgence (zone Lausanne)", "1", 95.00),
    ("Main-d'œuvre technicien, 10.03.2026", "3,5 h", 3.5 * 118.00),
    ("Carte électronique de commande chaudière", "1", 486.00),
    ("Sonde de température départ", "1", 78.50),
    ("Contrôle de combustion et remise en service", "1", 82.00),
]


def chf(x: float) -> str:
    whole, cents = f"{abs(x):.2f}".split(".")
    return ("-" if round(x, 2) < 0 else "") + f"{int(whole):,}".replace(",", "'") + "." + cents


def bill_photo(path: Path) -> float:
    pw, ph = 1100, 1500
    page = Image.new("RGB", (pw, ph), (250, 248, 242))
    d = ImageDraw.Draw(page)
    sans, bold = font("DejaVuSans.ttf", 22), font("DejaVuSans-Bold.ttf", 22)
    d.text((70, 70), "CHAUFFAGE ROCHAT Sàrl", font=font("DejaVuSans-Bold.ttf", 38), fill=(170, 40, 30))
    d.text((70, 122), "Installations sanitaires et chauffage", font=sans, fill=(60, 60, 60))
    d.text((70, 152), "Chemin des Fleurettes 5  ·  1007 Lausanne  ·  CHE-000.000.000 TVA", font=sans, fill=(60, 60, 60))
    y = 240
    for line in ("Madame Claire Moreau", "Avenue de Béthusy 42", "1005 Lausanne"):
        d.text((640, y), line, font=sans, fill=(20, 20, 20))
        y += 30
    d.text((70, 360), "FACTURE N° 2026-0317", font=font("DejaVuSans-Bold.ttf", 30), fill=(20, 20, 20))
    d.text((70, 405), "Date : 12 mars 2026        Intervention : 10 mars 2026", font=sans, fill=(20, 20, 20))
    d.text((70, 440), "Objet : dépannage chaudière murale gaz, appartement 3e étage", font=sans, fill=(20, 20, 20))
    y = 510
    d.line((70, y, pw - 70, y), fill=(40, 40, 40), width=2)
    d.text((80, y + 10), "Désignation", font=bold, fill=(20, 20, 20))
    d.text((720, y + 10), "Qté", font=bold, fill=(20, 20, 20))
    d.text((pw - 80, y + 10), "CHF", font=bold, fill=(20, 20, 20), anchor="ra")
    y += 50
    d.line((70, y, pw - 70, y), fill=(40, 40, 40), width=1)
    subtotal = 0.0
    for name, qty, amount in ITEMS:
        subtotal += amount
        d.text((80, y + 14), name, font=sans, fill=(20, 20, 20))
        d.text((720, y + 14), qty, font=sans, fill=(20, 20, 20))
        d.text((pw - 80, y + 14), chf(amount), font=sans, fill=(20, 20, 20), anchor="ra")
        y += 52
    vat = round(subtotal * 0.081, 2)
    total = round((subtotal + vat) * 20) / 20  # rounded to 5 centimes
    d.line((70, y + 10, pw - 70, y + 10), fill=(40, 40, 40), width=1)
    y += 26
    for label, amount, f in (("Sous-total", subtotal, sans), ("TVA 8,1 %", vat, sans),
                             ("Arrondi", total - subtotal - vat, sans), ("TOTAL", total, bold)):
        d.text((560, y), label, font=f, fill=(20, 20, 20))
        d.text((pw - 80, y), chf(amount), font=f, fill=(20, 20, 20), anchor="ra")
        y += 38
    y += 30
    y = paragraph(d, (70, y), "Remarque : selon la cliente, chaudière hors service depuis le 23.02.2026 "
                              "(ni chauffage ni eau chaude). Carte de commande défectueuse remplacée, sonde "
                              "changée, installation contrôlée et remise en service.", sans, pw - 140, 32)
    paragraph(d, (70, y + 20), "Payable dans les 30 jours. Merci de votre confiance.", sans, pw - 140, 32)
    # "paid" stamp in blue ink
    stamp = Image.new("RGBA", (360, 130), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    sd.rounded_rectangle((4, 4, 356, 126), 14, outline=(30, 60, 160, 230), width=5)
    sd.text((180, 38), "PAYÉ", font=font("DejaVuSans-Bold.ttf", 40), fill=(30, 60, 160, 230), anchor="mm")
    sd.text((180, 92), "12.03.2026 — comptant", font=font("DejaVuSans-Bold.ttf", 22), fill=(30, 60, 160, 230),
            anchor="mm")
    stamp = stamp.rotate(12, expand=True)
    page.paste(stamp, (600, ph - 330), stamp)

    # the photo: page lying on a wooden table, a little turned, light falling from the left
    photo = Image.new("RGB", (1300, 1700), (120, 86, 58))
    grain = np.random.default_rng(3).normal(0, 9, (1700, 1300, 1))
    photo = Image.fromarray(np.clip(np.asarray(photo, dtype=np.float32) + grain, 0, 255).astype(np.uint8))
    turned = page.rotate(2.3, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
    mask = Image.new("L", page.size, 255).rotate(2.3, expand=True)
    photo.paste(turned, (95, 85), mask)
    a = np.asarray(photo, dtype=np.float32)
    light = np.linspace(1.08, 0.78, a.shape[1])[None, :, None] * np.linspace(1.0, 0.9, a.shape[0])[:, None, None]
    a = np.clip(a * light, 0, 255).astype(np.uint8)
    Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)).save(path, quality=82)
    return total


# ---------------------------------------------------------------- 04 client recording (Magpie TTS)

RECORDING = [
    "Bonjour. Je m'appelle Claire Moreau. J'habite depuis avril 2019 un trois pièces et demie à l'avenue "
    "de Béthusy 42, à Lausanne, avec ma fille de huit ans.",
    "Le 23 février, la chaudière de l'appartement est tombée en panne : plus de chauffage et plus d'eau "
    "chaude. J'ai appelé la gérance le jour même, puis je leur ai écrit par courriel le lendemain. On m'a "
    "promis un technicien, mais personne n'est venu.",
    "Le 5 mars, je leur ai envoyé une lettre recommandée en leur donnant trois jours pour intervenir. "
    "Toujours rien. Alors le 10 mars, j'ai fait venir un chauffagiste, l'entreprise Rochat, qui a réparé "
    "la chaudière. J'ai payé la facture moi-même, un peu plus de mille deux cents francs.",
    "En avril, j'ai demandé à la gérance de me rembourser la facture et de réduire le loyer pour les "
    "deux semaines sans chauffage. Ils ont refusé, en disant que j'aurais dû attendre leur technicien.",
    "J'ai donc saisi la commission de conciliation en mai. À l'audience du 30 juin, on a trouvé un "
    "accord : la gérance m'a remboursé la facture et m'a accordé trois cents francs de réduction de loyer.",
    "Et puis, le 2 septembre, j'ai retiré à la poste une lettre recommandée : ils résilient mon bail pour "
    "le 31 décembre. La lettre parle des nombreuses difficultés dans nos relations. Je n'ai jamais eu un "
    "seul retard de loyer en sept ans.",
    "Ma fille va à l'école dans le quartier, et je voudrais vraiment rester. Est-ce que je peux contester "
    "ce congé, et jusqu'à quand ?",
]


def recording(path: Path, voice: str = "Magpie-Multilingual.FR-FR.Louise", rate: int = 22050) -> float:
    import riva.client

    tts = riva.client.SpeechSynthesisService(riva.client.Auth(uri="localhost:50052"))
    pause = np.zeros(int(rate * 0.6), dtype=np.int16).tobytes()
    pcm = b"".join(tts.synthesize(text, voice, "fr-FR", sample_rate_hz=rate).audio + pause for text in RECORDING)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return len(pcm) / 2 / rate


# ---------------------------------------------------------------- 05 lawyer's notes

NOTES = """Notes — premier entretien, 15 septembre 2026
Dossier : Moreau c. SI Béthusy-Soleil SA (gérance : Gérance Lac & Jura SA)
Mandante : Claire Moreau, 40 ans, infirmière au CHUV (taux 80 %), mère d'une fille de 8 ans
(scolarisée au collège du quartier). Locataire depuis le 01.04.2019.

Chronologie (selon la cliente et les pièces)
- 12.03.2019  bail signé ; début 01.04.2019 ; loyer net CHF 1'850.– + CHF 180.– d'acomptes
- 23.02.2026  panne de la chaudière murale (ni chauffage ni eau chaude) ; téléphone à la gérance
- 24.02.2026  courriel à la gérance ; promesse d'un technicien, aucune intervention
- 05.03.2026  lettre recommandée : délai de 3 jours pour réparer
- 10.03.2026  réparation par Chauffage Rochat Sàrl, mandaté par la cliente
- 12.03.2026  facture n° 2026-0317, payée comptant par la cliente
- 20.04.2026  lettre de la cliente : remboursement de la facture + réduction de loyer
- 04.05.2026  refus de la gérance (« il fallait attendre notre technicien »)
- 18.05.2026  requête à la Commission de conciliation du district de Lausanne
- 30.06.2026  audience : transaction — remboursement de la facture et réduction de CHF 300.–
              (la cliente dit avoir été payée le 15.07.2026)
- 31.08.2026  résiliation (formule officielle) pour le 31.12.2026, envoi recommandé
- 02.09.2026  la cliente retire le pli au guichet postal

Pièces reçues
- contrat de bail (PDF)
- lettre de résiliation + formule officielle (scan)
- facture Rochat (photo)
Manque : procès-verbal de la transaction du 30.06.2026 (à demander à la cliente),
         courriel du 24.02, lettre recommandée du 05.03, refus du 04.05.

Le litige sur la chaudière est réglé par la transaction du 30.06.2026 et n'est pas à rouvrir :
le mandat porte uniquement sur le congé du 31.08.2026.

À ce jour (15.09.2026), la cliente n'a entrepris aucune démarche contre le congé.
Aucun retard de paiement, aucune plainte de voisinage selon la cliente.
Objectif de la cliente : rester dans l'appartement ; à défaut, obtenir du temps
(année scolaire de sa fille jusqu'à fin juin 2027).

À vérifier
- délai pour contester le congé — date limite exacte
- motifs d'annulation possibles du congé
- date d'effet du congé au regard de l'échéance contractuelle (art. 2 du bail)
- subsidiairement : prolongation du bail
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lease_pdf(OUT / "01_Contrat_de_bail.pdf")
    pages = termination_pages()
    pages[0].save(OUT / "02_Resiliation_scan.pdf", save_all=True, append_images=pages[1:], resolution=150)
    total = bill_photo(OUT / "03_Facture_Rochat_photo.jpg")
    seconds = recording(OUT / "04_Enregistrement_cliente.wav")
    (OUT / "05_Notes_entretien.txt").write_text(NOTES, encoding="utf-8")
    print(f"bill total CHF {chf(total)}; recording {seconds:.0f} s; files in {OUT}")


if __name__ == "__main__":
    main()
