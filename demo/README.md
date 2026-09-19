# Demo kit

Two questions to show the assistant answering and asking back, and one client case with the kinds of
files a law firm actually receives. Everything here was run against the app on 19 September 2026;
timings are from those runs.

## 1. Two questions (English and French)

### Question 1 — answers right away

> How long does a tenant have to challenge a termination of the lease, and what happens if the deadline is missed?

> Dans quel délai un locataire doit-il contester la résiliation de son bail, et que se passe-t-il s'il laisse passer ce délai ?

Expected: 30 days from receipt (Art. 273 al. 1 CO/OR), a strict deadline that can't be extended; if
it is missed, the termination can no longer be challenged. Two to four citations. About 30 s in French;
the English run took 95 s because it made 8 tool calls, so start with the French one if time is short.

Point out: the "Thinking" chain, citations that open the decision, statute links (Art. 273 OR).

### Question 2 — asks you back

> My employer let me go yesterday. Am I still entitled to my salary?

> Mon employeur m'a licencié hier. Ai-je encore droit à mon salaire ?

The answer turns on whether the contract was ended immediately (Art. 337 CO) or with notice, so the
agent searches first and then asks one question with answer buttons, e.g. *"Was the termination
ordinary (with notice period) or immediate (without notice, for just cause)?"* — **Ordinary termination
with notice period** / **Immediate termination without notice**. Click a button and it answers in the
language of the question. It asked back in all 6 test runs, after 15–40 s; the exact wording and the
buttons vary from run to run, and the French one sometimes bundles two questions into one.

## 2. The case: Moreau c. SI Béthusy-Soleil SA (`case-moreau/`)

An invented Lausanne tenancy, in French. Claire Moreau's boiler broke on 23 February 2026; the property
manager did nothing, so she had it repaired and paid the plumber herself. She took them to the
conciliation authority and won a settlement on 30 June 2026. On 2 September 2026 she collected a
registered letter terminating her lease for 31 December 2026, "following the many difficulties in
our relations". She wants to stay; her daughter is at the local school.

| File | What it is | How the app reads it |
|---|---|---|
| `01_Contrat_de_bail.pdf` | the lease, 2 pages, born digital | Nemotron Parse, 3 s |
| `02_Resiliation_scan.pdf` | scanned termination letter + official form, image only, with the tenant's handwritten "Retiré à la poste le 2.9.2026" | Nemotron Parse OCR, 4 s |
| `03_Facture_Rochat_photo.jpg` | phone photo of the plumber's bill (CHF 1'248.00, stamped "PAYÉ") | Nemotron Parse OCR, 2 s; the line items come back as a table |
| `04_Enregistrement_cliente.wav` | the client telling her story, 97 s, French | Nemotron ASR, 10 s. **Matters page only**: the chat composer doesn't accept audio |
| `05_Notes_entretien.txt` | the lawyer's notes from the first meeting (15.09.2026): timeline, what's missing, what to check | plain text |

The transcript comes back with a few realistic mistakes ("Béthuzy", "Rocha", "compter ce congé"),
which shows the pipeline working with an imperfect recording.

### What a good answer contains (presenter's cheat sheet)

- **Deadline to contest: 2 October 2026.** 30 days from receipt (Art. 273 al. 1 CO), counted from
  2 September, when she collected the letter; the letter's own date is 31 August.
- **Grounds to annul:** Art. 271 al. 1 CO (contrary to good faith; the only stated reason is "difficulties"),
  Art. 271a al. 1 let. a CO (retaliation for asserting her rights), and above all
  **Art. 271a al. 1 let. e CO**: notice given within three years of a conciliation settlement with the
  landlord is annullable. The settlement was on 30 June 2026.
- **Effective date: 31 March 2027, not 31 December 2026.** Art. 2 of the lease only allows termination
  for 31 March each year, so the notice takes effect at the next valid date (Art. 266a al. 2 CO).
- **Fallback:** an extension of the lease (Art. 272 CO), to the end of the school year in June 2027.
- The repair itself (Art. 259b CO) is settled; the notes say not to reopen it.

## 3. How to show it

### A. Matters page: all five files at once (about 2–3 min)

1. **Matters** tab → drop all five files, including the recording, on "Drop files here" → **Open the matter**.
2. While it runs, point out the stages: intake (transcribing the recording, reading the scan), research
   one issue at a time, assessment, drafting.
3. Open a citation in the memo: a court decision opens in the side panel, and a client document opens
   on its page.
4. Download the memo as **Word**.

A finished matter is already on the Matters page as a backup: *"Résiliation de bail et congé pour fin
d'année"*. Read its memo before the demo. In the last run it correctly found **31 March 2027** and that
the deadline is still open, but it computed the deadline as 30 September (counting from 31 August
instead of 2 September), and it did not cite Art. 271a al. 1 let. e. Each run differs. When the memo
gets something wrong, that's a good moment for "every reference must be checked; that's why each
sentence carries its source".

### B. Assistant chat: attach the documents and ask

Attach `01`, `02`, `03` and `05` (paperclip), then ask. Each question below was asked once, in its own
conversation, with those four files attached:

| Question | Result |
|---|---|
| **Jusqu'à quelle date ma cliente peut-elle contester ce congé, et pour quels motifs pourrait-il être annulé ?** | Best: **2 October 2026** (it read the handwritten date on the scan), good faith, pretext, and the three-year protection after a conciliation. 50 s |
| **What does the plumber's invoice show, and was the tenant entitled to have the boiler repaired at the landlord's expense?** | Good: reads every line of the photo, total CHF 1,248, lease art. 5 (landlord pays repairs over CHF 150). 67 s |
| **Si le congé est valable, Mme Moreau peut-elle obtenir une prolongation du bail jusqu'à la fin de l'année scolaire de sa fille ?** | Good: decisions that granted extensions up to the end of the school year. 69 s (the English version also works, 90 s) |

Avoid:
- *"Does the termination take effect on 31 December 2026 under this lease?"* The chat answered **yes,
  which is wrong** (it's 31 March 2027). The Matters memo got this right.
- The deadline question **in English**. It got 2 October right but said the conciliation settlement
  doesn't help her, which is wrong. Ask it in French.

### C. Case Prep

From the finished matter, **Open in assistant** loads the whole case file (recording included) into a
chat; then ask the French deadline question above. This path wasn't tested with this case.

## Regenerating the files

```bash
uv run --with reportlab python demo/make_case.py
```

It needs the Magpie TTS NIM on :50052 for the recording. The documents come out the same every time;
the recording can differ slightly.

## App issues found while preparing this

- **Neither the agent nor the Matters pipeline is told today's date** (only the year, for search
  filters). A deadline question makes it guess whether the deadline has passed: one memo run
  concluded the client had already lost her right to contest. As a workaround, the notes say
  "À ce jour (15.09.2026), la cliente n'a entrepris aucune démarche".
- The memo shows internal document ids in its text, e.g. "(doc_91d98fc1b5a5)", and a party label in
  English ("Landlord (AG)") in a French memo.
- An OCR slip on the scan ("art. 2661" for 266l) carried through into an intake question.
- The parser drops page headers and pictures by design (`parsing.py`), so a letterhead or a
  boxed stamp on a scan is not read. That's why the receipt date is handwritten in the body.
