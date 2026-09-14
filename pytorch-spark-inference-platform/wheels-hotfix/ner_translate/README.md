# ner_translate dependency hotfix directory

Empty by default — everything in this directory except this file is
gitignored. **This is not the normal way to change `ner_translate`'s
dependencies.** For that, edit `models/pipelines/ner_translate/requirements.txt`,
regenerate the wheelhouse (`deploy/scripts/build_ner_translate_wheelhouse.sh`),
and rebuild the image.

## When to use this instead

Only when you need to patch a dependency on an **already-deployed,
air-gapped** `ner_translate` node and cannot do a full rebuild + `docker
save` + transfer cycle right now (see `docs/air_gapped_dep.md` §7 for the
equivalent pattern already used for code-only changes).

## How

1. On an internet-connected machine, download the replacement wheel(s) —
   e.g. `pip download somepackage==1.2.4 -d wheels-hotfix/ner_translate/`.
2. Transfer the `.whl` file(s) to the airgapped node via your normal
   transfer method, into this exact directory (it's already bind-mounted
   into both the `ner-translate-master` and `ner-translate-worker`
   containers as `/wheels-hotfix` — see `deploy/docker-compose.ner_translate.yml`).
3. Restart the container(s): `docker compose -f deploy/docker-compose.ner_translate.yml restart`.
   `deploy/apply_wheels_hotfix.sh` runs automatically on start, installs
   every `.whl` found here, and logs what it installed.

## Important: apply to every node, not just one

If the cluster has more than one worker, **drop the same wheel(s) into this
directory on every node** before restarting any of them. A hotfix applied
to only one node leaves the driver and executors on different dependency
versions — this repo has already hit a real bug from exactly that kind of
driver/worker environment mismatch (see `docs/FRAMEWORK_OVERVIEW.md`).
Nothing in this mechanism detects or warns about that drift for you.

## Cleaning up after a hotfix

Once the real fix has been rebuilt into the image (updated
`models/pipelines/ner_translate/requirements.txt` +
`build_ner_translate_wheelhouse.sh` + a fresh `docker build`), delete the
`.whl` files from this directory on every node — an override left behind
here will keep taking precedence over whatever the image now bakes in.
