# Reference - Memory Portability

Memory portability lets an operator export persistent user memory from one
deployment and restore it before another agent starts. It is an operator
workflow, not an LLM skill.

## Setup

Choose an existing, dedicated absolute host directory for archives and prepare
a private host group for it. The launcher does not create the directory or
change its ownership or permissions. It mounts the directory at the fixed
container path `/memory-transfer`; the agent never accepts arbitrary runtime
export paths.

Memory transfer is supported on Linux hosts only. Create a dedicated group, add
the operator to it, and prepare an operator-owned directory with setgid and
owner/group-only access. Do not use GID `0` (`root`) as the transfer group.
Start a new login session after changing group membership, before starting
Omega.

```sh
sudo groupadd omega-transfer
sudo usermod -aG omega-transfer "$USER"
sudo install -d -o "$USER" -g omega-transfer -m 2770 "$HOME/omega-transfers"
transfer_gid="$(getent group omega-transfer | cut -d: -f3)"
```

```sh
scripts/omega start -p OpenAI -t telegram \
  --memory-transfer-dir "$HOME/omega-transfers" \
  --memory-transfer-gid "$transfer_gid" \
  --enable-memory-export
```

`--enable-memory-export` is required because export is disabled by default.
The transfer directory must grant `rwx` to its configured private group, set
the setgid bit so new archives inherit that group, and grant no permissions to
other users. It must be owned by the launching user or `root`. The launcher
rejects symbolic links, mismatched GIDs, missing directories, unsafe modes, and
POSIX ACLs; it also rejects filesystems whose POSIX ACL state cannot be
verified. It does not infer whether a host group is private: the operator must
use a group that contains only principals authorized to read memory archives.

## Export

In the active chat, request one component. The export runs immediately. For IRC,
Telegram, Slack, and Mattermost, the export handler requires the authenticated
user ID persisted by the channel authorization layer. WebSocket export requires
a configured `WS_TOKEN`; the handler derives a non-reversible principal from the
token so the credential itself is never used as an identifier:

```text
/memory-export history
/memory-export ltm
/memory-export both
```

Completion is delivered through the active channel and includes the filename,
record count, size, and SHA-256.

Archives contain selected persistent user memory only:

```text
manifest.json
history/history.metta
vector/collections.json
vector/records.jsonl
```

History is the conversation trace. LTM is logical user-memory records from
ChromaDB. Prompts, credentials, logs, skills, and other operational state are
not exported. SHA-256 detects corruption, not archive authorship.

## Import

Import is an administrative startup operation. The archive argument is a plain
filename in the chosen transfer directory:

```sh
scripts/omega start -d singularitynet/omega:<tag> -p OpenAI -t telegram \
  --memory-transfer-dir "$HOME/omega-transfers" \
  --memory-transfer-gid "$transfer_gid" \
  --memory-import omegaclaw-memory-<timestamp>.tar.gz \
  --memory-mode overwrite
```

`overwrite` replaces the selected components after validation and rollback
preparation. `append` preserves existing history and adds imported LTM records
under new IDs. Select a single component with `--only-history` or
`--only-vector`. Without either option, both components are imported.

The importer validates archive paths, checksums, manifest metadata, record
counts, and embedding compatibility before changing live memory. It runs before
the agent loop starts. A receipt prevents a completed archive import from
running again on container restart.

## Recovering an interrupted memory migration

When upgrading from OmegaClaw, the launcher copies `omegaclaw-memory` to
`omega-memory`. Keep the terminal open and do not interrupt the launcher or
Docker until it reports `Memory copied`. Keep `omegaclaw-memory` until you have
checked that the agent remembers its previous data.

If the copy is interrupted, follow the appropriate path below before importing
memory or starting the agent again. Use the same image tag and launcher options
as in your original start command.

### Retry the copy from OmegaClaw

Use this path only if `omega-memory` contains no imported memory or new agent
data that you need to keep. Retrying replaces this volume with a fresh copy of
`omegaclaw-memory`.

1. List the containers holding the destination volume:

   ```sh
   docker ps -a --filter volume=omega-memory --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}'
   ```

2. If the launcher reports `volume is in use` or names a leftover container,
   stop and remove the identified container. Replace `CONTAINER_ID` with its
   actual ID; do not remove unrelated containers or the source volume:

   ```sh
   docker rm -f CONTAINER_ID
   ```

   This also applies to stopped containers. Copy containers started by the
   v0.1.20 launcher have no migration label and need to be removed manually.
   The updated launcher removes labelled copy containers and the `omega`
   container automatically when retrying.

3. Repeat your original `scripts/omega start ...` command, without
   `--memory-import` or the `--only-history` and `--only-vector` flags. Wait for
   `Memory copied`, then check the agent's memory.

### Keep memory imported or used after the interruption

Do not retry the copy or run another import before preserving the current
`omega-memory` volume. Stop any remaining copy containers identified above and
stop the `omega` agent if it is running. Back up the destination volume before
changing migration state.

The v0.1.20 launcher could leave `.migration-started` in `omegaclaw-memory`
after a memory import. In particular, an `--only-vector` import can leave
`history.metta` unchanged. The updated launcher's history check cannot detect
this case: a normal start can delete the imported vectors and copy the old
memory over them.

If the import completed successfully and you want to keep the current memory,
clear only the stale migration marker before starting again:

```sh
docker volume inspect omegaclaw-memory omega-memory
docker run --rm --entrypoint rm --volume omegaclaw-memory:/from \
  singularitynet/omega:v0.1.20 -f /from/.migration-started
```

Run the second command only if both volumes exist. Then repeat your normal
start command without `--memory-import` or the `--only-history` and
`--only-vector` flags. Clearing the marker keeps the current destination volume;
it does not finish a partial copy. If you are unsure whether
the import or copy completed, preserve both volumes and ask for help before
continuing.

If imported memory has already been overwritten, the old OmegaClaw volume
cannot recover that import. Restore it from the original import archive or a
backup of `omega-memory`.

## Security

Archives are private operator data; use a dedicated transfer directory and keep
it protected. Do not point `--memory-transfer-dir` at `$HOME`, a shared
directory, or a symbolic link. The launcher never relaxes permissions on
existing paths; it adds the configured private group only to the `nobody` agent
processes, while the root entrypoint continues to initialize Nginx and startup
services normally.
Memory export is denied when channel authentication is disabled, no authenticated
channel user has been persisted, or WebSocket has no `WS_TOKEN`.
