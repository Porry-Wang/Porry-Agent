import json
import os
import types

import pytest

from porry_agent import journal


def manifest(root, transaction_id):
    path = root / journal.TRANSACTION_DIRECTORY / transaction_id / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def begin_update(root, name="file.py", old=b"old\n", new=b"new\n", mode=0o644):
    path = root / name
    path.write_bytes(old)
    path.chmod(mode)
    return journal.begin_transaction(
        root,
        [journal.JournalFile(name, "update", old, mode, new, mode)],
    )


def begin_add(root, name="new.py", new=b"new\n", mode=0o644):
    return journal.begin_transaction(
        root,
        [journal.JournalFile(name, "add", None, None, new, mode)],
    )


def begin_delete(root, name="old.py", old=b"old\n", mode=0o644):
    path = root / name
    path.write_bytes(old)
    path.chmod(mode)
    return journal.begin_transaction(
        root,
        [journal.JournalFile(name, "delete", old, mode, None, None)],
    )


def begin_move(
    root, source="source.py", target="target.py", old=b"old\n", new=b"new\n"
):
    path = root / source
    path.write_bytes(old)
    path.chmod(0o755)
    return journal.begin_transaction(
        root,
        [
            journal.JournalFile(source, "delete", old, 0o755, None, None),
            journal.JournalFile(target, "add", None, None, new, 0o755),
        ],
    )


def test_begin_transaction_writes_prepared_manifest_and_backup(tmp_path):
    transaction_id = begin_update(tmp_path)

    data = manifest(tmp_path, transaction_id)
    directory = tmp_path / journal.TRANSACTION_DIRECTORY / transaction_id

    assert data["state"] == "prepared"
    assert data["entries"][0]["path"] == "file.py"
    assert (directory / "backups" / "0.bak").read_bytes() == b"old\n"


def test_mark_state_persists_transition(tmp_path):
    transaction_id = begin_update(tmp_path)

    journal.mark_state(tmp_path, transaction_id, "applying")
    assert manifest(tmp_path, transaction_id)["state"] == "applying"

    journal.mark_state(tmp_path, transaction_id, "committed")
    assert manifest(tmp_path, transaction_id)["state"] == "committed"


def test_recover_prepared_update_cleans_without_changing_file(tmp_path):
    transaction_id = begin_update(tmp_path)

    report = journal.recover_journal(tmp_path)

    assert report.rolled_back == (transaction_id,)
    assert (tmp_path / "file.py").read_bytes() == b"old\n"
    assert not (tmp_path / journal.TRANSACTION_DIRECTORY / transaction_id).exists()


def test_recover_applying_update_restores_old_value(tmp_path):
    transaction_id = begin_update(tmp_path)
    journal.mark_state(tmp_path, transaction_id, "applying")
    (tmp_path / "file.py").write_bytes(b"new\n")

    report = journal.recover_journal(tmp_path)

    assert report.rolled_back == (transaction_id,)
    assert (tmp_path / "file.py").read_bytes() == b"old\n"


def test_recover_applying_add_removes_new_file(tmp_path):
    transaction_id = begin_add(tmp_path)
    journal.mark_state(tmp_path, transaction_id, "applying")
    (tmp_path / "new.py").write_bytes(b"new\n")

    report = journal.recover_journal(tmp_path)

    assert report.rolled_back == (transaction_id,)
    assert not (tmp_path / "new.py").exists()


def test_recover_applying_delete_restores_file_and_mode(tmp_path, assert_mode):
    transaction_id = begin_delete(tmp_path, mode=0o755)
    journal.mark_state(tmp_path, transaction_id, "applying")
    (tmp_path / "old.py").unlink()

    report = journal.recover_journal(tmp_path)

    assert report.rolled_back == (transaction_id,)
    assert (tmp_path / "old.py").read_bytes() == b"old\n"
    assert_mode(tmp_path / "old.py", 0o755)


def test_recover_move_crash_after_target_write(tmp_path):
    transaction_id = begin_move(tmp_path)
    journal.mark_state(tmp_path, transaction_id, "applying")
    (tmp_path / "target.py").write_bytes(b"new\n")

    report = journal.recover_journal(tmp_path)

    assert report.rolled_back == (transaction_id,)
    assert (tmp_path / "source.py").read_bytes() == b"old\n"
    assert not (tmp_path / "target.py").exists()


def test_recover_move_crash_after_source_delete(tmp_path, assert_mode):
    transaction_id = begin_move(tmp_path)
    journal.mark_state(tmp_path, transaction_id, "applying")
    (tmp_path / "target.py").write_bytes(b"new\n")
    (tmp_path / "source.py").unlink()

    report = journal.recover_journal(tmp_path)

    assert report.rolled_back == (transaction_id,)
    assert (tmp_path / "source.py").read_bytes() == b"old\n"
    assert_mode(tmp_path / "source.py", 0o755)
    assert not (tmp_path / "target.py").exists()


def test_recover_update_conflict_preserves_external_change(tmp_path):
    transaction_id = begin_update(tmp_path)
    journal.mark_state(tmp_path, transaction_id, "applying")
    (tmp_path / "file.py").write_bytes(b"external\n")

    report = journal.recover_journal(tmp_path)

    assert report.needs_attention == (transaction_id,)
    assert report.blocked is True
    assert (tmp_path / "file.py").read_bytes() == b"external\n"
    assert manifest(tmp_path, transaction_id)["state"] == "needs_attention"


def test_recover_add_conflict_preserves_external_file(tmp_path):
    transaction_id = begin_add(tmp_path)
    journal.mark_state(tmp_path, transaction_id, "applying")
    (tmp_path / "new.py").write_bytes(b"external\n")

    report = journal.recover_journal(tmp_path)

    assert report.needs_attention == (transaction_id,)
    assert (tmp_path / "new.py").read_bytes() == b"external\n"


def test_recover_delete_conflict_preserves_external_file(tmp_path):
    transaction_id = begin_delete(tmp_path)
    journal.mark_state(tmp_path, transaction_id, "applying")
    (tmp_path / "old.py").write_bytes(b"external\n")

    report = journal.recover_journal(tmp_path)

    assert report.needs_attention == (transaction_id,)
    assert (tmp_path / "old.py").read_bytes() == b"external\n"


def test_recover_committed_only_cleans_journal(tmp_path):
    transaction_id = begin_update(tmp_path)
    journal.mark_state(tmp_path, transaction_id, "applying")
    (tmp_path / "file.py").write_bytes(b"new\n")
    journal.mark_state(tmp_path, transaction_id, "committed")

    report = journal.recover_journal(tmp_path)

    assert report.cleaned == (transaction_id,)
    assert (tmp_path / "file.py").read_bytes() == b"new\n"
    assert not (tmp_path / journal.TRANSACTION_DIRECTORY / transaction_id).exists()


def test_recovery_is_idempotent(tmp_path):
    transaction_id = begin_update(tmp_path)
    journal.mark_state(tmp_path, transaction_id, "applying")
    (tmp_path / "file.py").write_bytes(b"new\n")

    first = journal.recover_journal(tmp_path)
    second = journal.recover_journal(tmp_path)

    assert first.rolled_back == (transaction_id,)
    assert second == journal.RecoveryReport()
    assert (tmp_path / "file.py").read_bytes() == b"old\n"


def test_journal_lock_rejects_concurrent_recovery(tmp_path):
    with journal.journal_lock(tmp_path):
        with pytest.raises(journal.JournalError) as exc_info:
            journal.recover_journal(tmp_path)

    assert exc_info.value.error_type == "journal_locked"
    assert exc_info.value.retryable is True


def test_journal_lock_releases_after_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with journal.journal_lock(tmp_path):
            raise RuntimeError("boom")

    report = journal.recover_journal(tmp_path)

    assert report == journal.RecoveryReport()


def test_corrupt_manifest_is_reported_and_preserved(tmp_path):
    transaction_id = begin_update(tmp_path)
    path = tmp_path / journal.TRANSACTION_DIRECTORY / transaction_id / "manifest.json"
    path.write_text("{not json", encoding="utf-8")

    report = journal.recover_journal(tmp_path)

    assert report.errors
    assert report.blocked is True
    assert path.exists()


@pytest.mark.parametrize(
    "entries",
    [
        [123],
        [{"kind": "update"}],
        [{"path": "file.py", "kind": "rename"}],
        [{"path": "file.py", "kind": "update", "backup_name": 7}],
        [{"path": "file.py", "kind": "update", "mode": "644"}],
    ],
)
def test_corrupt_entry_shape_is_reported_not_raised(tmp_path, entries):
    transaction_id = begin_update(tmp_path)
    journal.mark_state(tmp_path, transaction_id, "applying")
    path = tmp_path / journal.TRANSACTION_DIRECTORY / transaction_id / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["entries"] = entries
    path.write_text(json.dumps(data), encoding="utf-8")

    report = journal.recover_journal(tmp_path)

    assert report.errors
    assert report.blocked is True
    assert path.exists()
    assert (tmp_path / "file.py").read_bytes() == b"old\n"


def test_begin_transaction_rejects_path_escape(tmp_path):
    with pytest.raises(journal.JournalError) as exc_info:
        journal.begin_transaction(
            tmp_path,
            [journal.JournalFile("../outside.py", "update", b"a", 0o644, b"b", 0o644)],
        )

    assert exc_info.value.error_type == "invalid_args"


def test_atomic_write_bytes_preserves_mode(tmp_path, assert_mode):
    path = tmp_path / "file.py"
    path.write_bytes(b"old\n")
    path.chmod(0o755)

    journal.atomic_write_bytes(path, b"new\n")

    assert path.read_bytes() == b"new\n"
    assert_mode(path, 0o755)


def test_recover_multiple_transactions(tmp_path):
    committed = begin_update(tmp_path, name="committed.py")
    journal.mark_state(tmp_path, committed, "committed")
    applying = begin_update(tmp_path, name="applying.py")
    journal.mark_state(tmp_path, applying, "applying")
    (tmp_path / "applying.py").write_bytes(b"new\n")

    report = journal.recover_journal(tmp_path)

    assert report.cleaned == (committed,)
    assert report.rolled_back == (applying,)
    assert (tmp_path / "applying.py").read_bytes() == b"old\n"


def test_nt_lock_routes_to_msvcrt(tmp_path, monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        LK_NBLCK=1,
        LK_UNLCK=2,
        locking=lambda descriptor, mode, nbytes: calls.append((mode, nbytes)),
    )
    monkeypatch.setattr(journal, "os", types.SimpleNamespace(name="nt"))
    monkeypatch.setattr(journal, "msvcrt", fake, raising=False)

    descriptor = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        journal._lock_descriptor(descriptor)
        journal._unlock_descriptor(descriptor)
    finally:
        os.close(descriptor)

    assert calls == [(fake.LK_NBLCK, 1), (fake.LK_UNLCK, 1)]
