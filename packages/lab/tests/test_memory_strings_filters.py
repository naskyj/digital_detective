"""dd_lab.memory.strings._keep - the noise-shape filters ported (and, in one
case, deliberately trimmed back) from mem_artifact_extractor.py. Every noise
example here was actually pulled from the real case's physmem.raw while
tuning this filter - not invented."""

from __future__ import annotations

from dd_lab.memory.strings import _keep, iter_strings


def _kept(s: str) -> bool:
    return _keep(s, set()) is not None


# --- unconditional noise-shape drops --------------------------------------

def test_drops_telemetry_prefixed_lines():
    assert not _kept("PageLoad.PaintTiming.navigationToFirstContentfulPaint some value")
    assert not _kept("OptimizationGuide.HintsFetcher.RequestStatus2 recorded here")


def test_drops_pure_hex_blobs():
    assert not _kept("a3f9c1d0e7b6485a2f19c0d3e7a8b4c1")


def test_drops_asset_and_schema_urls():
    assert not _kept("https://fonts.gstatic.com/s/inter/v12/some-font-file.woff2")
    assert not _kept('xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"')


def test_keeps_cdn_image_urls_with_ids():
    # deliberately NOT excluded - carries a user id / attachment filename
    assert _kept("https://cdn.discordapp.com/attachments/778/991/screenshot.png")


def test_drops_pki_certificate_noise():
    assert not _kept("Microsoft Windows Production PCA 2011")
    assert not _kept("DigiCert Global Root Certificate Authority")


def test_drops_code_signatures():
    assert not _kept("this program cannot be run in DOS mode")
    assert not _kept("cbuffer DriverConstants : register(b1)")


def test_drops_punctuation_heavy_code_lines():
    assert not _kept("if (x <= 3) { foo(); bar(); baz(); qux(); }")


# --- the real regression this filter was tuned to avoid --------------------

def test_appx_manifest_attribute_chains_are_not_treated_as_prose():
    # found while tuning: has 6+ words scattered through it, but never four
    # in a row - an earlier draft of the sentence check let this through
    s = ('<ActivatableClass ActivatableClassId="AccountsControlUI.Common.'
         'AccountsControlFrame" ThreadingModel="both" />')
    assert not _kept(s)


def test_boot_recovery_ui_text_is_dropped():
    assert not _kept("Press F9 to use a different operating system")
    assert not _kept("Preparing BitLocker recovery")  # only 3 words - no 4-in-a-row


# --- things that must still survive ----------------------------------------

def test_keeps_plain_chat_sentence():
    assert _kept("hey are you free to chat after school tomorrow")


def test_keeps_credential_shaped_lines():
    assert _kept("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.abc")
    assert _kept("password=hunter2andmore")


def test_keeps_evidentiary_json_fields():
    assert _kept('{"content":"see you at the usual spot tonight","channel":"12345"}')


def test_keeps_toast_notification_text():
    assert _kept("<text>hey beautiful, can i have your number?</text>")


def test_bare_at_no_longer_keeps_pure_noise():
    # measured against the real case's already-carved memory_string rows:
    # the old bare "@" clause was responsible for 72.4% of the entire
    # GENERAL/0.0-confidence bucket (815,978 of 1,126,582 rows), and of
    # those, exactly one was a genuine hit once a real email/mention shape
    # was required. These are real examples from that bucket, not invented.
    assert not _kept("MFE22@")
    assert not _kept("?vGetCoefficient@EXFORMOBJ@@QEBAXPEAUtagFLOATOBJ_XFORM@@@Z")
    assert not _kept("@%SystemRoot%\\system32\\drivers\\ztdns.sys,-1")
    assert not _kept("a `A`@")


def test_bare_at_false_positive_with_dot_local_part_is_rejected():
    # a naive email-shape regex permitting a bare "." as the whole local
    # part ("[\\w.+-]+@...") let this exact string (repeated 20x in the
    # real data) through as a false "email" - the local part must contain
    # an actual word character immediately before the "@".
    assert not _kept("5}.@e.kc")


def test_keeps_a_real_discord_mention():
    assert _kept("<@1473705902501531761>")


def test_default_cap_covers_a_real_full_ram_image():
    # real case's physmem.raw kept ~528k strings uncapped across its full
    # 5.83GiB (measured directly) - the cap must clear that with headroom,
    # not silently truncate a routine full-RAM capture.
    assert iter_strings.__defaults__[-1] >= 2_000_000
