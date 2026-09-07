"""Ed25519 signing for evidence manifests.

The collector generates (or is given) a keypair. The manifest is signed at
acquisition time; the lab app verifies the signature on import. A valid
signature plus matching file hashes is what lets an investigator assert the
evidence has not changed since collection.

Key material is stored as PEM. The private key may be passphrase-protected.
"""

from __future__ import annotations

from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


class SignatureError(Exception):
    pass


def generate_keypair() -> tuple[bytes, bytes]:
    """Return (private_pem, public_pem). Private key is unencrypted PEM."""
    priv = Ed25519PrivateKey.generate()
    private_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def load_private_key(pem: bytes, password: bytes | None = None) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(pem, password=password)
    if not isinstance(key, Ed25519PrivateKey):
        raise SignatureError("not an Ed25519 private key")
    return key


def load_public_key(pem: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, Ed25519PublicKey):
        raise SignatureError("not an Ed25519 public key")
    return key


def sign(private_pem: bytes, data: bytes, password: bytes | None = None) -> bytes:
    return load_private_key(private_pem, password).sign(data)


def verify(public_pem: bytes, data: bytes, signature: bytes) -> bool:
    try:
        load_public_key(public_pem).verify(signature, data)
        return True
    except InvalidSignature:
        return False


def public_key_fingerprint(public_pem: bytes) -> str:
    """Short hex fingerprint of the raw public key, for display."""
    from dd_core.hashing import sha256_bytes

    raw = load_public_key(public_pem).public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return sha256_bytes(raw)[:16]


def write_keypair(directory: str | Path, private_pem: bytes, public_pem: bytes) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    priv_path = directory / "collector_key.pem"
    pub_path = directory / "collector_key.pub.pem"
    priv_path.write_bytes(private_pem)
    priv_path.chmod(0o600)
    pub_path.write_bytes(public_pem)
    return priv_path, pub_path
