"""Streamlit authentication tests."""

from __future__ import annotations

from app.streamlit.auth import hash_password, verify_password


def test_hash_password_is_deterministic() -> None:
    assert hash_password("secret") == hash_password("secret")
    assert hash_password("secret") != hash_password("other")


def test_verify_password_accepts_matching_hash() -> None:
    password_hash = hash_password("secret")
    assert verify_password("secret", password_hash)


def test_verify_password_rejects_wrong_password() -> None:
    password_hash = hash_password("secret")
    assert not verify_password("wrong", password_hash)
