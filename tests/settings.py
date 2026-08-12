"""Test settings for the face_embed suite.

Two things have to happen before Django touches the URLconf: the fail-closed
settings need placeholder values, and the inference stack needs stubbing —
importing detection.views otherwise pulls in torch, transformers and a model
download. Both happen here because this module is loaded first.
"""

import os

from tests import stubs

stubs.install()

_PLACEHOLDERS = {
    "FACE_EMBED_SECRET_KEY": "test-only-not-a-real-secret-key",
    "FACE_EMBED_DEBUG": "False",
    "FACE_EMBED_ALLOWED_HOSTS": "localhost,127.0.0.1,testserver",
}

for _name, _value in _PLACEHOLDERS.items():
    os.environ.setdefault(_name, _value)

from face_embed.settings import *  # noqa: F401,F403,E402

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
