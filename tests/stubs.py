"""Stand-ins for the inference stack, installed before Django loads the URLconf.

detection/views.py imports detect_face and extract_face_embedding from
detection/face_functions.py, which imports torch, torchvision, transformers,
huggingface_hub and PIL at module scope — and downloads model weights from
Hugging Face the first time the model is built.

None of that is needed to test the HTTP contract. What this API promises is:
an image goes in, a JSON embedding and its dimension come out, and each failure
mode maps to a specific status code. Those promises live entirely in views.py.

So the heavy modules are replaced with the smallest objects that let the import
succeed. The two functions that actually matter are left importable and are
monkeypatched per test, which keeps the seam visible: a test that forgets to
patch them gets a NotImplementedError rather than a silent pass.

numpy and the real request/response path are NOT stubbed — those are cheap and
are part of what is being tested.
"""

import sys
import types


def _module(name, **attrs):
    """A stub module that yields an _Inert for anything not named explicitly.

    Enumerating every attribute these libraries expose is a losing game —
    face_functions.py reaches for cv2.CascadeClassifier, cv2.data.haarcascades
    and more, and the list grows whenever the model code changes. PEP 562 module
    __getattr__ lets the fallback be automatic, so the stub does not need
    maintaining every time the inference code does.
    """
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    mod.__getattr__ = lambda item, _n=name: _Inert(f"{_n}.{item}")
    return mod


class _Inert:
    """Absorbs whatever the module under import does to it.

    face_functions.py does real work at module scope — it builds a
    transforms.Compose pipeline, for instance — so a stub that raises on
    attribute access fails during import rather than at the point of use.
    Every attribute is another _Inert and every call returns one, which is
    enough for any chain of construction to complete.

    Nothing here is ever exercised by a passing test: the two functions views.py
    actually calls are monkeypatched, so a test that forgets to patch them gets
    an _Inert back and fails on the assertion instead of silently passing.
    """

    def __init__(self, name):
        self._name = name

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return _Inert(f"{self._name}.{item}")

    def __call__(self, *args, **kwargs):
        return _Inert(f"{self._name}()")

    def __repr__(self):
        return f"<stub {self._name}>"


def install():
    """Put the stubs in sys.modules. Safe to call more than once."""
    if "torch" in sys.modules and getattr(sys.modules["torch"], "_phasicon_stub", False):
        return

    torch = _module(
        "torch",
        _phasicon_stub=True,
        cuda=_module("torch.cuda", is_available=lambda: False),
        no_grad=lambda: _NullContext(),
        device=lambda *a, **k: "cpu",
    )
    torch.nn = _module("torch.nn", functional=_module("torch.nn.functional"))
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional
    sys.modules["torch.cuda"] = torch.cuda

    sys.modules["torchvision"] = _module("torchvision", transforms=_Inert("torchvision.transforms"))
    sys.modules["torchvision.transforms"] = _module("torchvision.transforms")

    sys.modules["PIL"] = _module("PIL", Image=_Inert("PIL.Image"))
    sys.modules["PIL.Image"] = _module("PIL.Image")

    sys.modules["transformers"] = _module("transformers", AutoModel=_Inert("AutoModel"))
    sys.modules["huggingface_hub"] = _module(
        "huggingface_hub", hf_hub_download=_Inert("hf_hub_download")
    )

    # cv2 needs real-ish behaviour: views.py calls imdecode and branches on a
    # None result to produce its 400. The default here decodes successfully;
    # tests that want the invalid-image path patch detection.views.cv2.imdecode.
    import numpy as np

    sys.modules["cv2"] = _module(
        "cv2",
        IMREAD_COLOR=1,
        IMREAD_GRAYSCALE=0,
        # face_functions.py builds a cascade path with string concatenation:
        #   cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_...xml')
        # so this one has to be a real string rather than an _Inert.
        data=_module("cv2.data", haarcascades="/stub/haarcascades/"),
        imdecode=lambda buf, flags: (
            None if len(buf) == 0 else np.zeros((64, 64, 3), dtype=np.uint8)
        ),
        cvtColor=lambda img, code: img,
        resize=lambda img, size, **k: np.zeros((size[1], size[0], 3), dtype=np.uint8),
        COLOR_BGR2RGB=4,
    )


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
