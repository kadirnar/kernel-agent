"""CuTe DSL toolchain (``kernel_agent/cute_dsl.py``, issue #133): the static ``doctor``
check and the compile cache across evaluations. CPU only: fake compile / export / load
functions; one test imports the installed DSL (no CUDA context, no launch)."""

import importlib.util
import json

import pytest

from kernel_agent import cute_dsl


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cute-cache"
    monkeypatch.setenv(cute_dsl.CACHE_ENV, str(root))
    monkeypatch.setattr(cute_dsl, "STATS", cute_dsl.Stats())
    return root


class Fake:
    """Stand-ins for cute.compile / export_to_c / load_module."""

    def __init__(self):
        self.compiled, self.exported, self.loaded = [], [], []

    def compile(self, fn, *args, options=""):
        self.compiled.append((fn, args, options))
        return ("compiled", args)

    def export(self, compiled, directory, prefix, tvm_ffi):
        self.exported.append((prefix, tvm_ffi))
        (directory / cute_dsl.OBJECT).write_bytes(b"\x7fELF" + prefix.encode())

    def load(self, obj, prefix, tvm_ffi):
        assert obj.read_bytes() == b"\x7fELF" + prefix.encode()
        self.loaded.append((prefix, tvm_ffi))
        return ("loaded", prefix)


def kernel(a):  # the "source" whose file digest keys the cache
    return a


def _call(fake, *args, key=("k", 1024), **kw):
    kw = {"options": "--enable-tvm-ffi", "arch": "sm_120a", **kw}
    return cute_dsl.compile_cached(
        kernel, *args, key=key, compile=fake.compile, export=fake.export, load=fake.load, **kw
    )


def test_compile_once_then_load_from_disk(cache):
    fake = Fake()
    assert _call(fake, 1, 2)[0] == "compiled"
    assert len(fake.compiled) == 1 and fake.exported[0][1] is True  # TVM-FFI export
    entries = cute_dsl.cache_entries()
    assert len(entries) == 1 and entries[0]["arch"] == "sm_120a"
    assert entries[0]["name"] == "kernel" and entries[0]["options"] == "--enable-tvm-ffi"
    assert isinstance(entries[0]["compile_s"], float)
    # a later evaluation (another process: the same key) loads the object file
    again = Fake()
    assert _call(again, 1, 2)[0] == "loaded" and again.compiled == []
    assert cute_dsl.STATS.hits == 1 and cute_dsl.STATS.misses == 1
    summary = cute_dsl.cache_summary()
    assert summary["entries"] == 1 and summary["median_compile_s"] is not None


def test_key_covers_specialisation_arch_version_options_and_source(cache, monkeypatch):
    fake = Fake()
    _call(fake)
    _call(fake, key=("k", 2048))  # another static shape
    _call(fake, arch="sm_121a")  # another architecture
    _call(fake, options="")  # no TVM-FFI: another calling convention
    _call(fake, source="edited")  # the kernel file changed
    monkeypatch.setattr(cute_dsl, "dsl_version", lambda: "9.9.9")  # a DSL upgrade
    _call(fake)
    assert len(fake.compiled) == 6 and len(cute_dsl.cache_entries()) == 6
    assert fake.exported[3][1] is False


def test_failures_fall_back_to_the_compiled_function(cache):
    fake = Fake()

    def broken_export(*args):
        raise RuntimeError("export not supported")

    out = cute_dsl.compile_cached(
        kernel, key=1, arch="sm_120a", compile=fake.compile, export=broken_export, load=fake.load
    )
    assert out[0] == "compiled" and cute_dsl.cache_entries() == []
    assert "export not supported" in cute_dsl.STATS.errors[-1]
    assert not any(p.name.startswith(".") or ".tmp" in p.name for p in cache.rglob("*"))
    # a corrupt entry is removed and compiled again
    _call(fake)
    assert len(cute_dsl.cache_entries()) == 1

    def bad_load(*args):
        raise OSError("truncated object")

    out = cute_dsl.compile_cached(
        kernel,
        key=("k", 1024),
        options="--enable-tvm-ffi",
        arch="sm_120a",
        compile=fake.compile,
        export=fake.export,
        load=bad_load,
    )
    assert out[0] == "compiled" and "truncated object" in cute_dsl.STATS.errors[-1]
    assert len(cute_dsl.cache_entries()) == 1  # stored again


def test_cache_off_and_clear(cache, monkeypatch):
    fake = Fake()
    _call(fake)
    assert cute_dsl.clear_cache() == 1 and cute_dsl.cache_entries() == []
    monkeypatch.setenv(cute_dsl.CACHE_ENV, "off")
    assert cute_dsl.cache_root() is None
    _call(fake)
    _call(fake)
    assert len(fake.compiled) == 3 and fake.loaded == []


def test_source_digest_follows_the_defining_file(tmp_path):
    assert cute_dsl.source_digest(kernel) == cute_dsl.source_digest(test_cache_off_and_clear)
    assert cute_dsl.source_digest(Fake()) == cute_dsl.source_digest(kernel)  # a callable object
    assert cute_dsl.source_digest(json.dumps) != cute_dsl.source_digest(kernel)


def test_arch_name(monkeypatch):
    monkeypatch.delenv(cute_dsl.ARCH_ENV, raising=False)
    assert cute_dsl.arch_name((12, 0)) == "sm_120a"
    assert cute_dsl.arch_name((8, 9)) == "sm_89"
    assert cute_dsl.arch_name(None) is None
    monkeypatch.setenv(cute_dsl.ARCH_ENV, "sm_120f")
    assert cute_dsl.arch_name((12, 0)) == "sm_120f"


def _probe(mxf8=True):
    return lambda: {
        "version": "4.8.0",
        "arches": ["sm_89", "sm_90a", "sm_100a", "sm_120", "sm_120a", "sm_121a"],
        "mxf8": "MmaMXF8Op" if mxf8 else None,
        "block_scaled_arches": ["sm_120a", "sm_120f", "sm_121a", "sm_121f"] if mxf8 else [],
    }


def _dists():
    return {
        "nvidia-cutlass-dsl": "4.8.0",
        "nvidia-cutlass-dsl-libs-base": "4.8.0",
        "nvidia-cutlass-dsl-libs-cu12": "4.8.0",
    }


def test_check_sm120(cache, monkeypatch):
    monkeypatch.delenv(cute_dsl.ARCH_ENV, raising=False)
    status = cute_dsl.check((12, 0), probe=_probe(), installed=True, distributions=_dists)
    assert status.ok and status.arch == "sm_120a" and status.block_scaled
    assert status.libs == ["cu12"] and status.version == "4.8.0"
    text = "\n".join(status.describe())
    assert "CuTe DSL 4.8.0" in text and "sm_120a supported" in text and "QMMA.SF" in text
    assert "compile cache:" in text
    # an older DSL without the block-scaled atom: say what it costs
    old = cute_dsl.check((12, 0), probe=_probe(False), installed=True, distributions=_dists)
    assert old.ok and old.block_scaled is False
    assert any("half rate" in n for n in old.notes)
    # another GPU: supported, no block-scaled MMA, no note about it
    ada = cute_dsl.check((8, 9), probe=_probe(), installed=True, distributions=_dists)
    assert ada.arch == "sm_89" and ada.block_scaled is False
    assert not any("half rate" in n for n in ada.notes)


def test_check_failures(cache, monkeypatch):
    monkeypatch.setenv(cute_dsl.ARCH_ENV, "sm_100a")
    missing = cute_dsl.check((12, 0), installed=False)
    assert not missing.ok and "not installed" in missing.describe()[0]

    def broken():
        raise ImportError("libcute_dsl_runtime.so: cannot open shared object file")

    bad = cute_dsl.check((12, 0), probe=broken, installed=True, distributions=_dists)
    assert not bad.ok and "import FAILED" in bad.describe()[0]
    wrong = cute_dsl.check((12, 0), probe=_probe(), installed=True, distributions=_dists)
    assert any("CUTE_DSL_ARCH=sm_100a differs from the GPU (sm_120)" in n for n in wrong.notes)
    nogpu = cute_dsl.check(None, probe=_probe(), installed=True, distributions=_dists)
    assert "architecture not checked" in "\n".join(nogpu.describe()) or nogpu.arch == "sm_100a"


@pytest.mark.skipif(importlib.util.find_spec("cutlass") is None, reason="no CuTe DSL")
def test_check_with_the_installed_dsl(cache, monkeypatch):
    """The real probe: imports only (``import cutlass.cute``), no CUDA context."""
    monkeypatch.delenv(cute_dsl.ARCH_ENV, raising=False)
    status = cute_dsl.check((12, 0))
    assert status.import_error is None, status.import_error
    assert status.arch == "sm_120a" and status.arch_known
    assert status.block_scaled, "MmaMXF8Op does not admit sm_120a"
