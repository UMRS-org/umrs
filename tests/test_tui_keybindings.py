from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

from umrs.tui.keybindings import KeyReader


class _FakeStdin:
    def __init__(self, chars: list[str]):
        self._chars = chars

    def fileno(self) -> int:
        return 0

    def read(self, count: int) -> str:
        del count
        return self._chars.pop(0) if self._chars else ""


def test_key_reader_reads_single_key_without_enter(monkeypatch) -> None:
    import umrs.tui.keybindings as keybindings

    state = {
        "attrs": [0, 0, 0, 0b111, 0, 0, [0, 0]],
    }
    calls: list[tuple[int, int, list[object]]] = []
    cbreak_calls: list[tuple[int, int]] = []

    fake_termios = SimpleNamespace(
        ECHO=0b001,
        ICANON=0b010,
        ISIG=0b100,
        VMIN=0,
        VTIME=1,
        TCSANOW=99,
    )

    def tcgetattr(fd: int) -> list[object]:
        assert fd == 0
        return copy.deepcopy(state["attrs"])

    def tcsetattr(fd: int, when: int, attrs: list[object]) -> None:
        assert fd == 0
        calls.append((fd, when, copy.deepcopy(attrs)))
        state["attrs"] = copy.deepcopy(attrs)

    def setcbreak(fd: int, when: int) -> None:
        assert fd == 0
        cbreak_calls.append((fd, when))
        attrs = copy.deepcopy(state["attrs"])
        attrs[3] &= ~(fake_termios.ECHO | fake_termios.ICANON)
        attrs[6][fake_termios.VMIN] = 1
        attrs[6][fake_termios.VTIME] = 0
        state["attrs"] = attrs

    fake_termios.tcgetattr = tcgetattr
    fake_termios.tcsetattr = tcsetattr
    fake_tty = SimpleNamespace(setcbreak=setcbreak)
    fake_stdin = _FakeStdin(["j"])
    keys: list[str] = []

    monkeypatch.setitem(sys.modules, "termios", fake_termios)
    monkeypatch.setitem(sys.modules, "tty", fake_tty)
    monkeypatch.setattr(keybindings.sys, "stdin", fake_stdin)
    monkeypatch.setattr(keybindings.select, "select", lambda readers, _writers, _errors, _timeout: (readers, [], []))

    reader: KeyReader

    def on_key(key: str) -> None:
        keys.append(key)
        reader.stop()

    reader = KeyReader(on_key)
    reader._run()

    assert keys == ["j"]
    assert cbreak_calls == [(0, fake_termios.TCSANOW)]
    assert calls[0][1] == fake_termios.TCSANOW
    assert calls[0][2][3] & fake_termios.ISIG == 0
    assert calls[-1][1] == fake_termios.TCSANOW
    assert calls[-1][2] == [0, 0, 0, 0b111, 0, 0, [0, 0]]