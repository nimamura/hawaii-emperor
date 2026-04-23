# CLAUDE.md — hawaii-emperor

Inherit global TDD discipline from `~/.claude/CLAUDE.md`:

- テストを先に書き、red を確認してからコミット
- 実装中はテストを変更せず、コードを修正し続ける
- 小さな well-named commits
- `data/**` と `outputs/**` は gitignore。最終 PNG のみ `git add -f` で履歴に残す

## シリーズの視覚言語

過去 2 作 (`planetary-hypsometry`, `fossil-slabs`) と style を合わせる。
2×2 グリッド + A/B/C/D ラベル、白背景、publication-grade matplotlib。

## 依存禁止

`/Users/nimamura/work_raithing/{planetary-hypsometry,fossil-slabs}/` のコードを
import/copy しない。スタイル参照のみ read-only で OK。
