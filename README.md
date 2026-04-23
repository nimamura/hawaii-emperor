# hawaii-emperor

Hawaii–Emperor 海山列の年代分布から、**太平洋プレートが 47 Ma に
NNW 進行から WNW 進行へ 60° 方向転換した** 瞬間を 1 枚で示す。

プレートの速度はおおむね変わらず ~8 cm/yr のまま。変わったのは向きだけ。
その化石が海底に 60 個以上の火山として直列に刻まれている。

![Hawaii–Emperor Bend](outputs/hawaii_emperor_bend.png)

## X 投稿文案

> 太平洋の海底に一列に並ぶ 60 個以上の火山 — ハワイ・皇帝海山列。
> 年代をプロットすると 47 Ma でチェーンがきれいに折れ曲がる。
> プレートの速度はほぼ変わらず、進路だけが 60° 変わった瞬間の化石。

## 各パネル

- **A**: 北太平洋の海山分布。年代でカラーエンコード。赤星が Daikakuji
  Seamount — 47 Ma bend の目印
- **B**: Kilauea からの大円距離 vs 年代。Hawaiian 側 (青) と Emperor 側
  (赤) をそれぞれ単独 OLS でフィット、47 Ma に垂直線
- **C**: 鎖状火山列の方位角 vs 年代。47 Ma を境に ~300° (WNW) から
  ~350° (ほぼ N) へのステップが明瞭
- **D**: 見かけのプレート速度。~8 cm/yr 付近で変動するが、bend を挟んで
  系統的には変わらない

## データソース

- O'Connor, J. M. et al. (2013) *Geochem. Geophys. Geosyst.* 14, 4564.
  Hawaiian–Emperor Ar-Ar 年代 compilation
- Sharp, W. D. & Clague, D. A. (2006) *Science* 313, 1281.
  Emperor seamount redate, bend age ~50 Ma 議論
- Clague, D. A. & Dalrymple, G. B. (1987/1989) USGS PP-1350.
  古典的 compilation

距離原点は Kilauea (19.42°N, -155.29°E = 204.71°E)。Emperor 海山列の
距離は Daikakuji (32.08°N, 172.30°E) を経由する along-chain 距離。

## 使い方

```bash
python -m pip install -e .[dev]
pytest
PYTHONPATH=. python scripts/generate_hawaii_emperor_figure.py
```

出力: `outputs/hawaii_emperor_bend.png`

## シリーズ

- 1 作目: [planetary-hypsometry](https://github.com/naoto-imamura) —
  惑星の hypsogram
- 2 作目: [fossil-slabs](https://github.com/naoto-imamura) —
  マントル遷移帯に凍結した白亜紀太平洋スラブ
- 3 作目: このリポジトリ — 太平洋プレートの 47 Ma 方向転換
