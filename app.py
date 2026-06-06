import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tempfile
import os
import urllib.parse

st.set_page_config(
    page_title="【精密解析】そらる・シンクロ率チェッカー",
    layout="centered"
)

# ====== データ読み込み ======
@st.cache_data
def load_soraru_data():
    return pd.read_csv("soraru_data.csv")

df = load_soraru_data()

# ====== 音声特徴量抽出 ======
def extract_features(uploaded_file, duration=15):
    suffix = ".mp3" if uploaded_file.type == "audio/mpeg" else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, duration=duration, mono=True)
        if len(y) == 0:
            raise ValueError("音声データが空です。")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    finally:
        os.remove(tmp_path)

# ====== 距離 → スコア変換 ======
def convert_to_score(dist, min_dist, max_dist):
    if max_dist == min_dist:
        return 50.0
    return float(max(5.0, min((1 - (dist - min_dist) / (max_dist - min_dist)) * 100, 100.0)))

# ====== 解析メイン ======
@st.cache_data
def precompute_stats(csv_path="soraru_data.csv"):
    df = pd.read_csv(csv_path)
    feats = df[[f"mfcc_{i}" for i in range(26)]].values.astype(np.float32)
    diff = feats[:, np.newaxis, :] - feats[np.newaxis, :, :]
    pair_dists = np.sqrt((diff ** 2).sum(axis=-1))
    upper = pair_dists[np.triu_indices(len(feats), k=1)]
    return feats, float(upper.min()), float(upper.max()), feats.mean(axis=0)

song_feats, dist_min, dist_max, soraru_center = precompute_stats()

def analyze(user_feat):
    dist_total = float(np.linalg.norm(user_feat - soraru_center))
    soraru_rate = convert_to_score(dist_total, dist_min, dist_max)

    diffs = song_feats - user_feat[np.newaxis, :]
    dists = np.sqrt((diffs ** 2).sum(axis=1))

    results = []
    for dist, (_, row) in zip(dists, df.iterrows()):
        results.append({
            "song": row["song"],
            "url": row["youtube_url"],
            "score": round(convert_to_score(float(dist), dist_min, dist_max), 1),
        })

    df_res = pd.DataFrame(results).sort_values("score", ascending=False)
    return round(soraru_rate, 1), df_res

# ====== コメント生成 ======
def generate_comment(rate: float) -> str:
    if rate >= 95:
        return ("【神域の同調】もはや判別不能、本人降臨レベルです。声の立ち上がりから消え際のスーッとした減衰まで、"
                "そらるさんの波形をそのままなぞったかのような一致を見せています。空気に溶けるような透明感と、"
                "耳元で囁かれているような実在感を同時に持っており、聴く人を一瞬で『碧の世界』へ引きずり込む魔力を持っています。")
    elif rate >= 90:
        return ("【転生クラス】驚異的なシンクロ率です。そらるさん特有の『温かみのある無機質さ』を見事に再現しています。"
                "特に高音域へ抜ける際の、切なさを孕んだ息の混ぜ方は天性のものと言えるでしょう。マイクを通した瞬間に完成されるその響きは、"
                "もはや転生したそらるさん本人と言っても過言ではありません。自信を持って『そらるボイス』を名乗ってください。")
    elif rate >= 85:
        return ("【至高の共鳴】『ビー玉の中の宇宙』をそのまま体現したような、澄んだ響きを持っています。中音域の安定感と、"
                "そこに乗る繊細なウィスパー成分の比率が黄金比に近い状態です。ふとした瞬間のニュアンスが驚くほど本人に似ているため、"
                "初見のリスナーは間違いなく耳を疑うはず。ミックスでリバーブを深めに掛ければ、完璧に化けるポテンシャルがあります。")
    elif rate >= 80:
        return ("【極めて高い親和性】かなりの高シンクロ率です。中音域の厚みと、吐息が混ざり合う『エモーショナルな質感』が非常に近く、"
                "そらるさんの楽曲、特にバラードでの表現力が爆発的に高まるタイプです。声の抜け感が非常にスムーズで、"
                "聴き手にストレスを与えない癒やしの成分がたっぷり詰まっています。あと一歩で、神域に手が届く位置にいます。")
    elif rate >= 75:
        return ("【ハイレベルな同調】かなり似ています。特にフレーズ終わりの『息の抜き方』や、言葉の頭に置くエッジボイスの使い方が、"
                "そらるさんの歌唱スタイルと深く共鳴しています。歌い方や声の表情に確かな『そらる味』があり、ファンなら思わずニヤリとしてしまうはず。"
                "意識して低音の響きを深めるだけで、さらにシンクロ率は跳ね上がる可能性を秘めています。")
    elif rate >= 70:
        return ("【確かなそらる成分】声の抜け感や、鼻に抜ける甘い響きにそらるさんのエッセンスを強く感じます。全体的に落ち着いたトーンでありながら、"
                "サビなどで見せる芯の強さがそらるさんの歌唱設計に非常に似ています。今のままでも十分『似ている』と言われるレベルですが、"
                "もう少しだけ『脱力感』を意識して歌うと、より本人に近いアンニュイな魅力が増すでしょう。")
    elif rate >= 65:
        return ("【潜在的シンクロ】ところどころに強い『そらる成分』を検知しました。全ての帯域ではありませんが、"
                "特定の音域（特に中低音）において、ハッとするほど似た響きを見せることがあります。自分の個性をベースにしつつ、"
                "そらるさんのエッセンスを絶妙なスパイスとして持っている状態です。寄せる技術を磨けば、まだまだ上を狙える伸び代を感じます。")
    elif rate >= 60:
        return ("【ハイブリッド・ボイス】あなた自身の個性を主軸にしつつ、そらるさんのような『静寂を纏った響き』を一部に持っています。"
                "全ての曲というよりは、特定の楽曲（例えば『銀の祈誓』のようなシリアスな曲）で特に高いシンクロ率を発揮するタイプです。"
                "自分の声を活かしながら、要所でそらるさんのテクニックを取り入れるのが一番輝くスタイルと言えます。")
    elif rate >= 55:
        return ("【共鳴の予感】声の密度や帯域のバランスに、そらるさんと共通するパーツを確認しました。現在はあなた独自の歌い方が強く出ていますが、"
                "声質そのものには『透明感』の素質が十分にあります。ウィスパーボイスの練習を重ねることで、"
                "あなたの喉の中に眠っている『そらる成分』をもっと引き出すことができるはず。可能性に満ちた数値です。")
    elif rate >= 50:
        return ("【唯一無二の響き】自分自身の個性が半分を占めている、非常に魅力的なブレンド具合です。そらるさんのような落ち着きを持ちつつも、"
                "あなたにしか出せない独自の色彩を纏った歌声です。無理に寄せるよりも、今の響きにそらるさんの楽曲の世界観を乗せることで、"
                "全く新しい『そらるソング』を生み出せる才能を秘めています。その個性を大切にしてください。")
    elif rate >= 40:
        return ("【独創のアーティスト・ボイス】そらるさんの成分をベースにしつつも、あなた独自の力強い響きがはっきりと顔を出しています。"
                "今のままでも『そらるさんの曲を自分流に歌いこなせる』、模倣を超えた自立したバランスの声質です。"
                "シンクロ率という枠に収まらない、歌い手としての強いアイデンティティを感じさせる結果となりました。")
    elif rate >= 30:
        return ("【ニュー・ジェネレーション】そらるさんの声質とは異なるベクトルで、非常にキャラクターの立った歌声です。"
                "シンクロ率は低めですが、それはあなたの声にしっかりとした『芯』があり、誰の影響も受けていない証拠です。"
                "そらるさんの楽曲をカバーしても、原曲の影に隠れない圧倒的な存在感を放つことができる、自立したボイスタイプと言えます。")
    elif rate >= 20:
        return ("【アンリミテッド・カラー】そらるさんの『碧』の世界とは対極にあるような、エネルギッシュまたは独自の色彩を持った声です。"
                "波形解析の結果、そらるさんの成分とは別の帯域で非常に高いエネルギーを検知しました。これは模倣ではなく創造に向いた声であり、"
                "あなたにしか出せない響きを武器に、新しい音楽の道を切り拓くべき喉の持ち主です。")
    elif rate >= 10:
        return ("【アイデンティティの確立】あなたの声が誰にも似ていない『完全オリジナル』であることを示しています。"
                "そらるさんのファンでありながら、自分自身の個性をこれほど純粋に保てているのは一つの才能です。"
                "そらるさんの楽曲をあなたが歌うことで、原曲とは全く違う、あなたにしか救えないリスナーに届く新しい命が吹き込まれるでしょう。")
    else:
        return ("【究極のオリジナリティ】測定不能！そらるさんの成分をほぼ検知できないほど、あなたの個性は突き抜けています。"
                "ある意味、このサイトで最も貴重な『0%に近い希少な響き』の持ち主です。既存の枠組みにはまらないその声を大切に、"
                "自分だけの道を突き進んでください。その響きは、誰にも真似できないあなただけの宝物です。")

# ====== URLパラメータ（共有モード） ======
params = st.query_params
shared_rate = None
if "rate" in params:
    try:
        shared_rate = float(params["rate"])
    except Exception:
        pass

if shared_rate is not None:
    st.subheader("🔁 共有された診断結果")
    st.markdown(f"""
    <div class="result-box">
        <h2>そらる・シンクロ率： {shared_rate:.1f}%</h2>
        <p>{generate_comment(shared_rate)}</p>
    </div>
    """, unsafe_allow_html=True)
    if "song1" in params:
        st.markdown("### あなたに近い そらる楽曲 TOP5")
        for i in range(1, 6):
            if f"song{i}" in params:
                st.markdown(f"**第{i}位：{params[f'song{i}']}**")
    st.stop()

# ====== CSS ======
st.markdown("""
<style>
body { background-color: #f4f8ff; font-family: 'Hiragino Maru Gothic ProN', 'Yu Gothic', sans-serif; }
.title-card {
    background: linear-gradient(135deg, #e9f2ff, #d7e6ff);
    padding: 35px 20px; border-radius: 18px;
    border: 2px solid #8fb4ff; margin-bottom: 25px;
    box-shadow: 0 4px 12px rgba(120,150,220,0.25);
}
.title-text { color: #0f1a33; font-weight: 800; font-size: 2.3rem; line-height: 1.3; text-align: center; }
.subtitle-text { color: #2a4d8f; font-size: 1.1rem; text-align: center; margin-top: 8px; }
.result-box {
    background: #f9fbff; padding: 22px;
    border-left: 6px solid #5fa8ff; border-radius: 10px;
    margin: 20px 0; border: 1px solid #bcd4ff;
    box-shadow: 0 3px 10px rgba(150,180,255,0.25);
}
.result-box h2, .result-box p { color: #0f1a33; }
.song-card {
    background: #ffffff; border: 2px solid #bcd4ff;
    padding: 18px; border-radius: 12px; margin-bottom: 14px;
    box-shadow: 0 3px 10px rgba(180,200,255,0.25);
}
.song-card h3, .song-card h4, .song-card a { color: #0f1a33; }
</style>
""", unsafe_allow_html=True)

# ====== タイトル ======
st.markdown("""
<div class="title-card">
    <div class="title-text">【精密解析】<br>そらる・シンクロ率チェッカー</div>
    <div class="subtitle-text">あなたの声に最も近い楽曲も判定！</div>
</div>
""", unsafe_allow_html=True)

# ====== アップロード ======
st.subheader("① 音声ファイルをアップロード")
st.write("**対応形式：** wav / mp3")
st.write("**推奨：** 10〜20秒のサビや盛り上がり部分（声が大きいところ）")
st.write("※ 声だけ・アカペラだとより精度が上がります")
uploaded_file = st.file_uploader(
    "ここに音声ファイルをドラッグ＆ドロップしてください",
    type=["wav", "mp3"]
)

st.markdown("---")

# ====== 解析 ======
st.subheader("② 精密解析")
if st.button("🔍 精密解析スタート"):
    if uploaded_file is None:
        st.warning("先に音声ファイルをアップロードしてください。")
    else:
        with st.spinner("解析中…"):
            try:
                user_feat = extract_features(uploaded_file)
                soraru_rate, result = analyze(user_feat)
            except Exception as e:
                st.error(f"解析中にエラーが発生しました：{e}")
                st.stop()

        st.success("解析が完了しました！")
        st.subheader("③ 結果")
        st.markdown(f"""
        <div class="result-box">
            <h2>あなたのそらる・シンクロ率： {soraru_rate:.1f}%</h2>
            <p>{generate_comment(soraru_rate)}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("④ あなたに近い そらる楽曲 TOP5")
        top5 = result.head(5).reset_index(drop=True)

        top1 = top5.iloc[0]
        st.markdown(f"""
        <div class="song-card">
            <h3>🥇 第1位：{top1['song']}（{top1['score']:.1f}%）</h3>
        </div>
        """, unsafe_allow_html=True)
        st.video(top1["url"])
        st.write(f"[YouTubeで開く]({top1['url']})")

        for i in range(1, len(top5)):
            row = top5.iloc[i]
            st.markdown(f"""
            <div class="song-card">
                <h4>第{i+1}位：{row['song']}（{row['score']:.1f}%）</h4>
                <a href="{row['url']}" target="_blank">YouTubeで開く</a>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ====== X共有 ======
        base_url = "https://huggingface.co/spaces/あなたのユーザー名/soraru-sync-checker"
        share_params = {
            "rate": f"{soraru_rate:.1f}",
            "song1": top5.iloc[0]["song"],
            "song2": top5.iloc[1]["song"],
            "song3": top5.iloc[2]["song"],
            "song4": top5.iloc[3]["song"],
            "song5": top5.iloc[4]["song"],
        }
        share_url = f"{base_url}?{urllib.parse.urlencode(share_params)}"
        tweet_url = f"https://twitter.com/intent/tweet?text=そらる・シンクロ率診断！&url={share_url}"
        st.markdown(f"[🔗 Xで結果をシェアする]({tweet_url})")
else:
    st.info("音声ファイルをアップロードしてから「精密解析スタート」を押してください。")
