import streamlit as st
import joblib

# ページ設定
st.set_page_config(page_title="🍺 AIビア・コンシェルジュ", layout="wide")

# タイトル
st.title("🍺 AIビア・コンシェルジュ")
st.write("今の気分と天気を教えてください。AIが最高の一杯をおすすめします！")

# モデルとラベルの読み込み
try:
    model = joblib.load('beer_model.pkl')
    beer_labels = joblib.load('beer_labels.pkl')
except FileNotFoundError:
    st.error("モデルファイルが見つかりません。先に学習を行ってください。")
    st.stop()

# サイドバーでユーザー入力
st.sidebar.header("☀️ 今の状況を教えてください")

# 天気の選択
weather_options = {
    "☀️ 晴れ（暑い！）": 0,
    "☁️ くもり": 1,
    "☔️ 雨": 2
}
weather_selected = st.sidebar.radio("**今の天気は？**", list(weather_options.keys()))
weather = weather_options[weather_selected]

# 気分の選択
mood_options = {
    "🚀 リフレッシュしたい（スッキリ飲みたい！）": 0,
    "💆 リラックスしたい（ゆったり飲みたい）": 1,
    "🔥 ガッツリ飲みたい（ガツンと来い！）": 2
}
mood_selected = st.sidebar.radio("**今の気分は？**", list(mood_options.keys()))
mood = mood_options[mood_selected]

# おつまみの選択
snack_options = {
    "🍗 揚げ物（唐揚げ、フライドポテトなど）": 0,
    "🥜 ナッツ・チーズ": 1,
    "🍰 スイーツ（意外と合う！）": 2
}
snack_selected = st.sidebar.radio("**何と一緒に飲みますか？**", list(snack_options.keys()))
snack = snack_options[snack_selected]

# 判定ボタン
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🎯 最高の一杯を探す！", use_container_width=True):
        # 予測
        input_data = [[weather, mood, snack]]
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # 結果表示
        st.subheader("✨ AIが選んだ最高の一杯！")
        beer_name = beer_labels[prediction]
        st.success(f"# 🍺 {beer_name}")
        
        # ビールの説明
        beer_descriptions = {
            0: "**IPA（インディア・ペールエール）**\n\n苦味と香りがガツンとくる、飲みごたえ抜群の一杯です。ホップの香りが心地よく、揚げ物との相性も最高！ガッツリ飲みたい時の王様です。",
            1: "**HAZY IPA（ヘイジーIPA）**\n\n濁りがあってフルーティー、ジュースのような飲みやすさが特徴。香りは豊かですが、IPAより飲みやすいので、クラフトビール初心者にもおすすめです。",
            2: "**ペールエール**\n\n香り豊かでバランスの良い、クラフトビールの王道。苦すぎず、香りも心地よく、どんなシーンでも活躍する万能選手です。",
            3: "**ピルスナー**\n\n喉越し最高！キンキンに冷やして飲みたい定番ビール。スッキリとした飲み口で、暑い日のリフレッシュに最適です。",
            4: "**スタウト（黒ビール）**\n\n濃厚でコクがある、深い味わいが特徴。スイーツとの相性が抜群で、チョコレートやナッツとの組み合わせは最高です。"
        }
        
        st.write(beer_descriptions[prediction])
        
        # 確信度の表示
        st.write("### 📊 判定の確信度")
        confidence = prediction_proba[prediction] * 100
        st.progress(confidence / 100)
        st.write(f"**確信度: {confidence:.1f}%**")


