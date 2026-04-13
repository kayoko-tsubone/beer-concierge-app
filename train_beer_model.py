import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. データの作成（ビール好きの経験則に基づいた相性表）
# 天気: 0:晴れ(暑い), 1:くもり, 2:雨
# 気分: 0:リフレッシュ, 1:リラックス, 2:ガッツリ
# おつまみ: 0:揚げ物, 1:ナッツ, 2:スイーツ
# ビールタイプ: 0:IPA, 1:HAZY IPA, 2:ペールエール, 3:ピルスナー, 4:スタウト

data = [
    # 晴れ(0)
    [0, 0, 0, 3], # 晴れ, リフレッシュ, 揚げ物 -> ピルスナー
    [0, 1, 1, 2], # 晴れ, リラックス, ナッツ -> ペールエール
    [0, 2, 0, 0], # 晴れ, ガッツリ, 揚げ物 -> IPA
    [0, 0, 1, 1], # 晴れ, リフレッシュ, ナッツ -> HAZY IPA
    # くもり(1)
    [1, 1, 1, 2], # くもり, リラックス, ナッツ -> ペールエール
    [1, 0, 0, 3], # くもり, リフレッシュ, 揚げ物 -> ピルスナー
    [1, 2, 0, 0], # くもり, ガッツリ, 揚げ物 -> IPA
    # 雨(2)
    [2, 1, 2, 4], # 雨, リラックス, スイーツ -> スタウト
    [2, 1, 1, 2], # 雨, リラックス, ナッツ -> ペールエール
    [2, 0, 1, 1], # 雨, リフレッシュ, ナッツ -> HAZY IPA
    # 追加パターン（学習を安定させるため）
    [0, 2, 1, 0], # 晴れ, ガッツリ, ナッツ -> IPA
    [2, 2, 0, 0], # 雨, ガッツリ, 揚げ物 -> IPA
    [1, 1, 2, 4], # くもり, リラックス, スイーツ -> スタウト
    [0, 0, 2, 1], # 晴れ, リフレッシュ, スイーツ -> HAZY IPA
]

columns = ['weather', 'mood', 'snack', 'beer_type']
df = pd.DataFrame(data, columns=columns)

# 2. データの分割
X = df.drop('beer_type', axis=1)
y = df['beer_type']

# 3. モデルの作成（決定木）
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

# 4. 精度の確認（データが少ないので簡易的）
print(f"学習データの精度: {model.score(X, y):.2f}")

# 5. モデルとラベルの保存
joblib.dump(model, 'beer_model.pkl')

# ラベル情報の保存
beer_labels = {
    0: 'IPA (インディア・ペールエール)',
    1: 'HAZY IPA (ヘイジーIPA)',
    2: 'ペールエール',
    3: 'ピルスナー',
    4: 'スタウト (黒ビール)'
}
joblib.dump(beer_labels, 'beer_labels.pkl')

print("ビール判定モデルとラベルを保存しました。")
