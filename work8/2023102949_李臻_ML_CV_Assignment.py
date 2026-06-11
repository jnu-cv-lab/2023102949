from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

# 任务1：加载数据
digits = load_digits()
X = digits.data
y = digits.target
images = digits.images

print("===== 任务1：数据信息 =====")
print("图像总数：", len(images))
print("图像大小：", images[0].shape)
print("标签：", np.unique(y))

# 画图并保存
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(y[i])
    plt.axis('off')
plt.savefig("digits_samples.png")  # 保存图片
plt.close()

# 任务2：划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("\n===== 任务2：数据划分 =====")
print("训练集：", X_train.shape)
print("测试集：", X_test.shape)

# 任务3：特征表示
print("\n===== 任务3：特征表示 =====")
print("8x8图像展平为64维向量")

# 任务4：训练模型
models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
y_preds = {}

print("\n===== 任务4：模型准确率 =====")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    y_preds[name] = y_pred
    print(f"{name:20s} {acc:.4f}")

# 任务5：输出表格
print("\n===== 任务5：结果表格 =====")
print("| 模型 | 测试准确率 |")
print("|------|------------|")
for name, acc in results.items():
    print(f"| {name} | {acc:.4f} |")

# 任务6：混淆矩阵 + 错误样本
best_model = "SVM"
y_pred = y_preds[best_model]
cm = confusion_matrix(y_test, y_pred)

print(f"\n===== 任务6：{best_model} 混淆矩阵 =====")
print(cm)

# 保存混淆矩阵
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.title(f"{best_model} Confusion Matrix")
plt.colorbar()
plt.xlabel("Predict")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.close()

# 保存错误样本
errors = np.where(y_pred != y_test)[0]
print("\n错误样本数：", len(errors))

plt.figure(figsize=(10, 4))
for i, idx in enumerate(errors[:5]):
    plt.subplot(1, 5, i+1)
    img = X_test[idx].reshape(8, 8)
    plt.imshow(img, cmap='gray')
    plt.title(f"T:{y_test[idx]}\nP:{y_pred[idx]}")
    plt.axis('off')
plt.savefig("error_samples.png")
plt.close()

print("\n✅ 所有图片已保存到文件夹！")