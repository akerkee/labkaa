import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler

# Датасэтті жүктейміз
data = pd.read_csv("C:/Users/User/Documents/1heart_failure_clinical_records_dataset.csv")

# Деректерді белгілерге және мақсатты айнымалыға бөлу
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# Деректерді жаттығу және сынақ үлгілеріне бөлу
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Негізгі модельдерді анықтау
models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(kernel='rbf', C=1, probability=True)),
]

# Метамодель құру (логистикалық регрессия)
meta_model = LogisticRegression()

#Стекинг жасау
stacking_model = StackingClassifier(estimators=models, final_estimator=meta_model)

# Оқу деректері бойынша стекингті оқыту
stacking_model.fit(X_train, y_train)

# Сынақ деректерінде стекинг болжамдарын алу
y_pred = stacking_model.predict(X_test)

# Модель өнімділігін бағалау
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Орташа модельдің сынақ деректерінде болжамдарын алу
stacking_y_proba = stacking_model.predict_proba(X_test)[:, 1]

# ROC кривизін құру
fpr, tpr, _ = roc_curve(y_test, stacking_y_proba)
roc_auc = auc(fpr, tpr)

# ROC кривизін графикке салу
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Stacking Model')
plt.legend(loc="lower right")
plt.show()
