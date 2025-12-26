import matplotlib.pyplot as plt
import pandas as pd

res_path = "../../results/feat_imp.csv"

df = pd.read_csv(res_path, header=0)
df = df.sort_values(by=["importance"], ascending=False)
print(df)

plt.bar(x=range(len(df["importance"])), height=df["importance"])

plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.grid(True)
plt.show()